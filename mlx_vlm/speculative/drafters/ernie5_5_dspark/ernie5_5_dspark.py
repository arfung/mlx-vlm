"""ERNIE 5.5 Flash DSpark speculative drafter.

ERNIE 5.5 Flash Preview bundles a DSpark draft head under ``model.mtp.*``:
three ERNIE decoder layers that attend over a sliding window only, a
projection of the target's last three layer outputs, a low-rank Markov head
and a final hyper-connection contraction. It reuses the shared DSpark
machinery of the DeepSeek-V4 drafter (``VanillaMarkov`` block sampling and the
``dflash`` round loop) and swaps in ERNIE layers.

Reference: https://github.com/baidu/vLLM-ERNIE (``models/ernie5_5/nvidia/dspark.py``).
"""

import re
from types import SimpleNamespace
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from ....models.base import scaled_dot_product_attention
from ....models.cache import RotatingKVCache
from ....models.deepseek_v4.hyper_connection import HyperHead, hc_expand
from ....models.deepseek_v4.language import LocalAttention
from ....models.ernie5_5_moe.language import (
    HyperConnection,
    LanguageModel,
    LatentMoE,
    VirtualHeads,
)
from ..deepseek_v4_dspark.deepseek_v4_dspark import DeepseekV4DsparkDraftModel

# Draft-level tensors named outside the per-layer ERNIE layout.
_HEAD_RENAMES = {
    "model.feature_projection.": "stages.0.main_proj.",
    "model.feature_norm.": "stages.0.main_norm.",
    "model.markov_head.": "markov_head.",
}


class DSparkAttention(LocalAttention):
    """ERNIE window attention in the DFlash cross-attention layout.

    The projected target hidden states are the accumulating KV context; the
    drafted block supplies the queries and a transient block KV. As in the
    reference, draft layers use plain RoPE at ``rope_theta`` and no compression.
    """

    def __init__(self, config):
        super().__init__(config, 0)
        self.o_gate = nn.Linear(
            config.q_lora_rank, config.o_groups * config.o_lora_rank, bias=False
        )
        self.vha = VirtualHeads(self.n_heads, config.vha_rank)

    def __call__(self, x: mx.array, x_ctx: mx.array, cache: RotatingKVCache):
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        offset = cache.offset

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps).transpose(0, 2, 1, 3)
        q = self.rope(q, offset + S)

        ctx_kv = self.rope(self.kv_norm(self.wkv(x_ctx))[:, None], offset)
        block_kv = self.rope(self.kv_norm(self.wkv(x))[:, None], offset + S)
        kv, _ = cache.update_and_fetch(
            ctx_kv, mx.zeros((B, 1, S, 0), dtype=ctx_kv.dtype)
        )
        kv = mx.concatenate([kv, block_kv], axis=2)

        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=None,
            scale=self.scale,
            mask=None,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = self.rope(out, offset + S, inverse=True).transpose(0, 2, 1, 3)
        out = self.vha(out).reshape(B, L, self.o_groups, -1).transpose(0, 2, 1, 3)
        out = self.wo_a(out).transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(out * mx.sigmoid(self.o_gate(qr)))


class DSparkStage(nn.Module):
    """One draft stage: an ERNIE decoder layer whose attention reads the
    projected target hidden. Parameter names match the ERNIE layer so the
    checkpoint maps straight in."""

    def __init__(self, config, stage_id: int):
        super().__init__()
        text = config.text_config
        self.attn = DSparkAttention(text)
        self.ffn = LatentMoE(text)
        self.attn_norm = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.attn_hc = HyperConnection(text)
        self.ffn_hc = HyperConnection(text)
        if stage_id == 0:
            targets = max(len(config.target_layer_ids), 1)
            self.main_proj = nn.Linear(
                text.hidden_size * targets, text.hidden_size, bias=False
            )
            self.main_norm = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        if stage_id == config.n_mtp_layers - 1:
            self.hc_head = HyperHead(text)
            self.norm = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)

    def __call__(self, h: mx.array, main_x: mx.array, cache: RotatingKVCache):
        x, post, comb = self.attn_hc(h)
        x = self.attn(self.attn_norm(x), main_x, cache)
        h = hc_expand(x, h, post, comb)
        x, post, comb = self.ffn_hc(h)
        x = self.ffn(self.ffn_norm(x))
        return hc_expand(x, h, post, comb)


class Ernie55DsparkDraftModel(DeepseekV4DsparkDraftModel):
    stage_class = DSparkStage

    def __init__(self, config):
        super().__init__(config)
        self._multimax = None

    def validate_target_compatibility(self, target_model):
        target = getattr(target_model, "language_model", target_model)
        config = target.config
        own = self.args
        fields = ("hidden_size", "vocab_size", "num_hidden_layers", "hc_mult")
        if config.model_type != "ernie5_5_moe" or any(
            getattr(config, k) != getattr(own, k) for k in fields
        ):
            raise ValueError("DSpark drafter is incompatible with the target model")
        if max(self.config.target_layer_ids) >= config.num_hidden_layers:
            raise ValueError("DSpark target layers are outside the target model")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        lm = getattr(target_model, "language_model", target_model)
        self._input_embed = lm.model.embed_tokens
        self._lm_head_fn = lm.lm_head
        self._multimax = lm.multimax
        return self

    def _logits(self, hidden: mx.array) -> mx.array:
        # The target projects in fp32. MultiMax follows the Markov bias.
        return self._lm_head_fn(hidden.astype(mx.float32))

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        if self._multimax is None:
            raise RuntimeError(
                "bind(target_model) must be called before draft_block()."
            )
        multimax = self._multimax
        return super().draft_block(
            last_bonus,
            hidden,
            cache,
            block_size,
            lambda logits: sampler(multimax(logits)),
            token_dtype,
        )

    def sanitize(self, weights: dict) -> dict:
        """Map draft tensors in the ERNIE layer layout onto the drafter.

        ``self`` only needs ``args`` (the ERNIE text config), so the splitter
        can call this without building the model. Expects the output of
        ``split.unbundle`` after ``prepare_preview_weights`` has repacked it.
        Weights that are already in the drafter layout pass through.
        """
        if not any(key.startswith("model.") for key in weights):
            return weights
        args = self.args
        renamed = {
            key: value
            for key, value in weights.items()
            # The confidence head is unused by the dflash round loop.
            if not key.startswith("model.confidence_head.")
        }
        stages = {
            int(m[1])
            for key in renamed
            if (m := re.match(r"model\.layers\.(\d+)\.", key))
        }
        if not stages:
            raise ValueError("No DSpark stage tensors found")
        last = max(stages)

        # Reuse the backbone's layer mapping on a config that has only the
        # draft stages, all of them non-MLA.
        context = SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=last + 1,
                o_groups=args.o_groups,
                o_lora_rank=args.o_lora_rank,
                n_routed_experts=args.n_routed_experts,
                compress_ratios=[0] * (last + 1),
            )
        )
        mapped = LanguageModel.sanitize(context, renamed)

        out = {}
        for key, value in mapped.items():
            if key == "model.markov_head.markov_w1.weight":
                # Stored as [rank, vocab]; the embedding wants [vocab, rank].
                value = value.T
            if key.startswith("model.layers."):
                key = "stages." + key.removeprefix("model.layers.")
            elif key.startswith("model.hc_head."):
                key = f"stages.{last}.hc_head." + key.removeprefix("model.hc_head.")
            elif key.startswith("model.norm."):
                key = f"stages.{last}.norm." + key.removeprefix("model.norm.")
            else:
                for src, dst in _HEAD_RENAMES.items():
                    if key.startswith(src):
                        key = dst + key.removeprefix(src)
                        break
                else:
                    raise ValueError(f"Unexpected DSpark tensor: {key}")
            out[key] = value
        return out


Model = Ernie55DsparkDraftModel
