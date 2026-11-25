"""ERNIE 5.5 Flash DSpark: target captures, checkpoint mapping and round loop."""

from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_vlm.models.ernie5_5_moe import Model, ModelConfig
from mlx_vlm.speculative.dflash import _dflash_rounds
from mlx_vlm.speculative.drafters import (
    DRAFTER_KIND_BY_MODEL_TYPE,
    ernie5_5_dspark,
)
from mlx_vlm.speculative.drafters.ernie5_5_dspark.split import (
    Ernie55DsparkSplitter,
    unbundle,
)
from mlx_vlm.speculative.drafters.ernie5_5_mtp.split import Ernie55MTPSplitter
from mlx_vlm.tests.test_models import model_case


def _randomize(module, seed):
    """Float32 random parameters, so no two expert scores or logits tie."""
    mx.random.seed(seed)
    params = []
    for name, value in tree_flatten(module.parameters()):
        if "norm" in name and name.endswith("weight"):
            params.append((name, mx.ones(value.shape, dtype=mx.float32)))
        else:
            params.append((name, mx.random.normal(value.shape, dtype=mx.float32) * 0.1))
    module.update(tree_unflatten(params))
    mx.eval(module.parameters())


def _target():
    config = dict(model_case("ernie5_5_moe")["config"])
    model = Model(ModelConfig.from_dict(config))
    _randomize(model, 0)
    return config, model


def _drafter(config, block_size=4):
    draft_config = ernie5_5_dspark.ModelConfig.from_dict(
        {
            "text_config": config,
            "n_mtp_layers": 2,
            "target_layer_ids": [0, 1],
            "mask_token_id": 3,
            "markov_rank": 8,
            "block_size": block_size,
        }
    )
    draft = ernie5_5_dspark.Model(draft_config)
    _randomize(draft, 1)
    return draft


def _greedy(logits):
    return mx.argmax(logits, axis=-1)


def test_capture_layer_ids_returns_stream_means():
    _, model = _target()
    lm = model.language_model
    tokens = mx.array([[5, 9, 2, 44, 7, 81]])

    captured = lm(tokens, cache=model.make_cache(), capture_layer_ids=[0, 1])
    assert [h.shape for h in captured.hidden_states] == [(1, 6, 64)] * 2

    final = lm(tokens, cache=model.make_cache(), return_hidden=True).hidden_states[0]
    streams = final.reshape(1, 6, lm.config.hc_mult, 64)
    assert mx.allclose(captured.hidden_states[1], streams.mean(axis=2), atol=1e-5)
    # The logits are unchanged by capturing.
    plain = lm(tokens, cache=model.make_cache())
    assert mx.allclose(captured.logits, plain.logits, atol=1e-5)


def test_preview_config_selects_the_dspark_splitter():
    assert DRAFTER_KIND_BY_MODEL_TYPE["ernie5_5_dspark"] == "dflash"
    base = {"model_type": "ernie5_5_moe"}
    assert type(Ernie55MTPSplitter.from_config(base)) is Ernie55MTPSplitter
    preview = {**base, "dspark_target_layer_ids": [40, 41, 42], "dspark_block_size": 7}
    splitter = Ernie55MTPSplitter.from_config(preview)
    assert type(splitter) is Ernie55DsparkSplitter
    assert splitter.select_keys("model.mtp.0.norm.weight", preview)
    assert not splitter.select_keys("model.layers.0.norm.weight", preview)
    assert splitter.extra_config(
        {**preview, "dspark_noise_token_id": 1024, "dspark_markov_rank": 256}
    ) == {
        "n_mtp_layers": 3,
        "target_layer_ids": [40, 41, 42],
        "mask_token_id": 1024,
        "markov_rank": 256,
        "block_size": 8,
    }


def test_bundled_tensors_map_onto_the_drafter():
    hidden, hc, experts, vocab, rank = 8, 4, 2, 16, 4
    stage = "model.mtp.1."
    source = {
        "model.mtp.0.self_attn.linear_q_down_proj.weight": mx.zeros((4, hidden)),
        "model.mtp.0.self_attn.vha_postmix_U": mx.zeros((2, 2)),
        "model.mtp.0.self_attn.core_attention.attn_sink": mx.zeros((2,)),
        "model.mtp.0.self_attn.gate_proj.weight": mx.zeros((8, 4)),
        "model.mtp.0.self_attn.linear_o_group_proj": mx.zeros((2 * 3, 6)),
        "model.mtp.0.self_attention_hyper_connection.mapping_proj.weight": mx.zeros(
            (hc * hidden, 24)
        ),
        "model.mtp.0.self_attention_hyper_connection.bias": mx.zeros((24,)),
        "model.mtp.0.self_attention_hyper_connection.alpha_pre": mx.array([1.0]),
        "model.mtp.0.self_attention_hyper_connection.alpha_post": mx.array([2.0]),
        "model.mtp.0.self_attention_hyper_connection.alpha_res": mx.array([3.0]),
        stage + "block_sparse_moe.gate.weight_1": mx.zeros((experts, hidden)),
        stage + "block_sparse_moe.shared_experts.w2.weight": mx.zeros((hidden, 4)),
        "model.mtp.feature_projection.weight": mx.zeros((hidden, 3 * hidden)),
        "model.mtp.feature_norm.weight": mx.ones((hidden,)),
        "model.mtp.markov_head.markov_w1.weight": mx.zeros((rank, vocab)),
        "model.mtp.markov_head.markov_w2.weight": mx.zeros((vocab, rank)),
        "model.mtp.mhc_contract.hc_head_fn": mx.zeros((hc * hidden, hc)),
        "model.mtp.mhc_contract.hc_head_base": mx.zeros((hc,)),
        "model.mtp.norm.weight": mx.ones((hidden,)),
        "model.mtp.confidence_head.projection.weight": mx.zeros((1, hidden + rank)),
    }
    for expert in range(experts):
        source[stage + f"block_sparse_moe.experts.{expert}.w1.weight"] = mx.full(
            (4, 4), float(expert)
        )

    context = SimpleNamespace(
        args=SimpleNamespace(o_groups=2, o_lora_rank=3, n_routed_experts=experts)
    )
    out = ernie5_5_dspark.Model.sanitize(context, unbundle(source))

    shapes = {key: value.shape for key, value in out.items()}
    assert shapes == {
        "stages.0.attn.wq_a.weight": (4, hidden),
        "stages.0.attn.vha.U": (2, 2),
        "stages.0.attn.attn_sink": (2,),
        "stages.0.attn.o_gate.weight": (8, 4),
        "stages.0.attn.wo_a.weight": (2, 3, 6),
        "stages.0.attn_hc.fn": (24, hc * hidden),
        "stages.0.attn_hc.base": (24,),
        "stages.0.attn_hc.scale": (3,),
        "stages.1.ffn.gate.weight_1": (experts, hidden),
        "stages.1.ffn.shared_experts.down_proj.weight": (hidden, 4),
        "stages.1.ffn.switch_mlp.gate_proj.weight": (experts, 4, 4),
        "stages.0.main_proj.weight": (hidden, 3 * hidden),
        "stages.0.main_norm.weight": (hidden,),
        "markov_head.markov_w1.weight": (vocab, rank),
        "markov_head.markov_w2.weight": (vocab, rank),
        "stages.1.hc_head.fn": (hc, hc * hidden),
        "stages.1.hc_head.base": (hc,),
        "stages.1.norm.weight": (hidden,),
    }
    assert out["stages.0.attn_hc.scale"].tolist() == [1.0, 2.0, 3.0]
    assert out["stages.1.ffn.switch_mlp.gate_proj.weight"][1, 0, 0].item() == 1.0
    # Already-mapped weights pass through unchanged.
    assert ernie5_5_dspark.Model.sanitize(context, out) is out


def _decode_one_by_one(model, prompt, count):
    lm = model.language_model
    cache = model.make_cache()
    token = _greedy(lm(mx.array([prompt]), cache=cache).logits[:, -1])
    tokens = []
    for _ in range(count):
        tokens.append(token.item())
        token = _greedy(lm(token[:, None], cache=cache).logits[:, -1])
    return tokens


def _decode_speculative(model, draft, prompt, count, block_size):
    lm = model.language_model
    cache = model.make_cache()
    output = lm(
        mx.array([prompt]),
        cache=cache,
        capture_layer_ids=list(draft.config.target_layer_ids),
    )
    hidden = mx.concatenate(output.hidden_states, axis=-1)
    tokens = [_greedy(output.logits[:, -1]).item()]
    for token, _ in _dflash_rounds(
        model,
        draft,
        cache,
        hidden,
        first_bonus=tokens[0],
        max_tokens=count,
        sampler=_greedy,
        draft_block_size=block_size,
        greedy_sampling=True,
        use_model_initial_block_size=False,
    ):
        tokens.append(int(token))
    return tokens


def test_dspark_rounds_reproduce_greedy_decoding():
    # Thirty tokens cross the tiny model's window (4) and compression ratio (8)
    # several times, so rejected drafts roll both caches back.
    config, model = _target()
    prompt = [5, 9, 2, 44, 7, 81]
    reference = _decode_one_by_one(model, prompt, 30)
    for block_size in (2, 4):
        draft = _drafter(config, block_size)
        tokens = _decode_speculative(model, draft, prompt, 30, block_size)
        assert tokens == reference
        assert len(draft.accept_lens) > 0
        assert all(0 <= n <= block_size - 1 for n in draft.accept_lens)


def test_drafter_rejects_a_mismatched_target():
    config, model = _target()
    other = dict(config, hidden_size=32)
    draft = _drafter(other)
    try:
        draft.bind(model)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("expected an incompatible target to be rejected")
