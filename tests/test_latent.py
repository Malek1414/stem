"""Latent handoff mechanics on a tiny random Qwen3 (no download, CPU, float32)."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from latent import attn, cache as kvc, latentmas  # noqa: E402
from latent.bench import correct, make_tasks  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
    return transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation=attn.NAME, dtype=torch.float32).eval()


def ids(*xs):
    return torch.tensor([list(xs)])


@torch.no_grad()
def test_kv_prefix_handoff_equals_one_pass(model):
    ctx, ask = ids(*range(5, 25)), ids(30, 31, 32)
    full = model(input_ids=torch.cat([ctx, ask], 1)).logits[:, -1]
    c = model(input_ids=ctx, use_cache=True).past_key_values
    blob = kvc.pack(c, "tiny")
    c2, hdr = kvc.unpack(blob, "tiny", "cpu")
    split = model(input_ids=ask, past_key_values=c2, use_cache=True).logits[:, -1]
    assert hdr["seq_len"] == 20 and torch.allclose(full, split, atol=2e-2)  # fp16 on the wire


@torch.no_grad()
def test_int8_is_close_and_smaller(model):
    c = model(input_ids=ids(*range(5, 45)), use_cache=True).past_key_values
    b16, b8 = kvc.pack(c, "tiny"), kvc.pack(c, "tiny", quant="int8")
    assert len(b8) < 0.65 * len(b16)  # head_dim 16 here; scales cost 1/8, at head_dim 128 int8 is ~0.51x
    c8, _ = kvc.unpack(b8, "tiny", "cpu")
    for (k, v), (k8, v8) in zip(kvc.layers_of(c), kvc.layers_of(c8)):
        assert (k - k8).abs().max() <= k.abs().amax(-1, keepdim=True).max() / 127 + 1e-4


def test_wrong_model_is_refused(model):
    c = model(input_ids=ids(1, 2, 3), use_cache=True).past_key_values
    with pytest.raises(ValueError, match="receiver runs"):
        kvc.unpack(kvc.pack(c, "tiny"), "other", "cpu")


@torch.no_grad()
def test_hidden_sender_equals_no_sender(model):
    """With no layer selected, the receiver must behave as if the operator sent nothing (RoPE is relative)."""
    ctx, ask = ids(*range(5, 25)), ids(30, 31, 32)
    alone = model(input_ids=ask).logits[:, -1]
    c = model(input_ids=ctx, use_cache=True).past_key_values
    with attn.hide_sender(20, set()):
        hidden = model(input_ids=ask, past_key_values=c, use_cache=True).logits[:, -1]
    assert torch.allclose(alone, hidden, atol=1e-4)


@torch.no_grad()
def test_selected_layers_only_travel(model):
    c = model(input_ids=ids(*range(5, 25)), use_cache=True).past_key_values
    blob = kvc.pack(c, "tiny", layers=[1, 2])
    hdr, _ = kvc.read_header(blob)
    assert hdr["layers"] == [1, 2] and len(blob) < 0.6 * len(kvc.pack(c, "tiny"))
    c2, _ = kvc.unpack(blob, "tiny", "cpu")
    assert kvc.layers_of(c2)[0][0].abs().sum() == 0  # missing layer is zero-filled (and must be masked)


@torch.no_grad()
def test_attention_mass_is_recorded(model):
    c = model(input_ids=ids(*range(5, 25)), use_cache=True).past_key_values
    rec = {}
    with attn.hide_sender(20, None, record=rec):
        model(input_ids=ids(30, 31), past_key_values=c, use_cache=True)
    assert set(rec) == {0, 1, 2, 3} and all(0 <= v <= 1 for v in rec.values())


@torch.no_grad()
def test_latent_thoughts_grow_the_cache(model):
    out = model(input_ids=ids(*range(5, 15)), use_cache=True, output_hidden_states=True)
    w_a, norm = latentmas.alignment(model)
    c, h = latentmas.think(model, out.past_key_values, out.hidden_states[-1][:, -1:], 6, w_a, norm)
    assert kvc.seq_len(c) == 16 and h.shape == (1, 1, 64)


def test_tasks_are_deterministic_and_answerable():
    a, b = make_tasks(9, 1), make_tasks(9, 1)
    assert a == b and a != make_tasks(9, 2)
    for t in a:
        assert t["answer"] in t["context"]
    assert correct("It was `parse_order`.", "parse_order") and not correct("parse_orders", "parse_order")
