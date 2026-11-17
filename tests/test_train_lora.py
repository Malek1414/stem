"""LoRA training on a tiny random Qwen3 with the real Qwen3 tokenizer (cached, offline)."""
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
from forge import train_lora  # noqa: E402

TOK = next(iter(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/")), None)
pytestmark = pytest.mark.skipif(TOK is None, reason="Qwen3 tokenizer not cached")
EX = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "list files"},
                   {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "Bash", "arguments": "{\"command\": \"ls\"}"}}]},
                   {"role": "tool", "tool_call_id": "c1", "content": "a.py b.py"},
                   {"role": "assistant", "content": "Two files: a.py and b.py."}],
      "tools": [{"type": "function", "function": {"name": "Bash", "description": "shell",
                 "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]}


def test_labels_cover_only_assistant_turns():
    tok = transformers.AutoTokenizer.from_pretrained(TOK)
    e = train_lora.encode(tok, EX, 4096)
    learned = tok.decode([t for t, l in zip(e["input_ids"], e["labels"]) if l != -100])
    assert "Two files: a.py and b.py." in learned and '"ls"' in learned and learned.count("<|im_end|>") == 2
    assert "list files" not in learned and "a.py b.py" not in learned and "sys" not in learned


def test_two_steps_train_and_save(tmp_path):
    os.environ["HF_HUB_OFFLINE"] = "1"
    tok = transformers.AutoTokenizer.from_pretrained(TOK)
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024)
    mdir = tmp_path / "m"
    transformers.AutoModelForCausalLM.from_config(cfg).save_pretrained(mdir)
    tok.save_pretrained(mdir)
    data = tmp_path / "d.jsonl"
    data.write_text("".join(json.dumps(EX) + "\n" for _ in range(8)))
    res = train_lora.main(["--model", str(mdir), "--data", str(data), "--out", str(tmp_path / "out"), "--max-len", "512",
                           "--accum", "2", "--save-every", "1", "--device", "cpu", "--lr", "1e-2", "--warmup", "1"])
    assert res["steps"] == 4 and res["final_eval_loss"] < res["first_eval_loss"]
    assert (tmp_path / "out" / "adapter_model.safetensors").exists()
    log = [json.loads(x) for x in (tmp_path / "out" / "log.jsonl").read_text().splitlines()]
    assert log[0]["step"] == 0 and log[-1]["step"] == 4


def _tiny(tmp_path):
    cfg = transformers.Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
                                   tie_word_embeddings=False)
    return transformers.AutoModelForCausalLM.from_config(cfg)


def test_load_model_matches_from_pretrained(tmp_path):
    ref = _tiny(tmp_path)
    ref.save_pretrained(tmp_path / "m")
    got = train_lora.load_model(tmp_path / "m", torch.float32, "cpu")
    for (k, a), (k2, b) in zip(sorted(ref.state_dict().items()), sorted(got.state_dict().items())):
        assert k == k2 and torch.equal(a, b), k


def test_load_model_maps_multimodal_checkpoint_names(tmp_path):
    """Qwen3.5/3.6 checkpoints name the text weights model.language_model.* and carry a vision tower; the causal-LM
    class names them model.* and has no vision tower."""
    from safetensors.torch import save_file
    ref = _tiny(tmp_path)
    ref.save_pretrained(tmp_path / "m")
    sd = {k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k: v.contiguous()
          for k, v in ref.state_dict().items()}
    sd["model.visual.patch_embed.weight"] = torch.zeros(2, 2)
    (tmp_path / "m" / "model.safetensors").unlink()
    save_file(sd, str(tmp_path / "m" / "model.safetensors"))
    got = train_lora.load_model(tmp_path / "m", torch.float32, "cpu")
    assert all(torch.equal(ref.state_dict()[k], v) for k, v in got.state_dict().items())


def test_loss_from_masks_the_replayed_history():
    from transformers import AutoTokenizer
    import pytest
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception:
        pytest.skip("no local Qwen tokenizer")
    from forge.train_lora import encode
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "OLDREPLY"},
            {"role": "user", "content": "b"}, {"role": "assistant", "content": "NEWREPLY"}]
    e = encode(tok, {"messages": msgs, "loss_from": 2}, 512)
    learned = tok.decode([i for i, l in zip(e["input_ids"], e["labels"]) if l != -100])
    assert "NEWREPLY" in learned and "OLDREPLY" not in learned


def test_tool_call_arguments_become_dicts_for_the_template():
    from forge.train_lora import dict_args
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}},
                                                                  {"id": "2", "function": {"name": "X", "arguments": "not json"}}]}]
    out = dict_args(msgs)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert out[0]["tool_calls"][1]["function"]["arguments"] == {"input": "not json"}
    assert isinstance(msgs[0]["tool_calls"][0]["function"]["arguments"], str)  # input untouched


def test_sparse_loss_equals_the_models_own_loss():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from forge.train_lora import sparse_loss
    cfg = AutoConfig.for_model("qwen2", vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, num_key_value_heads=1)
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg)
    ids = torch.randint(0, 50, (1, 12))
    lab = ids.clone()
    lab[0, :5] = -100
    lab[0, 8] = -100
    assert torch.allclose(sparse_loss(m, ids, lab), m(input_ids=ids, labels=lab).loss, atol=1e-5)


class _Tok:
    """Fake tokenizer: one token per character, chat template = role:content lines."""
    def apply_chat_template(self, msgs, tools=None, tokenize=False):
        return "".join(f"<|im_start|>{m['role']}\\n{m.get('content') or ''}<|im_end|>\\n" for m in msgs)

    def __call__(self, text, **kw):
        return {"input_ids": list(range(len(text)))}


def test_fit_drops_the_oldest_history_and_keeps_the_new_turns():
    from forge.train_lora import fit
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 200}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "B", "arguments": {}}}]},
            {"role": "tool", "content": "y" * 200}, {"role": "assistant", "content": "old"},
            {"role": "user", "content": "now"}, {"role": "assistant", "content": "NEW"}]
    ex = fit(_Tok(), {"messages": msgs, "loss_from": 6}, 150)
    m = ex["messages"]
    assert m[0]["content"] == "sys" and m[-1]["content"] == "NEW" and m[ex["loss_from"] - 1]["content"] == "now"
    assert all(x["role"] != "tool" for x in m[1:2])  # no orphan tool result right after the system prompt
    assert len(m) < len(msgs)
