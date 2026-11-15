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
