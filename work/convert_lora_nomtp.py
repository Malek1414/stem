"""convert_lora_to_gguf.py for checkpoints without an MTP head (Qwen3.8-27B Heretic: mtp_num_hidden_layers = 0).
The Qwen converter otherwise asserts that MTP layers exist; convert_hf_to_gguf has --no-mtp, the LoRA converter
does not. Usage: python convert_lora_nomtp.py <convert_lora_to_gguf args>"""
import runpy
import sys
from pathlib import Path

LC = Path.home() / "agent-forge" / "llama.cpp"
sys.path[:0] = [str(LC), str(LC / "gguf-py")]
from conversion import base  # noqa: E402

for name in dir(base):
    cls = getattr(base, name)
    if isinstance(cls, type) and "no_mtp" in vars(cls):
        cls.no_mtp = True
sys.argv = [str(LC / "convert_lora_to_gguf.py"), *sys.argv[1:]]
runpy.run_path(str(LC / "convert_lora_to_gguf.py"), run_name="__main__")
