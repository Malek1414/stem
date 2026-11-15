"""Copy a snapshot to a private Hugging Face dataset repo and prove the copy matches the manifest.

The dataset holds unscrubbed personal data by design, so this refuses to touch a repo that is public."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_REPO = "agent-forge-data"


def publish(snapshot: Path, repo: str | None = None) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    snapshot = Path(snapshot)
    repo_id = repo or f"{api.whoami()['name']}/{DEFAULT_REPO}"
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(repo_id)
    if not info.private:
        raise RuntimeError(f"{repo_id} is public; refusing to upload personal data to it")

    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(snapshot),
                      path_in_repo=snapshot.name, commit_message=f"snapshot {snapshot.name}")

    m = json.loads((snapshot / "manifest.json").read_text())
    want = {o["path"]: o["sha256"] for o in m["outputs"]}
    if m.get("raw"):
        want[m["raw"]["path"]] = m["raw"]["sha256"]
    remote = {}
    for f in api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo=snapshot.name, recursive=True,
                                expand=True):
        lfs = getattr(f, "lfs", None)
        if lfs is not None:
            remote[f.path[len(snapshot.name) + 1:]] = lfs.sha256
    small = [p for p in want if p not in remote]  # small files are stored without LFS: hash them by download
    for p in small:
        local = api.hf_hub_download(repo_id, f"{snapshot.name}/{p}", repo_type="dataset")
        from .extract import sha256_file
        remote[p] = sha256_file(Path(local))
    mismatches = sorted(p for p in want if remote.get(p) != want[p])
    commit = api.dataset_info(repo_id).sha
    return {"repo": repo_id, "private": True, "revision": commit, "checked": len(want), "mismatches": mismatches}
