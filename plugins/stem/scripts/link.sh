#!/usr/bin/env bash
# stem link CLI. Uses a local checkout when STEM_SRC points at one (with .venv), otherwise the
# GitHub version through uvx. For `mcp`, starts a background daemon first if none is running.
REPO="${STEM_REPO:-git+https://github.com/Malek1414/stem}"
if [[ -n "$STEM_SRC" && -x "$STEM_SRC/.venv/bin/python" ]]; then
  export PYTHONPATH="$STEM_SRC"
  CMD=("$STEM_SRC/.venv/bin/python" -m link)
else
  command -v uvx >/dev/null || { echo "stem needs uv (https://docs.astral.sh/uv/)" >&2; exit 1; }
  CMD=(uvx -q --from "$REPO" stem)
fi
if [[ "$1" == mcp ]] && ! "${CMD[@]}" status >/dev/null 2>&1; then
  H="${STEM_LINK_HOME:-$HOME/.stem/link}"
  mkdir -p "$H"
  # relay mode works behind NAT and VPNs; iroh upgrades to a direct path when it can
  nohup "${CMD[@]}" run --mode relay --name "${STEM_LINK_NAME:-$(whoami)}" >> "$H/daemon.log" 2>&1 &
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do sleep 1; "${CMD[@]}" status >/dev/null 2>&1 && break; done
fi
exec "${CMD[@]}" "$@"
