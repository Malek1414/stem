#!/usr/bin/env bash
# agent-forge link CLI. Uses a local checkout when AGENT_FORGE_SRC points at one (with .venv), otherwise the
# GitHub version through uvx. For `mcp`, starts a background daemon first if none is running.
REPO="${AGENT_FORGE_REPO:-git+https://github.com/Malek1414/agent-forge}"
if [[ -n "$AGENT_FORGE_SRC" && -x "$AGENT_FORGE_SRC/.venv/bin/python" ]]; then
  export PYTHONPATH="$AGENT_FORGE_SRC"
  CMD=("$AGENT_FORGE_SRC/.venv/bin/python" -m link)
else
  command -v uvx >/dev/null || { echo "agent-forge-link needs uv (https://docs.astral.sh/uv/)" >&2; exit 1; }
  CMD=(uvx -q --from "$REPO" agent-forge-link)
fi
if [[ "$1" == mcp ]] && ! "${CMD[@]}" status >/dev/null 2>&1; then
  H="${AGENT_FORGE_LINK_HOME:-$HOME/.agent-forge/link}"
  mkdir -p "$H"
  # relay mode works behind NAT and VPNs; iroh upgrades to a direct path when it can
  nohup "${CMD[@]}" run --mode relay --name "${AGENT_FORGE_LINK_NAME:-$(whoami)}" >> "$H/daemon.log" 2>&1 &
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do sleep 1; "${CMD[@]}" status >/dev/null 2>&1 && break; done
fi
exec "${CMD[@]}" "$@"
