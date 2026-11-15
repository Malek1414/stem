---
name: link-setup
description: Set up or repair the agent-forge link daemon on this machine - make it start at login, choose the node name and port, check it is reachable. Triggers on "set up the link", "install the link daemon", "link not running", "make the link permanent".
---

# Set up the link daemon

1. Needs `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`) and, while the repo is private, GitHub access
   to Malek1414/agent-forge (`gh auth login` then `gh auth setup-git`).
2. Ask the user for a node name (default: their username) and confirm the port (default 47100).
3. Run `bash "${CLAUDE_PLUGIN_ROOT}/scripts/install-daemon.sh" <name> <port>`.
   macOS: a LaunchAgent `com.agent-forge.link`. Linux: a systemd user unit; tell the user to run
   `sudo loginctl enable-linger $USER` so it survives logout (they must run sudo themselves).
4. Check with `link_status()`; it should show the name and `mode: relay`.
5. To connect, see the `link` skill (invite / join).

Developers with a checkout can set `AGENT_FORGE_SRC=/path/to/agent-forge` (with a `.venv`) before installing,
so the daemon runs their local code instead of the GitHub version.
