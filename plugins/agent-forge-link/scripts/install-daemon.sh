#!/usr/bin/env bash
# Make the link daemon permanent: launchd on macOS, systemd --user on Linux.
# usage: install-daemon.sh [name] [port]     (name defaults to your username, port to 47100)
set -e
NAME="${1:-$(whoami)}"; PORT="${2:-47100}"
BIN="$HOME/.agent-forge/bin"; mkdir -p "$BIN" "$HOME/.agent-forge/link"
cp "$(dirname "$0")/link.sh" "$BIN/link.sh"; chmod +x "$BIN/link.sh"
ARGS=(run --name "$NAME" --bind "0.0.0.0:$PORT" --mode relay)
if [[ "$(uname)" == Darwin ]]; then
  PL="$HOME/Library/LaunchAgents/com.agent-forge.link.plist"
  { echo '<?xml version="1.0" encoding="UTF-8"?>'
    echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
    echo '<plist version="1.0"><dict><key>Label</key><string>com.agent-forge.link</string>'
    echo '<key>ProgramArguments</key><array><string>/bin/bash</string>'"<string>$BIN/link.sh</string>"
    for a in "${ARGS[@]}"; do echo "<string>$a</string>"; done
    echo '</array><key>EnvironmentVariables</key><dict><key>PATH</key>'"<string>$PATH</string>"
    [[ -n "$AGENT_FORGE_SRC" ]] && echo "<key>AGENT_FORGE_SRC</key><string>$AGENT_FORGE_SRC</string>"
    echo '</dict><key>RunAtLoad</key><true/><key>KeepAlive</key><true/>'
    echo "<key>StandardOutPath</key><string>$HOME/Library/Logs/agent-forge-link.log</string>"
    echo "<key>StandardErrorPath</key><string>$HOME/Library/Logs/agent-forge-link.log</string></dict></plist>"; } > "$PL"
  launchctl bootout "gui/$(id -u)/com.agent-forge.link" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PL"
  echo "installed $PL (log ~/Library/Logs/agent-forge-link.log)"
else
  U="$HOME/.config/systemd/user"; mkdir -p "$U"
  cat > "$U/agent-forge-link.service" <<UNIT
[Unit]
Description=agent-forge link daemon
[Service]
Environment=PATH=$PATH
${AGENT_FORGE_SRC:+Environment=AGENT_FORGE_SRC=$AGENT_FORGE_SRC}
ExecStart=/bin/bash $BIN/link.sh ${ARGS[*]}
Restart=always
[Install]
WantedBy=default.target
UNIT
  systemctl --user daemon-reload && systemctl --user enable --now agent-forge-link
  echo "installed $U/agent-forge-link.service; run 'sudo loginctl enable-linger $USER' so it survives logout"
fi
