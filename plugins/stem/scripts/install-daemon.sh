#!/usr/bin/env bash
# Make the link daemon permanent: launchd on macOS, systemd --user on Linux.
# usage: install-daemon.sh [name] [port]     (name defaults to your username, port to 47100)
set -e
NAME="${1:-$(whoami)}"; PORT="${2:-47100}"
BIN="$HOME/.stem/bin"; mkdir -p "$BIN" "$HOME/.stem/link"
cp "$(dirname "$0")/link.sh" "$BIN/link.sh"; chmod +x "$BIN/link.sh"
ARGS=(run --name "$NAME" --bind "0.0.0.0:$PORT" --mode relay)
if [[ "$(uname)" == Darwin ]]; then
  PL="$HOME/Library/LaunchAgents/com.stem.link.plist"
  { echo '<?xml version="1.0" encoding="UTF-8"?>'
    echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
    echo '<plist version="1.0"><dict><key>Label</key><string>com.stem.link</string>'
    echo '<key>ProgramArguments</key><array><string>/bin/bash</string>'"<string>$BIN/link.sh</string>"
    for a in "${ARGS[@]}"; do echo "<string>$a</string>"; done
    echo '</array><key>EnvironmentVariables</key><dict><key>PATH</key>'"<string>$PATH</string>"
    [[ -n "$STEM_SRC" ]] && echo "<key>STEM_SRC</key><string>$STEM_SRC</string>"
    echo '</dict><key>RunAtLoad</key><true/><key>KeepAlive</key><true/>'
    echo "<key>StandardOutPath</key><string>$HOME/Library/Logs/stem-link.log</string>"
    echo "<key>StandardErrorPath</key><string>$HOME/Library/Logs/stem-link.log</string></dict></plist>"; } > "$PL"
  launchctl bootout "gui/$(id -u)/com.stem.link" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PL"
  echo "installed $PL (log ~/Library/Logs/stem-link.log)"
else
  U="$HOME/.config/systemd/user"; mkdir -p "$U"
  cat > "$U/stem-link.service" <<UNIT
[Unit]
Description=stem link daemon
[Service]
Environment=PATH=$PATH
${STEM_SRC:+Environment=STEM_SRC=$STEM_SRC}
ExecStart=/bin/bash $BIN/link.sh ${ARGS[*]}
Restart=always
[Install]
WantedBy=default.target
UNIT
  systemctl --user daemon-reload && systemctl --user enable --now stem-link
  echo "installed $U/stem-link.service; run 'sudo loginctl enable-linger $USER' so it survives logout"
fi
