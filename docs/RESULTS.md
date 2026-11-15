# agent-forge results log

## 2026-11-15 · benchmark B · qwen3-coder-next-abliterated-128k · snapshot v0002

Model record: Ollama digest 3e0033bef22e…, Q4_K_M, Ollama 0.24.0, temperature 0, seed 0.
Task: "only have one phone for now adapt" (hoopergpt, 2 Swift files).

| Run | Score | Steps | Seconds | What happened |
|---|---|---|---|---|
| 1 | 0/4 | 15 | 958 | Harness bug: tools were named bash/read_file while the history used Bash/Read. Every call returned "unknown tool". Fixed in commit "Harness: use Claude Code tool names". |
| 2 | 0/4 | 30 (cap) | 886 | Tools now used correctly. Model failure: first Read targeted a temp file from the original session outside the sandbox (refused), then the same `xcodebuild` command was repeated 29 times with no edit. |

Reading: tool use works end to end; the base model loops on long, noisy build output instead of acting.
This is one task, so it says nothing about the overall rate (95% interval for 0/1 is 0 to 0.79).

## 2026-11-15 · publish · snapshot v0002

Private HF dataset `malekhassan23/agent-forge-data`, revision 100981de096317e2126b23b761f7d6700f974a39.
6 files checked against the manifest (5 outputs + raw.tar.gz): 0 mismatches. Repo confirmed private.
