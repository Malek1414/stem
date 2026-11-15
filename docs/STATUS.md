# agent-forge status (read by the hourly watcher)

Updated: 2026-11-15. The watcher does the first unchecked item that does not need Malek's approval,
ticks it here, commits, and stops. Items marked (APPROVAL) wait for Malek; never start them unasked.

## Rules for the watcher

- DGX Spark: only via the existing NVIDIA Sync tunnels (Ollama on localhost:11434). Never plain ssh.
  One Spark job at a time; check nothing else is running on it first (`ps aux | grep "forge bench"`).
- Never use Haiku for anything.
- Do not send email, Slack or anything outward. Do not delete user files.
- Commit each finished step with the attribution lines from the session.

## Queue

- [x] Read results/qwen3-coder-next-abliterated-128k/v0002/bench_b.json (the "one phone" rerun after the
      tool-name fix). Record score, checks, steps, seconds and whether the model now used Bash/Read/Edit
      in docs/RESULTS.md. If it failed for a harness reason, fix the harness, add a test, rerun once.
- [x] Read results/published.jsonl. Confirm v0002 is on the private repo malekhassan23/agent-forge-data
      with zero mismatches; note the revision in docs/RESULTS.md. If the upload failed, rerun
      `uv run --no-project --with huggingface_hub python -m forge publish data/snapshots/v0002` once.
- [ ] Link spike 1 (no install on the Spark, scratch folder only): in a throwaway uv env, check that PyPI
      `iroh` (Python bindings) and `a2a-sdk` install on the Mac and can open one iroh connection between
      two local endpoints and carry one A2A message. Write findings to docs/research/2026-11-15-link-spikes.md.
- [ ] Link spike 2: read code.claude.com/docs/en/cross-session-messaging and the Claude Code Channels docs;
      note whether inbound push into a session works without a developer flag. Same findings file.
- [ ] Link spike 3: measure Mac -> Spark round-trip latency over the existing Ollama tunnel
      (20 tiny `/api/version` calls, median and p95). Same findings file.
- [ ] (APPROVAL) Build the link per the proposed design in the conversation; spec in
      docs/superpowers/specs/2026-11-15-link-design.md. Waits for Malek's yes.
- [ ] (APPROVAL) Decisions from docs/research/2026-11-15-training-and-eval-evidence.md: rotating
      held-out project; operator base from an open computer-use model; human calibration of judge/B rule.

When every non-APPROVAL item is ticked, the watcher deletes its own cron job and leaves a summary.
