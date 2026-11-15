# agent-forge, sub-project 1: dataset and benchmark

Date: 2026-11-15. Status: approved by interview, build started.

## Goal

Turn Malek's Claude Code sessions into (a) a versioned, hashed training dataset for two agents, the
**coder** and the **operator**, and (b) a benchmark that measures any model on Malek's own workflows,
reproducibly, even after the model is uninstalled.

Order of the whole project: 1 dataset + benchmark, 2 the link, 3 choosing operator and coder base
models, 4 training runs.

## Decisions (from the interview)

| Topic | Decision |
|---|---|
| Sessions | All sessions, split by agent. Automated runs excluded (entrypoint `sdk-py` / `sdk-cli`, claude-mem observer, Entropy autopilot). No scrubbing: personal data stays in. Consequence: dataset and weights stay private forever. |
| Routing | Per segment. Each session is cut into stretches by tool type; coder stretches train the coder, operator stretches train the operator, the boundaries become handoff examples for the link. |
| Training targets | Anthropic's terms forbid training competing models on Claude outputs without permission (no non-commercial exception). Permission requested from support@anthropic.com. Until granted, the **default is the fallback**: Claude's assistant turns are kept in the dataset but flagged `claude_output=true` and are not used as targets; targets come from open models attempting the tasks and passing outcome checks. Evaluating against Claude's actions (benchmark A) is allowed. |
| Held-out | One whole project, drawn at random with a recorded seed from the eligible projects (git history, at least 5 sessions, both coder and operator steps). Everything else trains. |
| Benchmarks | A: next-action prediction. B: task replay with fully automatic checks. C: judged outcomes with a pinned local judge model (weights hash, greedy decoding, fixed rubric). |
| Refresh | Versioned snapshots. New sessions keep flowing in; each training run points at exactly one snapshot. |
| Storage | Local copy plus a private Hugging Face repo, both verified against the same hashes. |

## Data facts (15 Nov 2026)

- `~/.claude/projects`: 1,530 session files (2.0 GB), including 321 subagent files under `<session>/subagents/`.
- 502 main sessions over 100 KB; about 30k tool calls (Bash 12k, MCP 6.6k, Read 1.7k, Write 675, Edit 367).
- 194 of them ran in a git repo, 284 did not, 24 folders are gone.
- Record types that matter: `user` (prompt is a string; tool results are `tool_result` blocks),
  `assistant` (`text`, `thinking`, `tool_use` blocks). Each record has `cwd`, `gitBranch`, `timestamp`,
  `entrypoint`, `isSidechain`.

## Components

Each is one module in `forge/` with one job and a CLI subcommand (`python -m forge <cmd>`).

1. **extract** (`forge/extract.py`): one session file in, one normalized trajectory out:
   `meta` (session id, file, sha256, cwd, project, git root, branch, start/end time, automated,
   subagent, parent session) and `steps` (ordered: `user_prompt`, `assistant_text`, `thinking`,
   `tool_call` {id, name, input}, `tool_result` {id, output, is_error}). Nothing is dropped or truncated.
2. **segment** (`forge/segment.py`): labels each tool call `coder`, `operator` or `neutral`, merges
   neutral steps into the neighbouring stretch, and emits segments plus handoff points.
   - coder: Edit, Write, MultiEdit, NotebookEdit, and Bash running tests, builds, package managers or git.
   - operator: computer-use, peekaboo, claude-in-chrome, chrome-devtools, social-chrome, blenderwright,
     phone tools, and Bash running `osascript`, `open`, `screencapture`.
   - neutral: everything else (Read, Grep, Glob, ToolSearch, WebSearch, Agent, ...).
3. **split** (`forge/split.py`): projects, eligibility, seeded random draw of the held-out project.
4. **snapshot** (`forge/snapshot.py`): builds `data/snapshots/vNNNN/` with `train/coder.jsonl`,
   `train/operator.jsonl`, `train/handoffs.jsonl`, `heldout/trajectories.jsonl`, a raw-source archive
   (`raw.tar.gz`) and `manifest.json` (source file hashes, output hashes, seed, held-out project, code
   commit, counts). Re-running with the same sources, seed and code gives byte-identical outputs.
5. **bench_a** (`forge/bench_a.py`): from the held-out project, one item per tool call: context up to
   that step plus the gold call. Scorer: tool name exact, arguments compared after normalization
   (whitespace, key order, path expansion; Bash compared by command tokens). Runner talks to any
   OpenAI- or Anthropic-compatible endpoint (Ollama on the Spark) at temperature 0 with a fixed seed and
   writes `results/<model>/<snapshot>/bench_a.json` with the model's weights digest.
6. **bench_b** (later in this sub-project): held-out sessions in git repos; restore the commit at
   session start (`git rev-list -1 --before=<start>`), replay the first prompt in a throwaway worktree,
   checks generated automatically from the session's end state (tests run, files written, build passes).
7. **bench_c** (later in this sub-project): sandboxed reruns graded by a pinned local judge.

## Reproducibility contract

A result is reproducible when these are all recorded and available: snapshot id + manifest hash,
model weights digest (Ollama manifest digest or safetensors sha256), runtime + version + quantization,
decoding settings, benchmark code commit. `python -m forge verify <snapshot>` re-hashes everything.

## Testing

Unit tests per module on small fixture sessions (`tests/fixtures/`): extraction preserves every block,
segmenting labels known tools correctly, the split is identical for the same seed, snapshots are
byte-identical across two runs, the bench A scorer matches equivalent calls and rejects different ones.

## Out of scope here

The link, model choice, training, the permission outcome. Benchmark B and C ship after A.
