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
| Held-out | One whole project, drawn at random with a recorded seed. A session belongs to the git repo whose files it touched most (edits count triple), not the folder it started in. Eligible: at least 5 sessions, at least 3 of them editing code, both coder and operator steps. v0002 drew **hoopergpt** from McCODZ, Neurotree, hoopergpt, nightcrawler. Everything else trains. |
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
   - A session with no coder or operator steps at all is one operator segment (the operator is the general tool caller).
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
6. **harness** (`forge/harness.py`): the agent loop used by B and C. Tools: bash, read_file,
   write_file, edit_file, done. Every shell command runs under macOS `sandbox-exec` with no network and
   no writes outside the work folder; the file tools refuse paths outside it. Safety lives here because
   the model under test may be abliterated.
7. **bench_b** (`forge/bench_b.py`): a task is one user request that edited the held-out repo. The repo
   is cloned (`--shared`, the original is never touched) at the last commit before the request, the
   model gets the conversation so far plus the request, and automatic checks score it: `changed:<file>`,
   `content:<file>` (line similarity >= 0.75 to the state Claude's edits produced, replayed on the
   committed file), `valid:<file>` (.py compiles, .json parses). Tasks are frozen per snapshot in
   `results/bench_b_tasks/`. v0002: 12 tasks.
8. **bench_c** (`forge/bench_c.py`): the other requests (research, outreach, planning). The model works
   in the same sandbox; a pinned local judge (default `gpt-oss:120b`, digest recorded) grades
   understood / approach / correctness / completeness / overall from 1 to 5 against what really
   happened, with a fixed rubric whose sha256 is recorded. Seeded sample of 20, frozen per snapshot.
   Limitation: no network or apps, so operator requests are judged on approach, not effect.
9. **publish** (`forge/publish.py`): uploads a snapshot to the private HF dataset repo
   `malekhassan23/agent-forge-data`, refuses if the repo is public, and checks every remote file's
   sha256 against the manifest. Each publish is logged in `results/published.jsonl`.

## Reproducibility contract

A result is reproducible when these are all recorded and available: snapshot id + manifest hash,
model weights digest (Ollama manifest digest or safetensors sha256), runtime + version + quantization,
decoding settings, benchmark code commit. `python -m forge verify <snapshot>` re-hashes everything.

## Testing

Unit tests per module on small fixture sessions (`tests/fixtures/`): extraction preserves every block,
segmenting labels known tools correctly, the split is identical for the same seed, snapshots are
byte-identical across two runs, the bench A scorer matches equivalent calls and rejects different ones.

## Out of scope here

The link, model choice, training, the permission outcome (requested 15 Nov 2026).
