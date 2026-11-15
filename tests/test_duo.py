import json

from forge import bench_b, duo
from tests.test_bench_b import call, repo, scripted, traj  # noqa: F401  (fixtures)


def handoff(**card):
    return call("handoff", **card)


def test_operator_hands_off_and_coder_edits(repo, tmp_path):
    t = traj(repo)
    task = bench_b.build_tasks([t])[0]
    op = scripted(call("Read", file_path="a.py"),
                  handoff(goal="set x to 2 in a.py", done_criteria={"description": "a.py has x = 2"},
                          context={"files": ["a.py"], "findings": ["x = 1 on line 1"]}),
                  call("done", summary="coder changed it"))
    seen = []

    def coder(msgs, tools):
        seen.append([m["content"] for m in msgs if m["role"] == "user"])
        names = {t["function"]["name"] for t in tools}
        assert "handoff" not in names and "Edit" in names
        return coder_script(msgs, tools)

    coder_script = scripted(call("Edit", file_path="a.py", old_string="x = 1", new_string="x = 2"), call("done", summary="edited"))
    r = bench_b.run_task(task, t["steps"], op, root=tmp_path / "w", coder=coder)
    assert r["score"] == 1.0 and r["status"] == "ok"
    assert "set x to 2" in seen[0][0] and "x = 1 on line 1" in seen[0][0]  # the coder only sees the card
    assert r["handoffs"] == 1 and r["steps"] <= 30


def test_operator_cannot_edit(repo, tmp_path):
    t = traj(repo)
    task = bench_b.build_tasks([t])[0]
    op = scripted(call("Edit", file_path="a.py", old_string="x = 1", new_string="x = 2"), call("done"))
    r = bench_b.run_task(task, t["steps"], op, root=tmp_path / "w", coder=scripted(call("done")))
    assert r["score"] < 1.0 and "not available" in json.dumps(r["transcript"])


def test_bad_card_is_returned_to_the_operator(repo, tmp_path):
    t = traj(repo)
    task = bench_b.build_tasks([t])[0]
    op = scripted(handoff(goal=""), call("done"))
    r = bench_b.run_task(task, t["steps"], op, root=tmp_path / "w", coder=scripted(call("done")))
    assert r["handoffs"] == 0 and "invalid card" in json.dumps(r["transcript"])


def test_budget_is_shared(repo, tmp_path):
    t = traj(repo)
    task = bench_b.build_tasks([t])[0]
    card = dict(goal="g", done_criteria={"description": "d"})
    op = scripted(*[handoff(**card)] * 50)
    loop = scripted(*[call("Read", file_path="a.py")] * 200)
    r = bench_b.run_task(task, t["steps"], op, root=tmp_path / "w", coder=loop, max_steps=10)
    assert r["steps"] <= 10
