# Speed of the 27B server on two agent-like prompts (fresh text, and an edit that copies a file); run on the Spark.
import json, time, urllib.request, sys
port = sys.argv[1]
code = open('/home/malekhassan/agent-forge/forge_snippet.py').read() if False else "\n".join(
    f"def f{i}(x):\n    y = x * {i} + {i % 7}\n    if y > {i * 3}:\n        return y - {i}\n    return y + {i}\n" for i in range(40))
prompts = {"fresh": "Explain in about 300 words how a camera can track a basketball in a driveway video, step by step.",
           "copy": "Here is a Python file:\n\n" + code + "\n\nRewrite the whole file unchanged except: rename f7 to seven. Output only the code."}
out = {}
for k, p in prompts.items():
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 700, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False}, "seed": 0}
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())
    tm = r.get("timings", {})
    out[k] = {"tok_s": round(tm.get("predicted_per_second", 0), 1), "n": tm.get("predicted_n"),
              "draft_accepted": tm.get("draft_n_accepted"), "draft_n": tm.get("draft_n"), "wall": round(time.time() - t, 1)}
print(json.dumps(out))
