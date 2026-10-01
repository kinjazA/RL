import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "answers.json")
OUT = os.path.join(HERE, "index.html")

with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

questions = list(data.keys())
stages = ["Base", "SFT", "SFT + DPO"]

# Embed answers as JSON (escape </script> just in case)
payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

METRICS = [
    ("回答中位字数", "883", "345", "443", "越接近题目所需越好"),
    ("自然收尾率", "100%", "100%", "100%", "越高越好"),
    ("强 5-gram 重复率", "90.6%", "9.4%", "10.9%", "越低越好"),
]

html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Qwen2.5-3B 面试助手 · SFT + DPO 微调对比</title>
<style>
  :root {{
    --bg: #f7f8fa; --card: #ffffff; --ink: #1a1d23; --muted: #6b7280;
    --line: #e5e7eb; --base: #6b7280; --sft: #2563eb; --dpo: #059669;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.7 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    -webkit-font-smoothing: antialiased; }}
  header {{ background: linear-gradient(135deg,#0f172a,#1e3a8a); color:#fff; padding: 40px 20px 36px; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 0 20px; }}
  header h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: .3px; }}
  header p {{ margin: 0; color: #cbd5e1; font-size: 14px; }}
  .badges {{ margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }}
  .badge {{ font-size: 12px; padding: 4px 10px; border-radius: 999px; border: 1px solid #334155; color:#dbeafe; }}
  main {{ padding: 28px 0 48px; }}
  section {{ margin-bottom: 28px; }}
  h2 {{ font-size: 17px; margin: 0 0 14px; }}
  .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
  .metric {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }}
  .metric .name {{ font-size: 13px; color: var(--muted); margin-bottom: 10px; }}
  .metric .row {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; }}
  .metric .row:last-child {{ margin-bottom: 0; }}
  .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
  .dot.base {{ background: var(--base); }} .dot.sft {{ background: var(--sft); }} .dot.dpo {{ background: var(--dpo); }}
  .metric .hint {{ font-size: 11px; color: #9ca3af; margin-top: 10px; }}
  .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
  .tab {{ border: 1px solid var(--line); background: var(--card); color: var(--muted);
    padding: 8px 14px; border-radius: 999px; cursor: pointer; font-size: 13px; }}
  .tab.active {{ background: #1e3a8a; color: #fff; border-color: #1e3a8a; }}
  .qtitle {{ font-size: 16px; font-weight: 600; margin: 0 0 14px; }}
  .cols {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
  .col {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px;
    display: flex; flex-direction: column; }}
  .col .stage {{ font-size: 13px; font-weight: 700; padding-bottom: 10px; margin-bottom: 10px;
    border-bottom: 1px solid var(--line); display: flex; align-items: center; }}
  .col.base .stage {{ color: var(--base); }} .col.sft .stage {{ color: var(--sft); }} .col.dpo .stage {{ color: var(--dpo); }}
  .col .answer {{ font-size: 14px; white-space: pre-wrap; word-break: break-word; }}
  .col .time {{ margin-top: auto; padding-top: 12px; font-size: 11px; color: #9ca3af; }}
  pre {{ background: #f3f4f6; border: 1px solid var(--line); border-radius: 8px; padding: 12px;
    overflow-x: auto; font: 12.5px/1.6 ui-monospace, "SF Mono", Consolas, monospace; white-space: pre-wrap; }}
  footer {{ border-top: 1px solid var(--line); padding: 24px 0; color: var(--muted); font-size: 13px; }}
  footer a {{ color: #2563eb; text-decoration: none; }}
  @media (max-width: 820px) {{
    .cols, .metrics {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>Qwen2.5-3B 面试助手 · SFT + DPO 微调对比</h1>
    <p>同一道面试题，在 Qwen2.5-3B-Instruct 基座上依次经过 QLoRA SFT 与 DPO 对齐后的回答对比（4-bit 加载，贪心解码）。</p>
    <div class="badges">
      <span class="badge">基座 Qwen2.5-3B-Instruct</span>
      <span class="badge">QLoRA SFT</span>
      <span class="badge">DPO 对齐</span>
      <span class="badge">reward-model 偏好对</span>
    </div>
  </div>
</header>
<main class="wrap">
  <section>
    <h2>独立测试集结果（64 题）</h2>
    <div class="metrics">
      {''.join(
        f'<div class="metric"><div class="name">{name}</div>'
        f'<div class="row"><span><span class="dot base"></span>Base</span><span>{base}</span></div>'
        f'<div class="row"><span><span class="dot sft"></span>SFT</span><span>{sft}</span></div>'
        f'<div class="row"><span><span class="dot dpo"></span>SFT+DPO</span><span>{dpo}</span></div>'
        f'<div class="hint">{hint}</div></div>'
        for name, base, sft, dpo, hint in METRICS
      )}
    </div>
  </section>
  <section>
    <h2>逐题三阶段对比</h2>
    <div class="tabs" id="tabs"></div>
    <h3 class="qtitle" id="qtitle"></h3>
    <div class="cols" id="cols"></div>
  </section>
</main>
<footer class="wrap">
  <p>模型：<a href="https://huggingface.co/Shawnno/qwen2.5-3b-interview-sft-lora">SFT LoRA</a> ·
  <a href="https://huggingface.co/Shawnno/qwen2.5-3b-interview-dpo-lora">SFT+DPO LoRA</a> ·
  代码：<a href="https://github.com/kinjazA/RL">github.com/kinjazA/RL</a></p>
</footer>
<script>
const DATA = {payload};
const QUESTIONS = Object.keys(DATA);
const STAGES = {stages};
let current = 0;
function renderTabs() {{
  const el = document.getElementById('tabs');
  el.innerHTML = QUESTIONS.map((q, i) =>
    `<button class="tab ${{i===current?'active':''}}" data-i="${{i}}">${{i+1}}</button>`).join('');
  el.querySelectorAll('.tab').forEach(b => b.onclick = () => {{ current = +b.dataset.i; render(); }});
}}
function render() {{
  const q = QUESTIONS[current];
  document.getElementById('qtitle').textContent = q;
  document.getElementById('cols').innerHTML = STAGES.map(st => {{
    const d = DATA[q][st];
    const cls = st === 'Base' ? 'base' : st === 'SFT' ? 'sft' : 'dpo';
    return `<div class="col ${{cls}}">
      <div class="stage"><span class="dot ${{cls}}"></span>${{st}}</div>
      <div class="answer">${{escapeHtml(d.answer)}}</div>
      <div class="time">耗时 ${{d.elapsed}}s</div>
    </div>`;
  }}).join('');
  renderTabs();
}}
function escapeHtml(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}
render();
</script>
</body>
</html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print("WROTE", OUT, len(html), "bytes")
