"""把看板用的真实 episode 数据渲染成对话内的连续播放器（逐帧 BEV + 分配线 + 指令 + IIN 实物图）。

数据全部来自 127.0.0.1:8765 的 /api/episode、/api/iin、/api/sim，不做任何模拟。
用法：python setup/render_replay.py [episode_id]
"""
import base64
import io
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = os.environ.get("BOARD_URL", "http://127.0.0.1:8765")
EP = sys.argv[1] if len(sys.argv) > 1 else "00006-HkseAnWCgqk_00806"
OUT_DIR = Path(os.environ.get("REPLAY_OUT")
               or Path(__file__).resolve().parent.parent / "out")
SIZE = 288

AG = [(34, 211, 238), (245, 158, 11), (167, 139, 250), (52, 211, 153)]
GO = [(239, 68, 68), (249, 115, 22), (234, 179, 8), (6, 182, 212), (139, 92, 246)]
try:
    F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    FS = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
except OSError:
    F = FS = ImageFont.load_default()


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return r.read()


def P(v):
    return ((v[0] + 2) / 2 * SIZE, (v[1] + 2) / 2 * SIZE)


def bev(buf):
    """前端下发的 geo 是 (48,48,2) uint8（通道 0=障碍，1=已探索），配色跟 decodeGeo 一致。"""
    a = np.frombuffer(buf, np.uint8).reshape(48, 48, 2).astype(np.float32) / 255.0
    rgb = np.empty((48, 48, 3), np.uint8)
    rgb[..., 0] = (20 + a[..., 0] * 150).astype(np.uint8)
    rgb[..., 1] = (24 + a[..., 1] * 90).astype(np.uint8)
    rgb[..., 2] = (30 + a[..., 1] * 130).astype(np.uint8)
    return Image.fromarray(rgb).resize((SIZE, SIZE), Image.NEAREST)


def frame_img(fr, frames, cur):
    ch = bev(base64.b64decode(fr["geo"]))
    d = ImageDraw.Draw(ch, "RGBA")

    for k in range(4):
        pts = [P(p["pose"][k]) for p in frames[: cur + 1] if p["valid_k"][k]]
        if len(pts) > 1:
            d.line(pts, fill=AG[k] + (150,), width=2)

    for i, p in enumerate(fr["peak"]):
        if not fr["valid_m"][i]:
            continue
        x, y = P(p)
        d.polygon([(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)], outline=GO[i], width=2)
        d.text((x + 9, y - 8), f"g{i}", fill=GO[i], font=FS)

    def link(a, g, color, dash):
        if g < 0 or not fr["valid_m"][g] or not fr["valid_k"][a]:
            return
        p, q = P(fr["pose"][a]), P(fr["peak"][g])
        if dash:
            n = max(2, int((abs(q[0] - p[0]) + abs(q[1] - p[1])) / 12))
            for t in range(n):
                f0, f1 = t / n, (t + 0.55) / n
                d.line(
                    [(p[0] + (q[0] - p[0]) * f0, p[1] + (q[1] - p[1]) * f0),
                     (p[0] + (q[0] - p[0]) * f1, p[1] + (q[1] - p[1]) * f1)],
                    fill=color, width=1)
        else:
            d.line([p, q], fill=color, width=3)

    for a, g in enumerate(fr["teacher"]):
        link(a, g, AG[a] + (235,), False)
    for a, g in enumerate(fr["pred"]):
        link(a, g, (255, 255, 255, 215), True)

    for k, p in enumerate(fr["pose"]):
        if not fr["valid_k"][k]:
            continue
        x, y = P(p)
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=AG[k], outline=(11, 16, 22), width=2)
        d.text((x - 6, y - 5), f"A{k}", fill=(11, 16, 22), font=F)

    locked = sum(1 for c in fr["contract"] if c >= 0)
    d.rectangle([0, 0, 172, 18], fill=(6, 10, 14, 205))
    d.text((4, 4), f"#{cur + 1} {fr['reason']} · locked {locked}/{sum(fr['valid_k'])}",
           fill=(226, 232, 240), font=FS)
    return ch, locked


def b64(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ep = json.loads(get("/api/episode?id=" + urllib.parse.quote(EP)).decode())
    sim = json.loads(get("/api/sim").decode())
    frames = ep["frames"]

    pngs, caps, rows = [], [], []
    for cur, fr in enumerate(frames):
        im, locked = frame_img(fr, frames, cur)
        pngs.append(b64(im))
        dec = "  ".join(
            f"A{a}: teacher g{fr['teacher'][a]} → 模型 g{fr['pred'][a]}"
            f"{'✓' if fr['teacher'][a] == fr['pred'][a] else '✗'} {fr['conf'][a] * 100:.0f}%"
            for a in range(4) if fr["valid_k"][a])
        caps.append({
            "t": (f"帧 {cur + 1}/{len(frames)} · event {fr['event_id']} · {fr['reason']}"
                  f" · 已锁定 {locked}/{sum(fr['valid_k'])} · 剩余目标 {sum(fr['valid_m'])}/{len(fr['valid_m'])}"),
            "d": dec,
        })
        agree = sum(1 for a in range(4)
                    if fr["valid_k"][a] and fr["teacher"][a] == fr["pred"][a])
        n_k = sum(fr["valid_k"])
        rows.append(
            f"<tr><td class='f'>{cur + 1}</td><td>{fr['reason']}</td>"
            f"<td class='num'>{agree}/{n_k}</td><td>{dec}</td></tr>")

    goals = []
    for g in ep["goals"]:
        txt = (g.get("tn_text") or "").strip()
        img = ""
        if g.get("iin_exists"):
            try:
                img = ('<img src="data:image/png;base64,'
                       + base64.b64encode(get(f"/api/iin?episode={EP}&goal={g['index']}")).decode()
                       + '" alt="">')
            except Exception:
                img = ""
        note = txt[:230] + ("…" if len(txt) > 230 else "") if txt else "（无文字描述，目标由图像给出）"
        goals.append(
            f"<div class='goal'><div class='gh'><span class='chip' style='background:rgb{GO[g['index'] % 5]}'>g{g['index']}</span>"
            f"<b>{g['category']}</b><em>{g['modality']}</em></div>"
            f"<div class='gb'>{img}<p>{note}</p></div></div>")

    an, th, orc, rnd = (sim["metrics"][k] for k in ("assignnet", "teacher", "oracle", "random"))

    html = f"""<style>
.vz{{font:14px/1.6 -apple-system,'Noto Sans CJK SC',sans-serif;color:#e2e8f0;background:#0b1016;padding:16px;border-radius:12px;max-width:960px}}
.vz h3{{margin:0 0 4px;font-size:16px}} .vz .sub{{color:#7d8ea1;font-size:12px;margin-bottom:12px}}
.vz .stage{{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}}
.vz img.frame{{width:430px;max-width:100%;border-radius:8px;border:1px solid #1e2937;image-rendering:pixelated}}
.vz .ctl{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}}
.vz button{{background:#1b2531;color:#e2e8f0;border:1px solid #2b3a4c;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:13px}}
.vz button:hover{{background:#243141}}
.vz input[type=range]{{flex:1;min-width:170px;accent-color:#22d3ee}}
.vz select{{background:#1b2531;color:#e2e8f0;border:1px solid #2b3a4c;border-radius:6px;padding:4px}}
.vz .cap{{color:#9fb0c3;font-size:12px;margin-top:8px}}
.vz .dec{{color:#e2e8f0;font-size:12px;font-family:ui-monospace,monospace;min-height:34px}}
.vz .kpis{{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}}
.vz .kpi{{background:#121a24;border:1px solid #1e2937;border-radius:8px;padding:8px 12px;min-width:100px}}
.vz .kpi span{{display:block;color:#7d8ea1;font-size:11px}} .vz .kpi b{{font-size:17px}}
.vz .lg{{color:#7d8ea1;font-size:12px;margin:6px 0}}
.vz .strip{{display:flex;gap:3px;overflow-x:auto;padding-bottom:6px;margin:12px 0 16px}}
.vz .strip img{{height:58px;border-radius:4px;border:1px solid #1e2937;cursor:pointer;opacity:.5;image-rendering:pixelated}}
.vz .strip img.on{{opacity:1;border-color:#22d3ee}}
.vz table{{border-collapse:collapse;width:100%;font-size:12px;margin-bottom:16px}}
.vz td{{border-bottom:1px solid #17212d;padding:3px 6px}}
.vz td.f,.vz td.num{{color:#7d8ea1;white-space:nowrap}}
.vz .goals{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}}
.vz .goal{{background:#121a24;border:1px solid #1e2937;border-radius:8px;padding:10px}}
.vz .gh{{display:flex;align-items:center;gap:8px;margin-bottom:6px}} .vz .gh em{{color:#7d8ea1;font-style:normal;font-size:11px;margin-left:auto}}
.vz .chip{{width:20px;height:20px;border-radius:5px;color:#0b1016;font-weight:700;font-size:11px;display:grid;place-items:center}}
.vz .gb{{display:flex;gap:10px}} .vz .gb img{{width:72px;height:72px;object-fit:cover;border-radius:6px;border:1px solid #1e2937}}
.vz .gb p{{margin:0;color:#b6c4d4;font-size:12px}}
.vz .note{{color:#7d8ea1;font-size:11px;margin-top:14px;border-top:1px solid #1e2937;padding-top:10px}}
</style>
<div class="vz">
<h3>{EP} · 连续回放</h3>
<div class="sub">场景 {ep['meta']['scene_key']} · {ep['meta']['num_agents']} 个 agent · {len(ep['goals'])} 个目标 · {len(frames)} 帧 · 权重 ckpt step {ep['ckpt_step']}</div>
<div class="kpis">
<div class="kpi"><span>SR（navmesh）</span><b>{an['SR']*100:.0f}%</b></div>
<div class="kpi"><span>MSPL</span><b>{an['MSPL']:.3f}</b></div>
<div class="kpi"><span>预算内完成</span><b>{an['SR_within_budget']*100:.1f}%</b></div>
<div class="kpi"><span>regret</span><b>{an['regret']:.2f}</b></div>
<div class="kpi"><span>本集 oracle makespan</span><b>{ep['meta']['oracle_makespan']:.1f}m</b></div>
</div>
<div class="stage">
<div>
<img class="frame" id="st" alt="BEV 帧">
<div class="cap" id="cap"></div>
<div class="dec" id="dec"></div>
<div class="ctl">
<button id="pp">⏸ 暂停</button>
<button id="rst">⟲ 从头</button>
<label>速度 <select id="sp"><option value="900">慢</option><option value="450" selected>中</option><option value="180">快</option></select></label>
<input type="range" id="rg" min="1" max="{len(frames)}" value="1">
</div>
</div>
<div style="flex:1;min-width:250px">
<div class="lg">红=障碍，蓝=已探索区域；细彩线=各 agent 已走轨迹。<br>
<b style="color:#e2e8f0">实线</b> = teacher 分配（按 agent 着色）<br>
<b style="color:#e2e8f0">白色虚线</b> = AssignNet 分配<br>
菱形 g0–g4 = 目标峰值，圆点 A0–A3 = agent 当前位置<br>
画面左上 locked n/k = 已锁定承诺（contract）的 agent 数</div>
<div class="lg">400 episode 平均 MSPL（越大越好）：<br>oracle {orc['MSPL']:.2f} · teacher {th['MSPL']:.2f} · <b style="color:#e2e8f0">AssignNet {an['MSPL']:.2f}</b> · random {rnd['MSPL']:.2f}</div>
</div>
</div>
<div class="strip" id="strip"></div>
<table><tr><td>帧</td><td>事件</td><td>与 teacher 一致</td><td>agent：teacher → 模型（置信度）</td></tr>{''.join(rows)}</table>
<h3 style="font-size:14px;margin:6px 0 8px">指令与目标状态</h3>
<div class="goals">{''.join(goals)}</div>
<div class="note">数据来自实时看板 127.0.0.1:8765 的 /api/episode（shard 真实张量 + 模型在线逐帧推理）、/api/iin（目标实物照片）、/api/sim（habitat navmesh 测地几何评测，已用 verify_navmesh 核对：36 场景平均误差 5 mm、相关系数 1.0）。</div>
<script>
const FR={json.dumps(pngs)}, CP={json.dumps(caps)};
let i=0, tm=null;
const st=document.getElementById('st'), cap=document.getElementById('cap'), dec=document.getElementById('dec'),
      rg=document.getElementById('rg'), pp=document.getElementById('pp'), strip=document.getElementById('strip'),
      sp=document.getElementById('sp');
FR.forEach((src,n)=>{{const im=document.createElement('img');im.src='data:image/png;base64,'+src;im.onclick=()=>{{stop();go(n);}};strip.appendChild(im);}});
function go(n){{i=n;st.src='data:image/png;base64,'+FR[i];cap.textContent=CP[i].t;dec.textContent=CP[i].d;rg.value=i+1;
  [...strip.children].forEach((e,k)=>e.classList.toggle('on',k===i));}}
function tick(){{go((i+1)%FR.length);}}
function start(){{if(!tm)tm=setInterval(tick,+sp.value);pp.textContent='⏸ 暂停';}}
function stop(){{if(tm){{clearInterval(tm);tm=null;}}pp.textContent='▶ 播放';}}
pp.onclick=()=>tm?stop():start();
document.getElementById('rst').onclick=()=>{{stop();go(0);}};
rg.oninput=e=>{{stop();go(+e.target.value-1);}};
sp.onchange=()=>{{if(tm){{clearInterval(tm);tm=setInterval(tick,+sp.value);}}}};
go(0);start();
</script>
</div>"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"m3-player-{EP}.html"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote", out, f"{len(html) / 1024:.0f} KB · {len(frames)} 帧")


if __name__ == "__main__":
    main()
