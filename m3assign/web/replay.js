/* 回放台：把 /api/episode 的真实逐帧数据做成连续插值动画。
   S.cur 是浮点帧号，帧与帧之间对 pose/peak 线性插值、对底图做交叉淡入，
   所以看起来是连续运动而不是幻灯片。 */
const $ = (id) => document.getElementById(id);
const AG = ['#22d3ee', '#f59e0b', '#a78bfa', '#34d399'];
const GO = ['#ef4444', '#f97316', '#eab308', '#06b6d4', '#8b5cf6'];
const f2 = (v) => (v === null || v === undefined || Number.isNaN(v) ? '-' : (+v).toFixed(2));

const S = {
  ep: null, cur: 0, raf: 0, last: 0, fps: 2.2, playing: false,
  goalSel: 0, srcRow: -1, geoCache: new Map(), heat: new Map(),
  show: { heat: true, pred: true, teach: true, trail: true, label: true, smooth: true },
};

function toast(msg) {
  const t = $('toast');
  t.textContent = msg; t.style.display = 'block';
  clearTimeout(t._h); t._h = setTimeout(() => { t.style.display = 'none'; }, 2600);
}

/* ---------- 顶部状态 ---------- */
async function pollState() {
  try {
    const st = await (await fetch('/api/state')).json();
    const b = $('trainBadge');
    b.textContent = st.status === 'no-run' ? '训练未启动' : st.status === 'running' ? '训练进行中' : st.status;
    b.className = 'badge' + (st.status === 'running' ? ' live' : '');
    $('ckBadge').textContent = `ckpt step ${st.best ? st.best.step : '-'} · 教师复现 ${((st.best?.teacher_acc || 0) * 100).toFixed(1)}%`;
  } catch (e) { /* 服务重启瞬间忽略 */ }
}

/* ---------- episode 列表 ---------- */
async function loadEpisodes() {
  $('epList').innerHTML = '<div class="note">载入中…</div>';
  const d = await (await fetch(`/api/episodes?q=${encodeURIComponent($('q').value)}&limit=120`)).json();
  const list = d.episodes || [];
  $('epList').innerHTML = list.length ? '' : '<div class="note">无匹配 episode</div>';
  list.forEach(e => {
    const div = document.createElement('div');
    div.className = 'ep' + (S.ep && S.ep.episode_id === e.episode_id ? ' sel' : '');
    div.innerHTML = `<b>${e.episode_id}</b><small>${e.scene} · K=${e.k} M=${e.m} · ${e.frames}帧 · ${e.categories.slice(0, 3).join(', ')}</small>`;
    div.onclick = () => selectEpisode(e.episode_id);
    $('epList').appendChild(div);
  });
}

/* ---------- 帧数据解码 ---------- */
function decodeGeo(frame) {
  if (S.geoCache.has(frame.row)) return S.geoCache.get(frame.row);
  const bin = Uint8Array.from(atob(frame.geo), c => c.charCodeAt(0));
  const img = new ImageData(48, 48);
  for (let i = 0; i < 48 * 48; i++) {
    const ob = bin[i * 2] / 255, ex = bin[i * 2 + 1] / 255;
    img.data[i * 4] = 20 + ob * 150;
    img.data[i * 4 + 1] = 24 + ex * 90;
    img.data[i * 4 + 2] = 30 + ex * 130;
    img.data[i * 4 + 3] = 255;
  }
  const off = document.createElement('canvas'); off.width = off.height = 48;
  off.getContext('2d').putImageData(img, 0, 0);
  S.geoCache.set(frame.row, off);
  return off;
}

async function heatFor(frame, goal) {
  const key = `${frame.row}:${goal}`;
  if (S.heat.has(key)) return S.heat.get(key);
  const d = await (await fetch(`/api/heat?row=${frame.row}&goal=${goal}`)).json();
  if (!d.data) return null;
  const bin = Uint8Array.from(atob(d.data), c => c.charCodeAt(0));
  const img = new ImageData(96, 96);
  for (let i = 0; i < 96 * 96; i++) {
    const v = bin[i] / 255;
    img.data[i * 4] = 255 * Math.min(1, v * 1.6);
    img.data[i * 4 + 1] = 255 * v * v;
    img.data[i * 4 + 2] = 120 * (1 - v);
    img.data[i * 4 + 3] = Math.min(200, v * 255);
  }
  const off = document.createElement('canvas'); off.width = off.height = 96;
  off.getContext('2d').putImageData(img, 0, 0);
  S.heat.set(key, off);
  return off;
}

const P = (v, size) => ({ x: (v[0] + 2) / 2 * size, y: (v[1] + 2) / 2 * size });
const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);

/* ---------- 三种图源：geo 通道 / heat 目标热图 / IIN 实物图 ---------- */
function tile(canvas, label, sub) {
  const d = document.createElement('div');
  d.className = 'src';
  d.appendChild(canvas);
  d.insertAdjacentHTML('beforeend', `<b>${label}</b><em>${sub || ''}</em>`);
  return d;
}

function pxCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

function geoChannel(frame, ch) {
  const bin = Uint8Array.from(atob(frame.geo), c => c.charCodeAt(0));
  const c = pxCanvas(48, 48), img = new ImageData(48, 48);
  for (let i = 0; i < 48 * 48; i++) {
    const v = bin[i * 2 + ch] ? (ch ? 200 : 210) : 22;
    img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
  }
  c.getContext('2d').putImageData(img, 0, 0);
  return c;
}

async function heatTile(row, goal) {
  const key = `src:${row}:${goal}`;
  let src = S.heat.get(key);
  if (!src) {
    const d = await (await fetch(`/api/heat?row=${row}&goal=${goal}`)).json();
    if (!d.data) return null;
    const bin = Uint8Array.from(atob(d.data), c => c.charCodeAt(0));
    const img = new ImageData(96, 96);
    for (let i = 0; i < 96 * 96; i++) {
      const v = bin[i] / 255;
      img.data[i * 4] = 255 * Math.min(1, v * 1.6);
      img.data[i * 4 + 1] = 255 * v * v;
      img.data[i * 4 + 2] = 60 + 60 * (1 - v);
      img.data[i * 4 + 3] = 255;
    }
    const c = pxCanvas(96, 96);
    c.getContext('2d').putImageData(img, 0, 0);
    src = c;
    S.heat.set(key, src);
  }
  const c = pxCanvas(96, 96);
  c.getContext('2d').drawImage(src, 0, 0);
  return c;
}

async function renderSources(frame) {
  const ep = S.ep;
  if (!ep) return;
  const row = frame.row;
  $('srcFrame').textContent = `帧 ${Math.floor(S.cur) + 1}/${ep.frames.length} · shard ${frame.shard} row ${frame.row}`;

  const explored = frame.geo ? geoChannel(frame, 1) : null;
  $('srcsGeo').innerHTML = '';
  $('srcsGeo').append(
    tile(geoChannel(frame, 0), '障碍占用 (geo·ch0)', `${frame.reason}`),
    tile(explored, '已探索 (geo·ch1)', '48×48 降采样 · 随 agent 行进增长'));

  const gBox = $('srcsHeat');
  gBox.innerHTML = '';
  const goals = ep.goals.filter(g => g.index < frame.valid_m.length);
  for (const g of goals) {
    const c = await heatTile(row, g.index);
    if (!c || S.ep !== ep || S.srcRow !== row) continue;
    const t = tile(c, `g${g.index} ${g.category}`, `${g.modality} · 峰值置信 ${(frame.peak[g.index][2] || 0).toFixed(2)}`);
    if (g.index === S.goalSel) t.classList.add('on');
    t.classList.add('pick');
    t.onclick = () => { S.goalSel = g.index; draw(); renderSources(frame); };
    gBox.append(t);
  }

  const iBox = $('srcsIin');
  iBox.innerHTML = '';
  const withIin = goals.filter(g => g.iin_exists);
  if (!withIin.length) {
    iBox.innerHTML = '<span class="note">本 episode 的目标都是文字/类别目标（TN/ON），数据集没有实物图</span>';
  }
  withIin.forEach(g => {
    const c = pxCanvas(96, 96);
    const im = new Image();
    im.onload = () => { c.getContext('2d').drawImage(im, 0, 0, 96, 96); };
    im.src = `/api/iin?episode=${encodeURIComponent(ep.episode_id)}&goal=${g.index}`;
    const t = tile(c, `g${g.index} ${g.category}`, 'IIN · 512×512 渲染后缩');
    if (g.index === S.goalSel) t.classList.add('on');
    t.classList.add('pick');
    t.onclick = () => { S.goalSel = g.index; draw(); renderSources(frame); };
    iBox.append(t);
  });
}

/* 帧间插值：pose / peak 线性混合；底图在相邻两帧间交叉淡入 */
function mix(i, t) {
  const fr = S.ep.frames;
  const a = fr[Math.min(i, fr.length - 1)], b = fr[Math.min(i + 1, fr.length - 1)];
  const pose = a.pose.map((p, k) => (t > 0 && b.valid_k[k] && a.valid_k[k]) ? lerp(p, b.pose[k], t) : p);
  const peak = a.peak.map((p, k) => (t > 0 && b.valid_m[k] && a.valid_m[k]) ? lerp(p, b.peak[k], t) : p);
  return { base: a, pose, peak };
}

/* ---------- 主绘制 ---------- */
function draw() {
  if (!S.ep) return;
  const c = $('bev'), size = c.width, ctx = c.getContext('2d');
  const i = Math.floor(S.cur), t = S.show.smooth ? S.cur - i : 0;
  const fr = S.ep.frames[i];
  const { pose, peak } = mix(i, t);

  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(decodeGeo(fr), 0, 0, size, size);
  if (t > 0 && i + 1 < S.ep.frames.length) {
    ctx.globalAlpha = t;
    ctx.drawImage(decodeGeo(S.ep.frames[i + 1]), 0, 0, size, size);
    ctx.globalAlpha = 1;
  }

  if (S.show.heat) {
    heatFor(fr, S.goalSel).then(h => {
      if (!h || !S.ep || S.ep.frames[i] !== fr) return;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(h, 0, 0, size, size);
      ctx.imageSmoothingEnabled = false;
      vectors(ctx, fr, pose, peak, size, i);
    });
  }
  vectors(ctx, fr, pose, peak, size, i);
  drawTime();
  $('frameInfo').textContent =
    `帧 ${i + 1}/${S.ep.frames.length}${S.show.smooth && t ? ` (+${(t * 100).toFixed(0)}%)` : ''} · event ${fr.event_id} · ${fr.reason}` +
    ` · 剩余目标 ${fr.valid_m.reduce((a, b) => a + b, 0)}/${fr.valid_m.length}` +
    ` · 与 teacher ${(agree(fr).a)}/${agree(fr).n} 一致`;
  syncPanels(fr);
  if (S.srcRow !== fr.row) { S.srcRow = fr.row; renderSources(fr); }
}

function vectors(ctx, fr, pose, peak, size, i) {
  peak.forEach((p, g) => {
    if (!fr.valid_m[g]) return;
    const q = P(p, size);
    ctx.strokeStyle = ctx.fillStyle = GO[g % 5]; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(q.x, q.y - 8); ctx.lineTo(q.x + 8, q.y); ctx.lineTo(q.x, q.y + 8); ctx.lineTo(q.x - 8, q.y); ctx.closePath(); ctx.stroke();
    if (S.show.label) { ctx.font = '13px sans-serif'; ctx.fillText('g' + g, q.x + 10, q.y - 7); }
  });

  if (S.show.trail) {
    for (let k = 0; k < 4; k++) {
      if (!fr.valid_k[k]) continue;
      ctx.strokeStyle = AG[k]; ctx.lineWidth = 1.6; ctx.globalAlpha = .45; ctx.beginPath();
      let started = false;
      for (let n = 0; n <= i; n++) {
        const p = S.ep.frames[n]; if (!p.valid_k[k]) continue;
        const q = P(p.pose[k], size); started ? ctx.lineTo(q.x, q.y) : (ctx.moveTo(q.x, q.y), started = true);
      }
      if (started) { const q = P(pose[k], size); ctx.lineTo(q.x, q.y); }
      ctx.stroke(); ctx.globalAlpha = 1;
    }
  }

  const link = (a, g, color, dash) => {
    if (g == null || g < 0 || !fr.valid_m[g] || !fr.valid_k[a]) return;
    const p = P(pose[a], size), q = P(peak[g], size);
    ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = dash ? 1.4 : 2.6;
    ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke(); ctx.setLineDash([]);
  };
  if (S.show.teach) fr.teacher.forEach((g, a) => link(a, g, AG[a], []));
  if (S.show.pred) fr.pred.forEach((g, a) => link(a, g, '#e2e8f0', [6, 5]));

  pose.forEach((p, k) => {
    if (!fr.valid_k[k]) return;
    const q = P(p, size);
    ctx.fillStyle = AG[k]; ctx.beginPath(); ctx.arc(q.x, q.y, 7, 0, 7); ctx.fill();
    ctx.strokeStyle = '#0b1016'; ctx.lineWidth = 2; ctx.stroke();
    if (S.show.label) { ctx.fillStyle = '#0b1016'; ctx.font = 'bold 11px sans-serif'; ctx.fillText('A' + k, q.x - 7, q.y + 4); }
  });
}

/* ---------- 时间轴 ---------- */
function agree(fr) {
  let a = 0, n = 0;
  fr.teacher.forEach((t, k) => { if (fr.valid_k[k] && t >= 0) { n++; if (fr.pred[k] === t) a++; } });
  return { a, n };
}

function drawTime() {
  const c = $('time'), ctx = c.getContext('2d'), n = S.ep.frames.length;
  ctx.clearRect(0, 0, c.width, c.height);
  const w = c.width / n;
  S.ep.frames.forEach((fr, i) => {
    const { a, n: k } = agree(fr);
    ctx.fillStyle = k ? (a === k ? '#15803d' : a ? '#a16207' : '#7f1d1d') : '#1b2735';
    ctx.fillRect(i * w + 1, 6, w - 2, 12);
    ctx.fillStyle = '#7d8ea1'; ctx.font = '9px sans-serif';
    if (w > 22) ctx.fillText(String(i + 1), i * w + 3, 28);
  });
  ctx.fillStyle = '#22d3ee';
  ctx.fillRect(S.cur * w - 1, 0, 2, c.height);
}

/* ---------- 缩略图带 ---------- */
function buildStrip() {
  const box = $('strip'); box.innerHTML = '';
  S.ep.frames.forEach((fr, i) => {
    const cv = document.createElement('canvas'); cv.width = cv.height = 52;
    const x = cv.getContext('2d'); x.imageSmoothingEnabled = false;
    x.drawImage(decodeGeo(fr), 0, 0, 52, 52);
    fr.pose.forEach((p, k) => {
      if (!fr.valid_k[k]) return;
      const q = P(p, 52); x.fillStyle = AG[k]; x.beginPath(); x.arc(q.x, q.y, 3, 0, 7); x.fill();
    });
    cv.onclick = () => { S.cur = i; draw(); };
    cv.dataset.i = i;
    box.appendChild(cv);
  });
}

/* ---------- 右侧面板 ---------- */
function syncPanels(fr) {
  const { a, n } = agree(fr);
  $('decSum').textContent = `${a}/${n} 一致`;
  $('decTable').innerHTML = fr.teacher.map((t, k) => {
    if (!fr.valid_k[k]) return '';
    const p = fr.pred[k], ok = t < 0 ? null : p === t, cost = (fr.travel[k] || [])[p];
    const core = fr.teacher_core[k], con = fr.contract[k];
    return `<tr><td><i class="dot" style="background:${AG[k]}"></i>A${k}</td>
      <td>${t < 0 ? '空闲' : 'g' + t}</td>
      <td>g${p} <span class="${ok === null ? 'note' : ok ? 'ok' : 'bad'}">${ok === null ? '—' : ok ? '✓' : '✗'}</span></td>
      <td class="num conf">${(fr.conf[k] * 100).toFixed(0)}%</td>
      <td class="num">${cost >= 9999 ? '不可达' : f2(cost)}</td>
      <td class="note">${core >= 0 ? '锁定 g' + core : con >= 0 ? '沿用 g' + con : '未锁定'}</td></tr>`;
  }).join('');

  [...$('strip').children].forEach((el, i) => el.classList.toggle('on', i === Math.floor(S.cur)));

  const f0 = S.ep.frames[0];
  $('goals').querySelectorAll('.goal').forEach(el => {
    const g = +el.dataset.g, st = el.querySelector('.st');
    const now = fr.valid_m[g];
    st.textContent = !f0.valid_m[g] ? '起始即无效' : now ? '待完成' : '已到达';
    st.className = 'pill st' + (now || !f0.valid_m[g] ? '' : ' done');
  });
}

function renderAgreeList() {
  $('agreeList').innerHTML = S.ep.frames.map((fr, i) => {
    const { a, n } = agree(fr);
    const pct = n ? a / n : 0;
    return `<div class="row" style="align-items:center;gap:6px;cursor:pointer" data-i="${i}">
      <span class="note" style="width:26px">${i + 1}</span>
      <span class="bar" style="flex:1"><i style="width:${pct * 100}%;background:${pct === 1 ? '#22c55e' : pct ? '#f59e0b' : '#ef4444'}"></i></span>
      <span class="note" style="width:30px;text-align:right">${a}/${n}</span></div>`;
  }).join('');
  $('agreeList').querySelectorAll('[data-i]').forEach(el => {
    el.onclick = () => { S.cur = +el.dataset.i; draw(); };
  });
}

/* ---------- 载入 episode ---------- */
async function selectEpisode(id) {
  S.geoCache.clear(); S.heat.clear(); S.srcRow = -1;
  const d = await (await fetch(`/api/episode?id=${encodeURIComponent(id)}`)).json();
  if (d.error) return toast('episode 载入失败: ' + d.error);
  S.ep = d; S.cur = 0; S.goalSel = (d.goals[0] || {}).index || 0;
  $('epBadge').textContent = `${d.episode_id} · ${d.frames.length}帧 · ckpt ${d.ckpt_step}`;
  $('epList').querySelectorAll('.ep').forEach(x => x.classList.toggle('sel', x.textContent.startsWith(id)));
  $('goals').innerHTML = d.goals.map(g => `
    <div class="goal" data-g="${g.index}">
      <h3><i class="dot" style="background:${GO[g.index % 5]}"></i>g${g.index} ${g.category}
        <span class="pill">${g.modality}</span><span class="pill st">待完成</span></h3>
      ${g.object_name ? `<div class="note">${g.object_name}</div>` : ''}
      <p>${g.tn_text || (g.modality === 'IIN'
        ? '目标以实物照片给出（见下方），无文字指令。'
        : `目标以物体类别给出，本场景另有 ${g.n_distractors} 个同类干扰项。`)}</p>
      ${g.iin_exists ? `<img loading="lazy" src="/api/iin?episode=${encodeURIComponent(d.episode_id)}&goal=${g.index}">` : ''}
    </div>`).join('') || '<div class="note">无目标元数据</div>';
  $('goals').querySelectorAll('.goal').forEach(el => el.onclick = () => { S.goalSel = +el.dataset.g; draw(); });
  buildStrip(); renderAgreeList(); draw();
}

/* ---------- 动画循环 ---------- */
function loop(ts) {
  S.raf = requestAnimationFrame(loop);
  if (!S.playing || !S.ep) { S.last = ts; return; }
  const dt = Math.min(0.25, (ts - S.last) / 1000); S.last = ts;
  S.cur += dt * S.fps;
  if (S.cur >= S.ep.frames.length - 1) S.cur = 0;
  draw();
}

/* ---------- 控制 ---------- */
function setPlay(on) {
  S.playing = on;
  $('btnPlay').textContent = on ? '⏸ 暂停' : '▶ 播放';
}
$('btnPlay').onclick = () => setPlay(!S.playing);
$('btnBack').onclick = () => { S.cur = 0; draw(); };
$('btnPrev').onclick = () => { S.cur = Math.max(0, Math.floor(S.cur) - 1); draw(); };
$('btnNext').onclick = () => { S.cur = Math.min(S.ep ? S.ep.frames.length - 1 : 0, Math.floor(S.cur) + 1); draw(); };
$('speed').onchange = (e) => { S.fps = +e.target.value; };
$('bev').onclick = () => setPlay(!S.playing);
$('time').onclick = (e) => {
  if (!S.ep) return;
  const r = e.target.getBoundingClientRect();
  S.cur = Math.max(0, Math.min(S.ep.frames.length - 1, (e.clientX - r.left) / r.width * S.ep.frames.length));
  draw();
};
const toggle = (id, key) => { $(id).onclick = () => { S.show[key] = !S.show[key]; $(id).classList.toggle('on', S.show[key]); draw(); }; };
toggle('btnHeat', 'heat'); toggle('btnPred', 'pred'); toggle('btnTeach', 'teach');
toggle('btnTrail', 'trail'); toggle('btnLabel', 'label'); toggle('btnSmooth', 'smooth');
$('reload').onclick = loadEpisodes;
$('q').oninput = () => { clearTimeout(window._qt); window._qt = setTimeout(loadEpisodes, 350); };
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { e.preventDefault(); setPlay(!S.playing); }
  else if (e.key === 'ArrowLeft') { S.cur = Math.max(0, Math.floor(S.cur) - 1); draw(); }
  else if (e.key === 'ArrowRight') { S.cur = Math.min(S.ep.frames.length - 1, Math.floor(S.cur) + 1); draw(); }
  else if (e.key === '[') { $('speed').selectedIndex = Math.max(0, $('speed').selectedIndex - 1); S.fps = +$('speed').value; }
  else if (e.key === ']') { $('speed').selectedIndex = Math.min(4, $('speed').selectedIndex + 1); S.fps = +$('speed').value; }
});
setInterval(() => { $('clock').textContent = new Date().toLocaleTimeString('zh-CN'); }, 1000);

pollState(); setInterval(pollState, 2000); loadEpisodes();
S.raf = requestAnimationFrame(loop);
(async () => {
  /* 默认选一个"真的有运动"的 episode：帧数多的才有 pose/geo 逐帧变化，
     否则打开看到的是同一个决策重复确认，像卡住一样。 */
  const d = await (await fetch('/api/episodes?limit=200')).json();
  const list = d.episodes || [];
  if (!list.length) return;
  const best = list.reduce((a, b) => (b.frames > a.frames ? b : a), list[0]);
  selectEpisode(best.episode_id);
})();
