const S = { ep: null, cur: 0, timer: null, heat: new Map(), geoCache: new Map(),
            show: { heat: true, pred: true, trail: true }, goalSel: 0, hist: [] };
const $ = (id) => document.getElementById(id);
const AG = ['#22d3ee', '#f59e0b', '#a78bfa', '#34d399'];
const GO = ['#ef4444', '#f97316', '#eab308', '#06b6d4', '#8b5cf6'];
const f2 = (v) => (v === null || v === undefined || Number.isNaN(v) ? '-' : Number(v).toFixed(2));
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const iinURL = (ep, g) => `/api/iin?episode=${encodeURIComponent(ep)}&goal=${g}`;

function toast(m) { const t = $('toast'); t.textContent = m; t.style.display = 'block'; setTimeout(() => t.style.display = 'none', 4000); }

/* ---------- 训练指标 ---------- */
async function pollState() {
  try {
    const st = await (await fetch('/api/state')).json();
    const badge = $('trainBadge');
    if (st.status === 'no-run') { badge.textContent = '训练未启动'; badge.className = 'badge'; }
    else {
      badge.textContent = st.status === 'running' ? '训练进行中' : st.status;
      badge.className = 'badge' + (st.status === 'running' ? ' live' : '');
      $('stepBadge').textContent = `step ${st.step}/${st.total_steps} · epoch ${st.epoch}/${st.epochs_total}`;
      $('gpuBadge').textContent = st.gpu_used_mb ? `GPU ${st.gpu_used_mb}/${st.gpu_total_mb} MB` : 'CPU';
      const tr = st.last_train || {}, va = st.last_val || {}, b = st.best || {};
      $('mStep').textContent = `${st.epoch || 0}/${st.epochs_total || 0}`;
      $('mTrainLoss').textContent = f2(tr.loss);
      $('mValLoss').textContent = f2(va.loss);
      $('mAcc').textContent = va.acc !== undefined ? (va.acc * 100).toFixed(1) + '%' : '-';
      $('mTeacher').textContent = va.teacher_acc !== undefined ? (va.teacher_acc * 100).toFixed(1) + '%' : '-';
      $('mCf').textContent = va.cf_acc !== undefined ? (va.cf_acc * 100).toFixed(1) + '%' : '-';
      $('mRatio').textContent = f2(va.makespan_ratio);
      $('mSps').textContent = tr.samples_per_sec ? Math.round(tr.samples_per_sec) : '-';
      $('ckBadge').textContent = `best 教师复现 ${(b.teacher_acc * 100 || 0).toFixed(1)}% · val loss ${f2(b.loss)}`;
    }
    syncTrain(st);
    const h = await (await fetch('/api/history?limit=800')).json();
    S.hist = h.rows || [];
    drawSpark();
  } catch (e) { /* 服务重启瞬间忽略 */ }
}

function drawSpark() {
  const c = $('spark'), ctx = c.getContext('2d');
  const w = c.clientWidth, dpr = devicePixelRatio || 1;
  c.width = w * dpr; c.height = 150 * dpr; ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, 150);
  const tr = S.hist.filter(r => r.split === 'train'), va = S.hist.filter(r => r.split === 'val');
  const n = Math.max(tr.length, va.length, 2);
  const losses = [...tr.map(r => r.loss), ...va.map(r => r.loss)].filter(Number.isFinite);
  const lo = Math.min(...losses, 0.1), hi = Math.max(...losses, 0.5);
  const X = (i) => 8 + (w - 16) * i / (n - 1);
  const Y = (v) => 12 + 100 * (1 - (v - lo) / Math.max(1e-6, hi - lo));
  ctx.strokeStyle = '#1f2a37'; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) { const y = 12 + g * 25; ctx.beginPath(); ctx.moveTo(8, y); ctx.lineTo(w - 8, y); ctx.stroke(); }
  const line = (rows, key, color, dashed) => {
    ctx.beginPath(); ctx.strokeStyle = color; ctx.setLineDash(dashed ? [4, 3] : []); ctx.lineWidth = 1.6;
    rows.forEach((r, i) => { const v = r[key]; if (!Number.isFinite(v)) return; const x = X(i * (n / rows.length)), y = Y(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke(); ctx.setLineDash([]);
  };
  line(tr, 'loss', '#64748b', true);
  line(va, 'loss', '#f59e0b', false);
  ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 1.6; ctx.beginPath();
  va.forEach((r, i) => { const x = X(i * (n / Math.max(1, va.length))), y = 136 - 24 * (r.teacher_acc || 0); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.fillStyle = '#7d8ea1'; ctx.font = '10px sans-serif';
  ctx.fillText(`loss ${lo.toFixed(2)}–${hi.toFixed(2)}`, 8, 148);
  ctx.fillText('教师复现率 0–100%', w - 104, 148);
}

/* ---------- navmesh 几何评测 ---------- */
const POLICY_CN = {
  oracle: 'oracle（min-max 最优）',
  balanced: 'balanced（负载均衡贪心）',
  geometric: 'geometric（就近派单）',
  hungarian: 'hungarian（最小代价匹配）',
  nearest: 'nearest（最近优先）',
  teacher: 'teacher（语义教师）',
  assignnet: 'AssignNet（我们的策略）',
  random: 'random（随机基线）',
};

async function pollSim() {
  try {
    const d = await (await fetch('/api/sim')).json();
    if (!d.metrics) return;
    const m = d.metrics, us = m.assignnet || {};
    $('mSR').textContent = us.SR !== undefined ? (us.SR * 100).toFixed(1) + '%' : '-';
    $('mMspl').textContent = f2(us.MSPL);
    $('mBudget').textContent = us.SR_within_budget !== undefined ? (us.SR_within_budget * 100).toFixed(1) + '%' : '-';
    $('simNote').textContent = `${d.episodes} 个 episode · ckpt step ${d.ckpt_step} · 评测分片 ${d.val_shards}`;
    const gc = d.geometry_check;
    $('simGeom').textContent = gc
      ? `几何核对：habitat 复算 ${gc.scenes} 个 val 场景 / ${gc.samples} 组矩阵（agent ${gc.agent}），` +
        `与 episode 内置测地矩阵平均误差 ${gc.mean_abs_err_m} m、场景最差 p95 ${gc.worst_p95_m} m、相关系数 ${gc.corr}` +
        `（${gc.clean_scenes}/${gc.scenes} 个场景误差 < 5 cm）`
      : '几何核对：等待 setup/run_verify_navmesh.sh';
    const rows = Object.keys(m)
      .sort((a, b) => (m[b].MSPL || 0) - (m[a].MSPL || 0))
      .map((p) => {
        const r = m[p], hl = p === 'assignnet' ? ' class="ok"' : '';
        return `<tr><td>${POLICY_CN[p] || p}</td><td class="num"${hl}>${(r.SR * 100).toFixed(1)}%</td>
          <td class="num">${f2(r.MSPL)}</td><td class="num">${f2(r.SPL_sum)}</td>
          <td class="num">${f2(r.makespan)}</td><td class="num">${f2(r.regret)}</td>
          <td class="num">${(r.SR_within_budget * 100).toFixed(1)}%</td>
          <td class="num">${(r.agree_oracle * 100).toFixed(0)}%</td></tr>`;
      })
      .join('');
    $('simTable').innerHTML =
      `<tr><td>策略</td><td class="num">SR</td><td class="num">MSPL</td><td class="num">SPL_sum</td>
       <td class="num">makespan</td><td class="num">比 oracle</td><td class="num">预算内</td><td class="num">与 oracle 一致</td></tr>` +
      rows +
      `<tr><td class="note" colspan="8">${d.note || ''}；AssignNet 一帧 argmax 平均只覆盖 ${(us.one_shot_goal_cover * 100).toFixed(0)}% 的目标，表内用的是它的排序做顺序认领（限额 ⌈M/K⌉）。
        hungarian/balanced 在推理时直接读 episode 的完整测地矩阵，AssignNet 只有 shard 里的局部代价图，不是同等信息量的对比。
        实测上限：只用 shard 输入做欧氏一对一匹配，首帧命中官方最优 28.8%、随机 26.0%，而看过全图测地矩阵的 teacher 是 57.0%；
        训练分片从 40 扩到 64（2.6× 数据）后 MSPL/regret 配对比较（400 episode）也没有显著改善——瓶颈是输入信息量而不是数据量或标签加权。</td></tr>`;
  } catch (e) { /* 忽略 */ }
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

/* ---------- episode 帧数据 ---------- */
async function selectEpisode(id) {
  S.geoCache.clear();
  $('epList').querySelectorAll('.ep').forEach(d => d.classList.toggle('sel', d.textContent.startsWith(id)));
  const d = await (await fetch(`/api/episode?id=${encodeURIComponent(id)}`)).json();
  if (d.error) return toast('episode 载入失败: ' + d.error);
  S.ep = d; S.cur = 0; S.heat.clear();
  $('slider').max = d.frames.length - 1;
  $('slider').value = 0;
  $('goals').innerHTML = d.goals.map((g, i) => `
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
  draw();
}

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

// 数据坐标区间 [-2, 0]（实测），映射到画布 [0, size]
const P = (v, size) => ({ x: (v[0] + 2) / 2 * size, y: (v[1] + 2) / 2 * size });

function draw() {
  if (!S.ep) return;
  const c = $('bev'), size = c.width, ctx = c.getContext('2d');
  const fr = S.ep.frames[S.cur];
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(decodeGeo(fr), 0, 0, size, size);
  if (S.show.heat) heatFor(fr, S.goalSel).then(h => { if (h && S.ep && S.ep.frames[S.cur] === fr) { ctx.imageSmoothingEnabled = true; ctx.drawImage(h, 0, 0, size, size); vectors(ctx, fr, size); } });
  vectors(ctx, fr, size);
  $('frameInfo').textContent = `帧 ${S.cur + 1}/${S.ep.frames.length} · event ${fr.event_id} · ${fr.reason} · 剩余目标 ${fr.valid_m.reduce((a, b) => a + b, 0)}/${fr.valid_m.length}`;
  renderFrameShots();
  decisions(fr);
}

function vectors(ctx, fr, size) {
  const G = (i) => GO[i % 5];
  // 目标峰值
  fr.peak.forEach((p, i) => {
    if (!fr.valid_m[i]) return;
    const q = P(p, size);
    ctx.strokeStyle = G(i); ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(q.x, q.y - 7); ctx.lineTo(q.x + 7, q.y); ctx.lineTo(q.x, q.y + 7); ctx.lineTo(q.x - 7, q.y); ctx.closePath(); ctx.stroke();
    ctx.fillStyle = G(i); ctx.font = '12px sans-serif'; ctx.fillText('g' + i, q.x + 9, q.y - 6);
  });
  // 轨迹
  if (S.show.trail) {
    for (let k = 0; k < 4; k++) {
      ctx.strokeStyle = AG[k]; ctx.lineWidth = 1.5; ctx.globalAlpha = .5; ctx.beginPath();
      for (let i = 0; i <= S.cur; i++) {
        const p = S.ep.frames[i]; if (!p.valid_k[k]) continue;
        const q = P(p.pose[k], size); i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y);
      }
      ctx.stroke(); ctx.globalAlpha = 1;
    }
  }
  // 分配线
  const link = (a, g, color, dash) => {
    if (g < 0 || !fr.valid_m[g] || !fr.valid_k[a]) return;
    const p = P(fr.pose[a], size), q = P(fr.peak[g], size);
    ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = dash ? 1.2 : 2.2;
    ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke(); ctx.setLineDash([]);
  };
  fr.teacher.forEach((g, a) => link(a, g, AG[a], []));
  if (S.show.pred) fr.pred.forEach((g, a) => link(a, g, '#e2e8f0', [5, 4]));
  // agent
  fr.pose.forEach((p, k) => {
    if (!fr.valid_k[k]) return;
    const q = P(p, size);
    ctx.fillStyle = AG[k]; ctx.beginPath(); ctx.arc(q.x, q.y, 6, 0, 7); ctx.fill();
    ctx.fillStyle = '#0b1016'; ctx.font = 'bold 10px sans-serif'; ctx.fillText('A' + k, q.x - 6, q.y + 4);
  });
}

function decisions(fr) {
  let agree = 0, labeled = 0;
  const rows = fr.teacher.map((t, k) => {
    if (!fr.valid_k[k]) return '';
    const p = fr.pred[k], ok = t < 0 ? null : p === t;
    if (t >= 0) { labeled++; if (ok) agree++; }
    const cost = (fr.travel[k] || [])[p];
    const core = fr.teacher_core[k], con = fr.contract[k];
    return `<tr><td><i class="dot" style="background:${AG[k]}"></i>A${k}</td>
      <td>${t < 0 ? '空闲' : 'g' + t}</td>
      <td>g${p} <span class="${ok === null ? 'note' : ok ? 'ok' : 'bad'}">${ok === null ? '—' : ok ? '✓' : '✗'}</span></td>
      <td class="num">${(fr.conf[k] * 100).toFixed(0)}%</td>
      <td class="num">${cost >= 9999 ? '不可达' : f2(cost)}</td>
      <td class="note">${core >= 0 ? '锁定 g' + core : con >= 0 ? '沿用 g' + con : '未锁定'}</td></tr>`;
  }).join('');
  $('decTable').innerHTML =
    `<tr><td class="note">teacher vs 模型：${agree}/${labeled} 一致 · 实线=teacher · 虚线=模型</td></tr>` + rows;
  // 目标完成状态
  const f0 = S.ep.frames[0];
  $('goals').querySelectorAll('.goal').forEach(el => {
    const g = +el.dataset.g;
    const st = el.querySelector('.st');
    if (!st) return;
    const wasValid = f0.valid_m[g], nowValid = fr.valid_m[g];
    st.textContent = !wasValid ? '起始即无效' : (nowValid ? '待完成' : '已到达');
    st.className = 'pill st' + (nowValid || !wasValid ? '' : ' done');
  });
}

/* ---------- 训练轨迹 ---------- */
const T = { follow: true, step: 1, total: 1, slot: 0, goal: 0, run: 'latest', busy: false, last: 0, autoGoal: true, retried: false, autoRun: true };
S.lastState = null;

function tImg(b64, w, h, paint) {
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const img = new ImageData(w, h);
  for (let i = 0; i < w * h; i++) paint(img.data, i, bin);
  const off = document.createElement('canvas'); off.width = w; off.height = h;
  off.getContext('2d').putImageData(img, 0, 0);
  return off;
}

async function loadRuns() {
  try {
    const d = await (await fetch('/api/runs')).json();
    const sel = $('trRun');
    const list = d.runs || [];
    sel.innerHTML = '<option value="latest">latest</option>' + list
      .map(r => `<option value="${r.run}">${r.run} · ${r.status} ${r.step || 0}/${r.total_steps || 0}</option>`)
      .join('');
    // 默认看最近一次训练（runs/latest 会被还原成定版 checkpoint，不一定是刚跑完的那个）
    if (T.autoRun && list.length) { T.run = list[0].run; T.autoRun = false; }
    sel.value = T.run;
  } catch (e) { /* 忽略 */ }
}

async function loadTrainStep() {
  if (T.busy) return;
  T.busy = true;
  try {
    const u = `/api/train_step?step=${T.step}&slot=${T.slot}&goal=${T.goal}&run=${encodeURIComponent(T.run)}`;
    const d = await (await fetch(u)).json();
    if (d.error) { $('trMeta').textContent = d.error; return; }
    drawTrain(d);
  } catch (e) {
    $('trMeta').textContent = '训练帧载入失败：' + e;
  } finally { T.busy = false; }
}

async function syncTrain(st) {
  if (!st) return;
  S.lastState = st;
  let use = st;
  if (T.run !== 'latest') {
    try { use = await (await fetch(`/api/state?run=${encodeURIComponent(T.run)}`)).json(); }
    catch (e) { use = st; }
  }
  T.total = Math.max(1, use.step || 1);
  $('trSlider').max = T.total;
  if (T.follow) { T.step = T.total; $('trSlider').value = T.step; }
  if (T.busy || Date.now() - T.last < 1200) return;
  T.last = Date.now();
  loadTrainStep();
}

function drawTrain(d) {
  const c = $('trBev'), size = c.width, ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = false; ctx.clearRect(0, 0, size, size);
  const g = tImg(d.geo, 48, 48, (px, i, b) => {
    const ob = b[i * 2] / 255, ex = b[i * 2 + 1] / 255;
    px[i * 4] = 20 + ob * 150; px[i * 4 + 1] = 24 + ex * 90; px[i * 4 + 2] = 30 + ex * 130; px[i * 4 + 3] = 255;
  });
  ctx.drawImage(g, 0, 0, size, size);
  const hm = tImg(d.heat, 96, 96, (px, i, b) => {
    const v = b[i] / 255;
    px[i * 4] = 255 * Math.min(1, v * 1.6); px[i * 4 + 1] = 255 * v * v;
    px[i * 4 + 2] = 120 * (1 - v); px[i * 4 + 3] = Math.min(200, v * 255);
  });
  ctx.imageSmoothingEnabled = true; ctx.drawImage(hm, 0, 0, size, size);
  const P = (v) => ({ x: (v[0] + 2) / 2 * size, y: (v[1] + 2) / 2 * size });
  d.peak.forEach((p, i) => {
    if (!d.valid_m[i]) return;
    const q = P(p);
    ctx.strokeStyle = GO[i % 5]; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(q.x, q.y - 7); ctx.lineTo(q.x + 7, q.y); ctx.lineTo(q.x, q.y + 7); ctx.lineTo(q.x - 7, q.y);
    ctx.closePath(); ctx.stroke();
    ctx.fillStyle = GO[i % 5]; ctx.font = '12px sans-serif'; ctx.fillText('g' + i, q.x + 9, q.y - 6);
  });
  d.teacher.forEach((gidx, a) => {
    if (gidx < 0 || !d.valid_m[gidx] || !d.valid_k[a]) return;
    const p = P(d.pose[a]), q = P(d.peak[gidx]);
    ctx.strokeStyle = AG[a]; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
  });
  d.pose.forEach((p, k) => {
    if (!d.valid_k[k]) return;
    const q = P(p);
    ctx.fillStyle = AG[k]; ctx.beginPath(); ctx.arc(q.x, q.y, 6, 0, 7); ctx.fill();
    ctx.fillStyle = '#0b1016'; ctx.font = 'bold 10px sans-serif'; ctx.fillText('A' + k, q.x - 6, q.y + 4);
  });
  $('trStep').textContent = `step ${d.step}/${d.total_steps} · epoch ${d.epoch}`;
  $('trMeta').textContent = `${d.episode_id} · ${d.scene} · shard ${d.shard}#${d.offset} · event ${d.event_id} (${d.reason}) · batch 第 ${d.slot + 1}/${d.batch} 张 · 热图 g${d.goal}`;
  $('trSlot').max = Math.max(0, d.batch - 1);
  $('trGoal').value = String(d.goal);
  [...$('trGoal').options].forEach(o => { o.disabled = !d.valid_m[+o.value]; });

  // teacher_100k 的 shard 里没有逐步 RGB；训练集自带的真实像素只有 IIN 模态目标的那张
  // 512² 正视实拍（不是模型输入，只作"目标长什么样"的参照）。TN/ON 目标没有图。
  const goals = d.goals || [];
  const gm = goals.find(o => o.i === d.goal) || {};
  const shots = goals.filter(o => o.img);
  const validShot = shots.find(o => o.valid);
  const img = $('trImg'), none = $('trImgNone');

  // 自动模式：大图跟着本帧的指派走（优先 A0 的目标），这样 RGB 和热图说的都是
  // "agent 现在要去找的那个东西"，跟随训练时每一步都能看到照片变化。
  const assigned = (d.teacher || []).map((gi, k) => (d.valid_k[k] ? gi : -1)).filter(i => i >= 0);
  const assignedShot = assigned.map(i => goals.find(o => o.i === i)).find(o => o && o.img && o.valid);
  const pickShot = assignedShot || validShot;
  if (T.autoGoal && pickShot && pickShot.i !== d.goal && !T.retried) {
    T.retried = true; T.goal = pickShot.i; loadTrainStep(); return;
  }
  if (pickShot && pickShot.i === d.goal) T.retried = false;
  const shot = gm.img ? gm : (validShot || shots[0] || null);
  if (shot) {
    const u = iinURL(d.episode_id, shot.i);
    if (img.getAttribute('src') !== u) img.src = u;
    img.classList.add('on'); none.style.display = 'none';
  } else {
    img.classList.remove('on'); img.removeAttribute('src'); none.style.display = 'flex';
  }
  const onIt = (d.teacher || []).map((gi, k) => (d.valid_k[k] && gi === (shot && shot.i) ? `A${k}` : null)).filter(Boolean);
  $('trImgCap').textContent = shot
    ? `g${shot.i} · ${shot.cat || '未知类别'} · 模态 ${shot.mod} · `
      + (shot.i === d.goal ? '本帧选中目标' : shot.valid ? '本帧另一有效目标' : '本帧已失效')
      + (onIt.length ? ` · ${onIt.join('/')} 被指派于此` : ' · 本帧未指派')
      + ' · 训练集实拍 1.8 m 正视'
    : `${'g' + d.goal} · ${gm.cat || '未知类别'} · 模态 ${gm.mod || '?'} · 该 episode 无实拍图`;
  renderTrainAgents(d);
  renderTrainThumbs(d, shot);
}

// 每个在场 agent 当前被分到哪个目标，就把那个目标的训练集实拍贴在它旁边；
// 跟随训练时随 step 变化（换 episode / 目标被找到 / 分配改变都会换图）。
function renderTrainAgents(d) {
  const host = $('trAgents');
  const gmap = new Map((d.goals || []).map(o => [o.i, o]));
  const key = [d.episode_id, d.valid_k.join(''), d.valid_m.join(''), d.teacher.join(',')].join('|');
  if (host.dataset.key === key) return;
  host.dataset.key = key;
  const cards = [];
  (d.teacher || []).forEach((gi, k) => {
    if (!d.valid_k[k]) return;
    const g = gmap.get(gi);
    const alive = gi >= 0 && d.valid_m[gi];
    const pic = alive && g && g.img
      ? `<img class="ph" src="${iinURL(d.episode_id, gi)}" alt="目标 g${gi} 实拍">`
      : `<span class="none">${esc(alive ? ((g && g.cat) || '无实拍') : '已失效')}</span>`;
    cards.push(`<div class="trAgent" style="--ag:${AG[k % 4]}">${pic}
      <div><b>A${k}</b><span class="cap">→ ${esc(alive ? `g${gi} · ${(g && g.cat) || '未知类别'}` : '无有效目标')}</span>
      <span class="sub">${esc(alive && g ? `模态 ${g.mod || '?'}` : '')}</span></div></div>`);
  });
  host.innerHTML = cards.join('') || '<span class="note">本帧无在场 agent</span>';
}

function renderTrainThumbs(d, shot) {
  const th = $('trThumbs');
  const shots = (d.goals || []).filter(o => o.img);
  const key = d.episode_id + '|' + d.goal + '|' + shots.map(o => o.i + (o.valid ? 'v' : 'f')).join(',')
    + '|' + d.teacher.map((gi, k) => (d.valid_k[k] ? gi : -1)).join(',');
  if (th.dataset.key === key) return;
  th.dataset.key = key;
  th.innerHTML = shots.map(o => {
    const tags = d.teacher.map((gi, k) => (d.valid_k[k] && gi === o.i)
      ? `<i style="background:${AG[k % 4]}">A${k}</i>` : '').join('');
    return `<div class="trThumb${o.i === (shot && shot.i) ? ' on' : ''}${o.valid ? '' : ' stale'}"
      data-g="${o.i}" data-valid="${o.valid ? 1 : 0}"
      title="g${o.i} ${esc(o.cat)}${o.valid ? '' : '（本帧已失效）'}">
      <img src="${iinURL(d.episode_id, o.i)}" alt="g${o.i}">
      <span class="agTags">${tags}</span></div>`;
  }).join('');
}

// episode 回放：照片跟着当前帧的失效状态和教师分配一起变
function renderFrameShots() {
  const host = $('frameShots');
  if (!S.ep || !S.ep.frames || !S.ep.frames.length) { host.innerHTML = ''; return; }
  const fr = S.ep.frames[S.cur];
  const key = [S.ep.episode_id, S.cur, S.goalSel, fr.teacher.join(','), fr.valid_m.join('')].join('|');
  if (host.dataset.key === key) return;
  host.dataset.key = key;
  host.innerHTML = (S.ep.goals || []).map(g => {
    const alive = !!fr.valid_m[g.index];
    const tags = fr.teacher.map((gi, k) => (fr.valid_k[k] && gi === g.index)
      ? `<i style="background:${AG[k % 4]}">A${k}</i>` : '').join('');
    const pic = g.iin_exists
      ? `<img src="${iinURL(S.ep.episode_id, g.index)}" alt="g${g.index}">`
      : `<span class="ph">${esc(g.category || '无实拍')}</span>`;
    return `<div class="fs${g.index === S.goalSel ? ' sel' : ''}${alive ? '' : ' stale'}" data-g="${g.index}"
      title="g${g.index} ${esc(g.category || '')} · ${esc(g.modality || '?')}${alive ? '' : ' · 本帧已失效'}">
      ${pic}${tags ? `<span class="tags">${tags}</span>` : ''}${alive ? '' : '<span class="found">失效</span>'}
      <div class="cap">g${g.index} · ${esc(g.category || '?')}</div></div>`;
  }).join('');
  host.querySelectorAll('.fs').forEach(el => el.onclick = () => { S.goalSel = +el.dataset.g; draw(); });
}

$('trFollow').onclick = () => {
  T.follow = !T.follow;
  $('trFollow').classList.toggle('on', T.follow);
  if (T.follow && T.run === 'latest' && S.lastState) {
    T.step = S.lastState.step || T.step;
    $('trSlider').value = T.step;
    loadTrainStep();
  }
};
$('trSlider').oninput = (e) => {
  T.follow = false; $('trFollow').classList.remove('on');
  T.step = +e.target.value;
  loadTrainStep();
};
$('trSlot').onchange = (e) => { T.slot = Math.max(0, +e.target.value || 0); loadTrainStep(); };
$('trGoal').onchange = (e) => { T.goal = +e.target.value; T.autoGoal = false; loadTrainStep(); };
$('trThumbs').onclick = (e) => {
  const tile = e.target.closest('.trThumb');
  if (!tile || tile.dataset.valid === '0') return;
  T.goal = +tile.dataset.g; T.autoGoal = false; $('trGoal').value = T.goal; loadTrainStep();
};
$('trRun').onchange = async (e) => {
  T.run = e.target.value;
  T.autoRun = false;
  try {
    const st = await (await fetch(`/api/state?run=${encodeURIComponent(T.run)}`)).json();
    T.follow = st.status === 'running';
    T.total = Math.max(1, (T.follow ? st.step : st.total_steps) || 1);
    $('trSlider').max = T.total; T.step = T.total; $('trSlider').value = T.step;
  } catch (err) { /* 忽略 */ }
  $('trFollow').classList.toggle('on', T.follow);
  loadTrainStep();
};

/* ---------- 控制 ---------- */
$('slider').oninput = (e) => { S.cur = +e.target.value; draw(); };
$('btnPlay').onclick = () => {
  if (S.timer) { clearInterval(S.timer); S.timer = null; $('btnPlay').textContent = '▶ 播放'; return; }
  $('btnPlay').textContent = '⏸ 暂停';
  S.timer = setInterval(() => {
    if (!S.ep) return;
    S.cur = (S.cur + 1) % S.ep.frames.length;
    $('slider').value = S.cur; draw();
  }, +$('speed').value);
};
const toggle = (id, key) => { $(id).onclick = () => { S.show[key] = !S.show[key]; $(id).classList.toggle('on', S.show[key]); draw(); }; };
toggle('btnHeat', 'heat'); toggle('btnPred', 'pred'); toggle('btnTrail', 'trail');
$('reload').onclick = loadEpisodes;
$('q').oninput = () => { clearTimeout(window._qt); window._qt = setTimeout(loadEpisodes, 350); };
setInterval(() => { $('clock').textContent = new Date().toLocaleTimeString('zh-CN'); }, 1000);
pollState(); setInterval(pollState, 1500); pollSim(); setInterval(pollSim, 5000); loadEpisodes();
loadRuns(); loadTrainStep();
