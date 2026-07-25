'use strict';

/* ── State ──────────────────────────────────────────────────────────────── */
const G = {
  token: localStorage.getItem('dc_token') || '',
  inspector: localStorage.getItem('dc_inspector') || '',
  conveyor: '',   // entered per-inspection on the setup screen, not persisted
  jobId: null,
  putUrl: null,
  file: null,
  localVideo: null,
  videoW: 0,
  videoH: 0,
  frameOk: false,
  roi: null,      // {x,y,w,h} in canvas px
  roiMode: 'locked',  // editing | locked
  dragStart: null,
  pollTid: null,
  lightboxOpen: false,
  lightboxItems: [],
  lightboxIndex: -1,
  defectLightboxItems: [],
  normalLightboxItems: [],
  activeTab: 'defect',
  allDefects: [],
  allNormals: [],
  activeDefectSubtab: null,
  runtimeConfig: null,
  inspectionMode: 'auto_baseline',
  reviewRevision: 0,
  geometryLine: null, // two points normalized inside ROI: {top:{x,y}, bottom:{x,y}}
  geometryDrag: null,
};

/* ── Session info ───────────────────────────────────────────────────────── */
function updateSessionInfo() {
  const parts = [];
  if (G.inspector) parts.push('KTV: ' + G.inspector);
  if (G.conveyor)  parts.push('Băng: ' + G.conveyor);
  const text = parts.join('  |  ');
  ['hdr-session-setup', 'hdr-session-result'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  });
}
updateSessionInfo();

/* ── Screen navigation ──────────────────────────────────────────────────── */
function show(id) {
  document.querySelectorAll('.s').forEach(el => el.style.display = 'none');
  document.getElementById(id).style.display = 'flex';
}

/* ── API ────────────────────────────────────────────────────────────────── */
async function api(path, opts = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: {
      'Authorization': 'Bearer ' + G.token,
      'Content-Type': 'application/json',
      ...(opts.headers || {}),
    },
  });
  if (res.status === 401) {
    localStorage.removeItem('dc_token');
    G.token = '';
    show('s-login');
    setErr('err-login', 'Phiên đã hết hạn hoặc mã không đúng');
    throw Object.assign(new Error('Unauthorized'), { status: 401 });
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || 'HTTP ' + res.status);
  }
  return res.json();
}

async function loadRuntimeConfig() {
  G.runtimeConfig = await api('/api/runtime-config');
  if (G.runtimeConfig.local_mode) {
    document.getElementById('btn-pick').style.display = 'none';
    document.getElementById('btn-pick-local').textContent = 'Chọn video từ thư mục máy';
  }
}

function getTriggerBandConfig() {
  return G.runtimeConfig?.collection?.trigger_band || null;
}

/* ── Error helpers ──────────────────────────────────────────────────────── */
function setErr(id, msg) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.style.display = msg ? 'block' : 'none';
}

/* ══════════════════════════════════════════════════════════════════
   LOGIN
═══════════════════════════════════════════════════════════════════ */
document.getElementById('btn-login').addEventListener('click', async () => {
  const inspector = document.getElementById('inp-inspector').value.trim();
  const token     = document.getElementById('inp-token').value.trim();
  setErr('err-login', '');
  if (!inspector) { setErr('err-login', 'Vui lòng nhập tên người kiểm tra'); return; }
  if (!token)     { setErr('err-login', 'Vui lòng nhập mã truy cập'); return; }
  G.token = token;
  try {
    const res = await fetch('/api/health', { headers: { 'Authorization': 'Bearer ' + token } });
    if (res.status === 401) { setErr('err-login', 'Mã truy cập không đúng'); return; }
  } catch (_) { /* network error — proceed */ }
  G.inspector = inspector;
  localStorage.setItem('dc_token', token);
  localStorage.setItem('dc_inspector', inspector);
  updateSessionInfo();
  try {
    await loadRuntimeConfig();
  } catch (e) {
    if (e.status !== 401) setErr('err-login', 'Không tải được cấu hình hệ thống: ' + e.message);
    return;
  }
  show('s-setup');
});

document.getElementById('inp-token').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('btn-login').click();
});

/* ══════════════════════════════════════════════════════════════════
   LOGOUT
═══════════════════════════════════════════════════════════════════ */
document.getElementById('btn-logout').addEventListener('click', () => {
  localStorage.removeItem('dc_token');
  localStorage.removeItem('dc_inspector');
  G.token = ''; G.inspector = ''; G.conveyor = '';
  updateSessionInfo();
  show('s-login');
});

/* ══════════════════════════════════════════════════════════════════
   VIDEO PICKER + FRAME EXTRACTION
═══════════════════════════════════════════════════════════════════ */
const canvas = document.getElementById('roi-canvas');
const ctx = canvas.getContext('2d');
const videoEl = document.createElement('video');
videoEl.muted = true; videoEl.playsInline = true; videoEl.preload = 'metadata';

document.getElementById('btn-pick').addEventListener('click', () => {
  document.getElementById('inp-video').click();
});

document.getElementById('inp-video').addEventListener('change', function () {
  const f = this.files[0];
  if (!f) return;
  selectVideo(URL.createObjectURL(f), f, null);
});

function selectVideo(url, file, localVideo) {
  G.file = file; G.localVideo = localVideo; G.roi = null; G.geometryLine = null; G.frameOk = false; G.roiMode = 'locked';
  G.dragStart = null;
  canvas.classList.remove('editing');
  document.getElementById('roi-section').style.display = 'block';
  document.getElementById('video-name').style.display = 'block';
  document.getElementById('video-name').textContent =
    file.name + ' (' + (file.size / 1048576).toFixed(1) + ' MB)';
  document.getElementById('canvas-hint').textContent = 'Bấm Chọn vùng chụp để bắt đầu';
  updateRoiStatus();
  updateRoiControls();

  videoEl.addEventListener('seeked', onFrame, { once: true });
  videoEl.addEventListener('loadedmetadata', () => {
    videoEl.currentTime = Math.min(0.5, Math.max(0, (videoEl.duration || 1) - 0.05));
  }, { once: true });
  videoEl.src = url;
  videoEl.load();
}

document.getElementById('btn-pick-local').addEventListener('click', () => openLocalBrowser(''));

async function openLocalBrowser(path) {
  try {
    const data = await api('/api/local/videos?path=' + encodeURIComponent(path));
    const panel = document.getElementById('local-browser');
    panel.style.display = 'block';
    const up = data.path ? `<button class="btn btn-outline btn-sm" onclick="openLocalBrowser(decodeURIComponent('${localArg(data.parent || '')}'))">↑ Thư mục cha</button>` : '';
    const entries = data.entries.map(item => item.is_dir
      ? `<button class="btn btn-outline btn-sm" onclick="openLocalBrowser(decodeURIComponent('${localArg(item.path)}'))">📁 ${esc(item.name)}</button>`
      : `<button class="btn btn-outline btn-sm" onclick="selectLocalVideo(decodeURIComponent('${localArg(item.path)}'), decodeURIComponent('${localArg(item.name)}'), ${Number(item.size_bytes)}, decodeURIComponent('${localArg(item.content_type)}'))">🎬 ${esc(item.name)}</button>`
    ).join(' ');
    panel.innerHTML = `<div style="font-size:12px;color:var(--muted);margin-bottom:8px">${esc(data.root_label)}${data.path ? ' / ' + esc(data.path) : ''}</div>${up}<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px">${entries || '<span>Không có video</span>'}</div>`;
  } catch (e) { setErr('err-setup', 'Không mở được thư mục local: ' + e.message); }
}

function localArg(value) {
  return encodeURIComponent(String(value)).replace(/'/g, '%27');
}

async function selectLocalVideo(path, name, size, contentType) {
  try {
    const url = '/api/local/video?path=' + encodeURIComponent(path) + '&token=' + encodeURIComponent(G.token);
    selectVideo(url, { name, size, type: contentType }, path);
    document.getElementById('local-browser').style.display = 'none';
  } catch (e) { setErr('err-setup', e.message); }
}

function onFrame() {
  G.videoW = videoEl.videoWidth;
  G.videoH = videoEl.videoHeight;
  const displayW = canvas.getBoundingClientRect().width || canvas.offsetWidth || 360;
  canvas.width = Math.round(displayW);
  canvas.height = Math.round(displayW * G.videoH / G.videoW);
  G.frameOk = true;
  redraw();
  updateRoiStatus();
  updateRoiControls();
}

/* ══════════════════════════════════════════════════════════════════
   CANVAS ROI EDITING  (touch + mouse)
═══════════════════════════════════════════════════════════════════ */
const MIN_ROI_SIZE = 20;
const HANDLE_SIZE = 16;

function evXY(e) {
  const r = canvas.getBoundingClientRect();
  const src = e.touches ? e.touches[0] : e.changedTouches ? e.changedTouches[0] : e;
  return {
    x: (src.clientX - r.left) * (canvas.width / r.width),
    y: (src.clientY - r.top) * (canvas.height / r.height),
  };
}

function createDefaultRoi() {
  G.roi = clampRoi({
    x: canvas.width * 0.10,
    y: canvas.height * 0.20,
    w: canvas.width * 0.80,
    h: canvas.height * 0.55,
  });
}

function ensureGeometryLine() {
  if (!G.roi || G.geometryLine) return;
  G.geometryLine = { top: { x: 0.5, y: 0.0 }, bottom: { x: 0.5, y: 1.0 } };
}

function isGeometryMode() {
  return document.getElementById('inp-inspection-mode').value === 'geometry_v2';
}

function geometryCanvasPoints() {
  ensureGeometryLine();
  if (!G.roi || !G.geometryLine) return null;
  return {
    top: { x: G.roi.x + G.geometryLine.top.x * G.roi.w, y: G.roi.y + G.geometryLine.top.y * G.roi.h },
    bottom: { x: G.roi.x + G.geometryLine.bottom.x * G.roi.w, y: G.roi.y + G.geometryLine.bottom.y * G.roi.h },
  };
}

function isRoiValid() {
  return Boolean(G.roi && G.roi.w >= MIN_ROI_SIZE && G.roi.h >= MIN_ROI_SIZE);
}

function clampRoi(roi) {
  const w = Math.max(MIN_ROI_SIZE, Math.min(roi.w, canvas.width));
  const h = Math.max(MIN_ROI_SIZE, Math.min(roi.h, canvas.height));
  return {
    x: Math.max(0, Math.min(roi.x, canvas.width - w)),
    y: Math.max(0, Math.min(roi.y, canvas.height - h)),
    w,
    h,
  };
}

function setRoiEditing() {
  setErr('err-setup', '');
  if (!G.frameOk) return;
  if (!G.roi) createDefaultRoi();
  G.roiMode = 'editing';
  G.dragStart = null;
  canvas.classList.add('editing');
  updateRoiStatus();
  updateRoiControls();
  redraw();
}

function lockRoi() {
  setErr('err-setup', '');
  if (!isRoiValid()) { setErr('err-setup', 'Vùng kiểm tra chưa hợp lệ'); return; }
  G.roiMode = 'locked';
  G.dragStart = null;
  canvas.classList.remove('editing');
  updateRoiStatus();
  updateRoiControls();
  redraw();
}

function hitTestRoiHandle(p) {
  if (!G.roi) return null;
  const x1 = G.roi.x, y1 = G.roi.y, x2 = G.roi.x + G.roi.w, y2 = G.roi.y + G.roi.h;
  const handles = [
    ['nw', x1, y1], ['ne', x2, y1], ['sw', x1, y2], ['se', x2, y2],
    ['n', (x1 + x2) / 2, y1], ['s', (x1 + x2) / 2, y2],
    ['w', x1, (y1 + y2) / 2], ['e', x2, (y1 + y2) / 2],
  ];
  for (const [name, hx, hy] of handles) {
    if (Math.abs(p.x - hx) <= HANDLE_SIZE && Math.abs(p.y - hy) <= HANDLE_SIZE) return name;
  }
  if (p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) return 'move';
  return null;
}

function applyRoiDrag(p) {
  const start = G.dragStart;
  if (!start || !G.roi) return;
  const dx = p.x - start.x;
  const dy = p.y - start.y;
  const r = start.roi;
  let next = { ...r };

  if (start.handle === 'move') {
    next.x = r.x + dx;
    next.y = r.y + dy;
  } else {
    let left = r.x;
    let top = r.y;
    let right = r.x + r.w;
    let bottom = r.y + r.h;
    if (start.handle.includes('w')) left += dx;
    if (start.handle.includes('e')) right += dx;
    if (start.handle.includes('n')) top += dy;
    if (start.handle.includes('s')) bottom += dy;
    left = Math.max(0, Math.min(left, canvas.width - MIN_ROI_SIZE));
    top = Math.max(0, Math.min(top, canvas.height - MIN_ROI_SIZE));
    right = Math.max(MIN_ROI_SIZE, Math.min(right, canvas.width));
    bottom = Math.max(MIN_ROI_SIZE, Math.min(bottom, canvas.height));
    if (right - left < MIN_ROI_SIZE) {
      if (start.handle.includes('w')) left = right - MIN_ROI_SIZE;
      else right = left + MIN_ROI_SIZE;
    }
    if (bottom - top < MIN_ROI_SIZE) {
      if (start.handle.includes('n')) top = bottom - MIN_ROI_SIZE;
      else bottom = top + MIN_ROI_SIZE;
    }
    next = { x: left, y: top, w: right - left, h: bottom - top };
  }

  G.roi = clampRoi(next);
  updateRoiStatus();
  redraw();
}

function onCanvasPointerStart(e) {
  if (!G.frameOk || !G.roi) return;
  const p = evXY(e);
  if (isGeometryMode() && G.roiMode === 'locked') {
    const points = geometryCanvasPoints();
    const endpoint = points && (Math.hypot(p.x - points.top.x, p.y - points.top.y) <= HANDLE_SIZE ? 'top' : Math.hypot(p.x - points.bottom.x, p.y - points.bottom.y) <= HANDLE_SIZE ? 'bottom' : null);
    if (endpoint) { e.preventDefault(); G.geometryDrag = endpoint; return; }
  }
  if (G.roiMode !== 'editing') return;
  const handle = hitTestRoiHandle(p);
  if (!handle) return;
  e.preventDefault();
  G.dragStart = { x: p.x, y: p.y, roi: { ...G.roi }, handle };
}

function onCanvasPointerMove(e) {
  if (G.geometryDrag && G.roi && isGeometryMode()) {
    e.preventDefault();
    const p = evXY(e);
    G.geometryLine[G.geometryDrag] = {
      x: Math.max(0, Math.min(1, (p.x - G.roi.x) / G.roi.w)),
      y: Math.max(0, Math.min(1, (p.y - G.roi.y) / G.roi.h)),
    };
    redraw();
    return;
  }
  if (G.roiMode !== 'editing' || !G.dragStart) return;
  e.preventDefault();
  const p = evXY(e);
  applyRoiDrag(p);
}

function onCanvasPointerEnd(e) {
  if (G.geometryDrag) { e.preventDefault(); G.geometryDrag = null; return; }
  if (G.roiMode !== 'editing' || !G.dragStart) return;
  e.preventDefault();
  G.dragStart = null;
  updateRoiControls();
}

canvas.addEventListener('touchstart', onCanvasPointerStart, { passive: false });
canvas.addEventListener('touchmove', onCanvasPointerMove, { passive: false });
canvas.addEventListener('touchend', onCanvasPointerEnd, { passive: false });
canvas.addEventListener('mousedown', onCanvasPointerStart);
canvas.addEventListener('mousemove', onCanvasPointerMove);
canvas.addEventListener('mouseup', onCanvasPointerEnd);
canvas.addEventListener('mouseleave', onCanvasPointerEnd);

document.getElementById('btn-edit-roi').addEventListener('click', setRoiEditing);
document.getElementById('btn-lock-roi').addEventListener('click', lockRoi);

function updateRoiStatus() {
  const el = document.getElementById('roi-status');
  if (!G.roi || !isRoiValid()) {
    el.textContent = 'Chưa chọn vùng';
    el.style.color = 'var(--amber)';
    document.getElementById('canvas-hint').textContent = G.frameOk ? 'Bấm Chọn vùng chụp để bắt đầu' : '';
  } else if (G.roiMode === 'editing') {
    const v = toVideo(G.roi);
    el.textContent = 'Đang chỉnh: ' + v.w + '×' + v.h + ' px';
    el.style.color = 'var(--blue)';
    document.getElementById('canvas-hint').textContent = 'Kéo trong vùng để di chuyển, kéo nút để đổi kích thước';
  } else {
    const v = toVideo(G.roi);
    el.textContent = 'Đã khóa: ' + v.w + '×' + v.h + ' px';
    el.style.color = 'var(--green)';
    document.getElementById('canvas-hint').textContent = 'Vùng kiểm tra đã khóa';
  }
}

function updateRoiControls() {
  const editBtn = document.getElementById('btn-edit-roi');
  const lockBtn = document.getElementById('btn-lock-roi');
  const canEdit = G.frameOk;
  const editing = G.roiMode === 'editing';
  const valid = isRoiValid();
  editBtn.disabled = !canEdit;
  editBtn.textContent = valid ? 'Chỉnh sửa vùng' : 'Chọn vùng chụp';
  lockBtn.disabled = !editing || !valid;
  lockBtn.classList.toggle('ready', editing && valid);
  document.getElementById('btn-submit').disabled = !(valid && G.roiMode === 'locked');
}

function redraw() {
  if (!G.frameOk) return;
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  if (!G.roi) return;

  // Dim outside ROI
  ctx.fillStyle = 'rgba(0,0,0,.45)';
  ctx.fillRect(0, 0, canvas.width, G.roi.y);
  ctx.fillRect(0, G.roi.y, G.roi.x, G.roi.h);
  ctx.fillRect(G.roi.x + G.roi.w, G.roi.y, canvas.width - G.roi.x - G.roi.w, G.roi.h);
  ctx.fillRect(0, G.roi.y + G.roi.h, canvas.width, canvas.height - G.roi.y - G.roi.h);

  // ROI border
  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 2;
  ctx.setLineDash(G.roiMode === 'editing' ? [6, 3] : []);
  ctx.strokeRect(G.roi.x + 1, G.roi.y + 1, G.roi.w - 2, G.roi.h - 2);

  const bandConfig = getTriggerBandConfig();
  if (bandConfig && !isGeometryMode()) {
    // Trigger band fill
    const cy = G.roi.y + G.roi.h * Number(bandConfig.position_ratio);
    const ht = G.roi.h * Number(bandConfig.thickness_ratio) / 2;
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(34,211,238,.28)';
    ctx.fillRect(G.roi.x, cy - ht, G.roi.w, ht * 2);

    // Band border
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2;
    ctx.strokeRect(G.roi.x, cy - ht, G.roi.w, ht * 2);

    // Band centre line
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(G.roi.x, cy); ctx.lineTo(G.roi.x + G.roi.w, cy);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (isGeometryMode()) drawGeometryPreview();

  if (G.roiMode === 'editing') drawRoiHandles();
}

function drawGeometryPreview() {
  const points = geometryCanvasPoints();
  if (!points) return;
  const dx = points.bottom.x - points.top.x, dy = points.bottom.y - points.top.y;
  const length = Math.hypot(dx, dy);
  if (length < 1) return;
  const d = { x: dx / length, y: dy / length }, h = { x: -d.y, y: d.x };
  const geometry = G.runtimeConfig?.geometry_v2;
  const bandWidth = G.roi.w * Number(geometry?.defaults?.chain_band_width_ratio || .05);
  const offset = bandWidth / 2;
  ctx.fillStyle = 'rgba(250,204,21,.15)';
  ctx.beginPath();
  ctx.moveTo(points.top.x + h.x * offset, points.top.y + h.y * offset);
  ctx.lineTo(points.bottom.x + h.x * offset, points.bottom.y + h.y * offset);
  ctx.lineTo(points.bottom.x - h.x * offset, points.bottom.y - h.y * offset);
  ctx.lineTo(points.top.x - h.x * offset, points.top.y - h.y * offset);
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle = '#facc15'; ctx.lineWidth = 2.5; ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(points.top.x, points.top.y); ctx.lineTo(points.bottom.x, points.bottom.y); ctx.stroke();
  for (const point of [points.top, points.bottom]) {
    ctx.beginPath(); ctx.arc(point.x, point.y, 6, 0, Math.PI * 2); ctx.fillStyle = '#facc15'; ctx.fill();
    ctx.strokeStyle = '#713f12'; ctx.lineWidth = 1.5; ctx.stroke();
  }
  const trigger = geometry?.trigger;
  if (trigger) {
    const center = Number(trigger.center_ratio), thickness = Number(trigger.height_ratio);
    const midpoint = { x: points.top.x + dx * center, y: points.top.y + dy * center };
    const halfAlong = length * thickness / 2, halfAcross = Math.max(G.roi.w, G.roi.h) / 2;
    ctx.fillStyle = 'rgba(34,211,238,.20)';
    ctx.beginPath();
    ctx.moveTo(midpoint.x - d.x * halfAlong + h.x * halfAcross, midpoint.y - d.y * halfAlong + h.y * halfAcross);
    ctx.lineTo(midpoint.x + d.x * halfAlong + h.x * halfAcross, midpoint.y + d.y * halfAlong + h.y * halfAcross);
    ctx.lineTo(midpoint.x + d.x * halfAlong - h.x * halfAcross, midpoint.y + d.y * halfAlong - h.y * halfAcross);
    ctx.lineTo(midpoint.x - d.x * halfAlong - h.x * halfAcross, midpoint.y - d.y * halfAlong - h.y * halfAcross);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 1.5; ctx.stroke();
  }
}

function drawRoiHandles() {
  const x1 = G.roi.x, y1 = G.roi.y, x2 = G.roi.x + G.roi.w, y2 = G.roi.y + G.roi.h;
  const corners = [[x1, y1], [x2, y1], [x1, y2], [x2, y2]];
  const edges = [
    [(x1 + x2) / 2, y1], [(x1 + x2) / 2, y2],
    [x1, (y1 + y2) / 2], [x2, (y1 + y2) / 2],
  ];
  // Small white squares with green outline so the exact corner stays visible
  // and the handles cover as little of the image as possible.
  const draw = (pts, half) => {
    for (const [x, y] of pts) {
      ctx.beginPath();
      ctx.rect(x - half, y - half, half * 2, half * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#16a34a';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  };
  draw(corners, 3.5);
  draw(edges, 3);
}

/* ══════════════════════════════════════════════════════════════════
   COORDINATE CONVERSION
═══════════════════════════════════════════════════════════════════ */
function toVideo(roi) {
  const scaleX = G.videoW / canvas.width;
  const scaleY = G.videoH / canvas.height;
  return {
    x: Math.round(roi.x * scaleX),
    y: Math.round(roi.y * scaleY),
    w: Math.max(1, Math.round(roi.w * scaleX)),
    h: Math.max(1, Math.round(roi.h * scaleY)),
  };
}

/* ══════════════════════════════════════════════════════════════════
   SUBMIT → CREATE JOB
═══════════════════════════════════════════════════════════════════ */
document.getElementById('btn-submit').addEventListener('click', async () => {
  setErr('err-setup', '');
  const conveyor = document.getElementById('inp-conveyor').value.trim();
  if (!conveyor) { setErr('err-setup', 'Vui lòng nhập tên băng / máy'); return; }
  if (!G.file) { setErr('err-setup', 'Chưa chọn video'); return; }
  if (!G.roi || !isRoiValid()) { setErr('err-setup', 'Chưa chọn vùng kiểm tra, hoặc vùng quá nhỏ'); return; }
  if (G.roiMode !== 'locked') { setErr('err-setup', 'Vui lòng bấm ✓ để xác nhận vùng kiểm tra trước khi bắt đầu.'); return; }
  const inspectionMode = document.getElementById('inp-inspection-mode').value;
  if (inspectionMode === 'geometry_v2') ensureGeometryLine();
  G.conveyor = conveyor;
  updateSessionInfo();

  const mime = getMime(G.file);
  if (!mime) { setErr('err-setup', 'Định dạng video không được hỗ trợ (mp4/mov/webm)'); return; }

  const vRoi = toVideo(G.roi);
  // Clamp to actual frame bounds
  vRoi.w = Math.min(vRoi.w, G.videoW - vRoi.x);
  vRoi.h = Math.min(vRoi.h, G.videoH - vRoi.y);

  const body = {
    content_type: mime,
    size_bytes: G.file.size,
    roi: {
      x: vRoi.x, y: vRoi.y, w: vRoi.w, h: vRoi.h,
      frame_width: G.videoW,
      frame_height: G.videoH,
    },
    inspector_name: G.inspector,
    conveyor_name: G.conveyor,
    inspection_mode: inspectionMode,
  };
  if (body.inspection_mode === 'geometry_v2') {
    const geometry = G.runtimeConfig?.geometry_v2;
    if (!geometry?.available) { setErr('err-setup', geometry?.unavailable_reason || 'Geometry V2 hiện không khả dụng'); return; }
    body.geometry = {
      schema_version: geometry.input_schema_version,
      chain_centerline: {
        top: { x: G.geometryLine.top.x * vRoi.w, y: G.geometryLine.top.y * vRoi.h },
        bottom: { x: G.geometryLine.bottom.x * vRoi.w, y: G.geometryLine.bottom.y * vRoi.h },
      },
      chain_band_width_ratio: geometry.defaults.chain_band_width_ratio,
      motion_direction: geometry.defaults.motion_direction,
    };
  }

  try {
    const data = await api(G.localVideo ? '/api/local/jobs' : '/api/jobs', { method: 'POST', body: JSON.stringify(G.localVideo ? { ...body, path: G.localVideo } : body) });
    G.jobId = data.job_id;
    G.putUrl = data.presigned_put_url;
    if (G.localVideo) startPolling(); else doUpload(mime);
  } catch (e) {
    if (e.status !== 401) setErr('err-setup', e.message);
  }
});

document.getElementById('inp-inspection-mode').addEventListener('change', () => {
  const geometry = isGeometryMode();
  document.getElementById('geometry-centerline-help').style.display = geometry ? 'block' : 'none';
  if (geometry) ensureGeometryLine();
  redraw();
});

/* ══════════════════════════════════════════════════════════════════
   UPLOAD VIDEO → R2 (direct XHR PUT)
═══════════════════════════════════════════════════════════════════ */
function doUpload(mime) {
  show('s-uploading');
  const bar = document.getElementById('prog-bar');
  const pct = document.getElementById('prog-pct');
  bar.style.width = '0'; pct.textContent = '0%';

  const xhr = new XMLHttpRequest();
  xhr.open('PUT', G.putUrl);
  xhr.setRequestHeader('Content-Type', mime);

  xhr.upload.addEventListener('progress', e => {
    if (e.lengthComputable) {
      const p = Math.round(e.loaded / e.total * 100);
      bar.style.width = p + '%';
      pct.textContent = p + '%';
    }
  });

  xhr.addEventListener('load', async () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      try {
        await api('/api/jobs/' + G.jobId + '/upload-complete', { method: 'POST' });
        startPolling();
      } catch (e) {
        if (e.status !== 401) {
          show('s-setup');
          setErr('err-setup', 'Xác nhận upload thất bại: ' + e.message);
        }
      }
    } else {
      show('s-setup');
      setErr('err-setup', 'Upload thất bại (HTTP ' + xhr.status + ')');
    }
  });

  xhr.addEventListener('error', () => {
    show('s-setup');
    setErr('err-setup', 'Lỗi kết nối khi tải lên R2');
  });

  xhr.send(G.file);
}

/* ══════════════════════════════════════════════════════════════════
   POLLING
═══════════════════════════════════════════════════════════════════ */
function startPolling() {
  show('s-polling');
  setErr('err-poll', '');
  document.getElementById('poll-msg').textContent = 'Đang chờ xử lý…';
  document.getElementById('poll-detail').textContent = 'Xin chờ trong giây lát';
  clearTimeout(G.pollTid);
  doPoll();
}

async function doPoll() {
  try {
    const data = await api('/api/jobs/' + G.jobId + '/status');
    document.getElementById('poll-msg').textContent = data.message;
    document.getElementById('poll-detail').textContent = 'Trạng thái: ' + data.status;

    if (data.status === 'completed') { await loadResult(); return; }

    if (data.status === 'failed' || data.status === 'upload_expired') {
      // Stop spinner, show error, keep cancel button
      document.querySelector('#s-polling .spinner').style.display = 'none';
      setErr('err-poll', data.message);
      return;
    }
    G.pollTid = setTimeout(doPoll, 2500);
  } catch (e) {
    if (e.status === 401) return;
    G.pollTid = setTimeout(doPoll, 3000);
  }
}

document.getElementById('btn-cancel-poll').addEventListener('click', () => {
  clearTimeout(G.pollTid);
  document.querySelector('#s-polling .spinner').style.display = '';
  show('s-setup');
});

/* ══════════════════════════════════════════════════════════════════
   RESULT
═══════════════════════════════════════════════════════════════════ */
const DEFECT_LABEL_KEYS = ['bent_left', 'bent_right', 'bent_both', 'broken_left', 'broken_right', 'broken_center', 'broken', 'uncertain', 'other', '_unclassified'];
const DEFECT_LABELS = {
  bent_left:     'Cong trái',
  bent_right:    'Cong phải',
  bent_both:     'Cong cả 2',
  broken:        'Gãy',
  broken_left:   'Gãy bên trái',
  broken_right:  'Gãy bên phải',
  broken_center: 'Gãy giữa',
  uncertain:     'Cần xem xét',
  other:         'Không xác định rõ',
  _unclassified: 'Chưa phân loại',
};
const CALIB_REASONS = {
  length_too_short: 'Quá ngắn',
  length_too_long: 'Quá dài',
  width_too_small: 'Quá hẹp',
  width_too_large: 'Quá rộng',
};

function effectiveStatus(bar, normalBucket = false) {
  return bar.final_reviewed_status || bar.vision_status || (normalBucket ? 'normal' : bar.defect_type || '_unclassified');
}

function geometryDiagnosticsHtml(bar) {
  const analysis = bar.geometry_analysis;
  const diagnostics = analysis && analysis.diagnostics;
  if (!diagnostics) return '';
  const angle = value => Number.isFinite(value) ? `${Number(value).toFixed(2)}°` : '—';
  const machine = bar.vision_status || '—';
  const reviewed = bar.final_reviewed_status || 'chưa review';
  const states = [diagnostics.center_state, diagnostics.left_side_state, diagnostics.right_side_state]
    .map(value => value || '—').join(' / ');
  const reasons = Array.isArray(analysis.reason_codes) ? analysis.reason_codes.join(', ') : '—';
  return `<div class="defect-dims" style="margin-top:5px;font-size:12px;color:var(--muted)">
    Máy: ${esc(machine)} · Review: ${esc(reviewed)}<br>
    Center / Trái / Phải: ${esc(states)} · Góc T/P: ${esc(angle(diagnostics.left_angle_deg))} / ${esc(angle(diagnostics.right_angle_deg))}<br>
    Tilt/Kink: ${esc(angle(diagnostics.global_tilt_deg))} / ${esc(angle(diagnostics.center_kink_deg))} · Evidence: ${esc(reasons)}
  </div>`;
}

function defectSubtabKey(d) {
  return effectiveStatus(d);
}

async function loadResult() {
  const data = await api('/api/jobs/' + G.jobId + '/result');
  show('s-result');
  resetSaveButton();
  window.scrollTo(0, 0);
  updateScrollTopBtn();

  document.getElementById('stat-grid').innerHTML =
    statBox(data.total_bars, 'Tổng thanh', '') +
    statBox(data.defect_bars, 'Thanh lỗi', data.defect_bars > 0 ? 'bad' : 'ok');

  const defects = data.defects || [];
  const normals = data.normals || [];
  G.allDefects = defects;
  G.allNormals = normals;
  G.reviewRevision = Number(data.review?.revision || 0);

  G.lightboxIndex = -1;

  document.getElementById('no-defect').style.display = defects.length === 0 ? 'block' : 'none';

  const hasAny = defects.length > 0 || normals.length > 0;
  document.getElementById('gallery-tabs').style.display = hasAny ? '' : 'none';
  document.getElementById('tab-count-defect').textContent = defects.length;
  document.getElementById('tab-count-normal').textContent = normals.length;

  _initDefectSubtabs();
  const firstKey = DEFECT_LABEL_KEYS.find(k => _filterBySubtab(defects, k).length > 0) || DEFECT_LABEL_KEYS[0];
  switchDefectSubtab(firstKey);
  _renderNormals();

  switchTab(defects.length > 0 ? 'defect' : 'normal');
}

function _filterBySubtab(items, key) {
  return items.filter(d => defectSubtabKey(d) === key);
}

function _initDefectSubtabs() {
  document.getElementById('defect-subtabs').innerHTML = DEFECT_LABEL_KEYS.map(key => {
    const count = _filterBySubtab(G.allDefects, key).length;
    return `<button class="subtab-btn" data-key="${key}" onclick="switchDefectSubtab('${key}')"
      ${count === 0 ? 'disabled' : ''}>${DEFECT_LABELS[key]} (${count})</button>`;
  }).join('');
}

function switchDefectSubtab(key) {
  G.activeDefectSubtab = key;
  document.querySelectorAll('#defect-subtabs .subtab-btn').forEach(btn =>
    btn.classList.toggle('active', btn.dataset.key === key)
  );
  const filtered = _filterBySubtab(G.allDefects, key);
  G.defectLightboxItems = filtered.filter(d => d.snapshot_url).map(d => ({
    src: d.snapshot_url,
    meta: `${d.bar_id || ('Track #' + d.track_id)} · ${DEFECT_LABELS[effectiveStatus(d)] || ''}`,
  }));
  if (G.activeTab === 'defect') { G.lightboxItems = G.defectLightboxItems; G.lightboxIndex = -1; }
  let di = 0;
  document.getElementById('defect-list').innerHTML = filtered.map(d => {
    const idx = d.snapshot_url ? di++ : -1;
    const ekey = 'ed:' + esc(d.bar_id || String(d.track_id));
    const status = effectiveStatus(d);
    const vlmBadge = `<span class="tag ${status === 'uncertain' ? '' : 'tag-defect'}" style="${status === 'uncertain' ? 'background:#f59e0b;color:#111827' : ''}">${DEFECT_LABELS[status] || esc(status)}</span>`;
    const calibBadges = (d.reasons || []).map(r =>
      `<span class="tag" style="font-size:11px;opacity:.7">${CALIB_REASONS[r] || esc(r)}</span>`
    ).join('');
    const editOptions = ['bent_left','bent_right','bent_both','broken_left','broken_right','broken_center','broken','other'].map(k =>
      `<button class="edit-type-btn" onclick="applyCorrection('d:${esc(d.bar_id||String(d.track_id))}','${k}')">${DEFECT_LABELS[k]}</button>`
    ).join('') + `<button class="edit-type-btn ok-btn" onclick="applyCorrection('d:${esc(d.bar_id||String(d.track_id))}','normal')">Bình thường</button>`;
    return `
  <div class="defect-card">
    ${d.snapshot_url
      ? `<img src="${esc(d.snapshot_url)}" loading="lazy" alt="${esc(d.bar_id || String(d.track_id))}" class="defect-preview" data-index="${idx}">`
      : `<div style="background:#1e293b;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;color:#64748b;font-size:13px">Không có ảnh</div>`}
    <div class="defect-info">
      <div class="defect-tags">${vlmBadge}${calibBadges}<button class="btn-edit-toggle" onclick="toggleCardEdit('${ekey}')">✏ Sửa</button></div>
      <div class="defect-dims">${esc(d.bar_id || ('Track #' + d.track_id))} &nbsp;·&nbsp; Dài: ${d.legacy_measurements_available === false ? '—' : (d.length || 0).toFixed(1)} &nbsp;·&nbsp; Rộng: ${d.legacy_measurements_available === false ? '—' : (d.width || 0).toFixed(1)}</div>
      ${geometryDiagnosticsHtml(d)}
      <div class="card-edit-panel" id="${ekey}" style="display:none">${editOptions}</div>
    </div>
  </div>`;
  }).join('');
}

function _renderNormals() {
  let ni = 0;
  document.getElementById('normal-list').innerHTML = G.allNormals.map(n => {
    const idx = n.snapshot_url ? ni++ : -1;
    const ekey = 'en:' + n.track_id;
    const editOptions = ['bent_left','bent_right','bent_both','broken_left','broken_right','broken_center','broken','other'].map(k =>
      `<button class="edit-type-btn" onclick="applyCorrection('n:${n.track_id}','${k}')">${DEFECT_LABELS[k]}</button>`
    ).join('');
    return `
  <div class="defect-card">
    ${n.snapshot_url
        ? `<img src="${esc(n.snapshot_url)}" loading="lazy" alt="Track ${esc(String(n.track_id))}" class="normal-preview" data-index="${idx}">`
        : `<div style="background:#1e293b;aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;color:#64748b;font-size:13px">Không có ảnh</div>`}
    <div class="defect-info">
      <div class="defect-tags"><button class="btn-edit-toggle" onclick="toggleCardEdit('${ekey}')">✏ Sửa lỗi</button></div>
      <div class="defect-dims">Track #${n.track_id} &nbsp;·&nbsp; Dài: ${n.legacy_measurements_available === false ? '—' : (n.length || 0).toFixed(1)} &nbsp;·&nbsp; Rộng: ${n.legacy_measurements_available === false ? '—' : (n.width || 0).toFixed(1)}</div>
      ${geometryDiagnosticsHtml(n)}
      <div class="card-edit-panel" id="${ekey}" style="display:none">${editOptions}</div>
    </div>
  </div>`;
  }).join('');
}

function toggleCardEdit(key) {
  const el = document.getElementById(key);
  if (!el) return;
  el.style.display = el.style.display === 'none' ? '' : 'none';
}

function applyCorrection(key, newType) {
  const isDefect = key.startsWith('d:');
  const id = key.slice(2);

  if (isDefect) {
    const idx = G.allDefects.findIndex(d => (d.bar_id || String(d.track_id)) === id);
    if (idx === -1) return;
    const bar = G.allDefects[idx];
    if (newType === 'normal') {
      G.allDefects.splice(idx, 1);
      G.allNormals.push({ ...bar, final_reviewed_status: 'normal' });
    } else {
      G.allDefects[idx] = { ...bar, final_reviewed_status: newType };
    }
  } else {
    const trackId = parseInt(id);
    const idx = G.allNormals.findIndex(n => n.track_id === trackId);
    if (idx === -1) return;
    const bar = G.allNormals[idx];
    G.allNormals.splice(idx, 1);
    G.allDefects.push({ ...bar, final_reviewed_status: newType });
  }

  const defectCount = G.allDefects.length;
  document.getElementById('tab-count-defect').textContent = defectCount;
  document.getElementById('tab-count-normal').textContent = G.allNormals.length;
  document.getElementById('no-defect').style.display = defectCount === 0 ? 'block' : 'none';
  document.getElementById('stat-grid').innerHTML =
    statBox(G.allDefects.length + G.allNormals.length, 'Tổng thanh', '') +
    statBox(defectCount, 'Thanh lỗi', defectCount > 0 ? 'bad' : 'ok');
  _initDefectSubtabs();
  const activeKey = G.activeDefectSubtab && _filterBySubtab(G.allDefects, G.activeDefectSubtab).length > 0
    ? G.activeDefectSubtab
    : (DEFECT_LABEL_KEYS.find(k => _filterBySubtab(G.allDefects, k).length > 0) || DEFECT_LABEL_KEYS[0]);
  switchDefectSubtab(activeKey);
  _renderNormals();
}

const VALID_DEFECT_TYPES = ['normal', 'bent_left', 'bent_right', 'bent_both', 'broken_left', 'broken_right', 'broken_center', 'broken'];

function collectCorrections() {
  const out = [];
  for (const d of G.allDefects) out.push({ track_id: d.track_id, defect_type: effectiveStatus(d) });
  for (const n of G.allNormals) out.push({ track_id: n.track_id, defect_type: effectiveStatus(n, true) });
  return out;
}

function hasUnclassified() {
  return G.allDefects.some(d => !VALID_DEFECT_TYPES.includes(effectiveStatus(d)));
}

function setSaveMsg(text, isErr) {
  const el = document.getElementById('msg-save');
  el.style.display = 'block';
  el.textContent = text;
  el.classList.toggle('err', !!isErr);
}

function resetSaveButton() {
  const btn = document.getElementById('btn-save-report');
  btn.disabled = false;
  btn.textContent = 'Lưu kết quả';
  const el = document.getElementById('msg-save');
  el.style.display = 'none';
  el.textContent = '';
  el.classList.remove('err');
}

async function saveReport() {
  if (hasUnclassified()) {
    setSaveMsg('Còn thanh chưa phân loại — hãy phân loại hết (trái/phải/2 bên/gãy) hoặc đánh dấu bình thường trước khi lưu.', true);
    return;
  }
  const btn = document.getElementById('btn-save-report');
  btn.disabled = true;
  btn.textContent = 'Đang lưu...';
  try {
    const res = await api('/api/jobs/' + G.jobId + '/report', {
      method: 'POST',
      body: JSON.stringify({
        inspector_name: G.inspector,
        conveyor_name: G.conveyor,
        expected_review_revision: G.reviewRevision,
        corrections: collectCorrections(),
      }),
    });
    btn.textContent = 'Đã lưu ✓';
    setSaveMsg('Đã lưu PDF: ' + res.filename + ' · Đã cập nhật Excel: ' + res.excel_filename, false);
  } catch (e) {
    btn.disabled = false;
    btn.textContent = 'Lưu kết quả';
    setSaveMsg('Lỗi khi lưu: ' + e.message, true);
  }
}

document.getElementById('btn-save-report').addEventListener('click', saveReport);

/* ── Scroll-to-top button (result screen, works for both galleries) ───────── */
const _scrollTopBtn = document.getElementById('btn-scroll-top');
function _pageScrollTop() {
  return window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0;
}
function _onResultScreen() {
  return document.getElementById('s-result').style.display !== 'none';
}
function updateScrollTopBtn() {
  if (!_scrollTopBtn) return;
  _scrollTopBtn.classList.toggle('show', _onResultScreen() && _pageScrollTop() > 300);
}
if (_scrollTopBtn) {
  // The page (window) scrolls — the .s screens use min-height:100svh, so content
  // taller than the viewport scrolls the document, not the inner .body.
  window.addEventListener('scroll', updateScrollTopBtn, { passive: true });
  _scrollTopBtn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

function switchTab(tab) {
  G.activeTab = tab;
  ['defect', 'normal'].forEach(t => {
    document.getElementById('tab-btn-' + t).classList.toggle('active', t === tab);
    document.getElementById('panel-' + t).style.display = t === tab ? '' : 'none';
  });
  G.normalLightboxItems = G.allNormals.filter(n => n.snapshot_url).map(n => ({
    src: n.snapshot_url,
    meta: `Track #${n.track_id} · Dài: ${(n.length || 0).toFixed(1)} · Rộng: ${(n.width || 0).toFixed(1)}`,
  }));
  G.lightboxItems = tab === 'defect' ? G.defectLightboxItems : G.normalLightboxItems;
  G.lightboxIndex = -1;
  updateScrollTopBtn();
}

function statBox(n, lbl, cls) {
  return `<div class="stat-box"><div class="stat-num ${cls}">${n}</div><div class="stat-lbl">${lbl}</div></div>`;
}

document.getElementById('btn-new-job').addEventListener('click', () => {
  G.jobId = null; G.putUrl = null; G.file = null; G.localVideo = null;
  G.roi = null; G.frameOk = false; G.roiMode = 'locked';
  G.dragStart = null;
  G.conveyor = '';
  canvas.classList.remove('editing');
  document.getElementById('inp-video').value = '';
  document.getElementById('inp-conveyor').value = '';   // re-enter band each inspection
  document.getElementById('roi-section').style.display = 'none';
  document.getElementById('video-name').style.display = 'none';
  setErr('err-setup', '');
  resetSaveButton();
  updateSessionInfo();
  updateRoiControls();
  show('s-setup');
});

/* ══════════════════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════════════════ */
function getMime(file) {
  if (file.type && file.type.startsWith('video/')) {
    const t = file.type;
    if (['video/mp4', 'video/webm', 'video/quicktime'].includes(t)) return t;
  }
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  return { mp4: 'video/mp4', m4v: 'video/mp4', mov: 'video/quicktime', webm: 'video/webm' }[ext] || null;
}

function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function showLightboxItem(index) {
  if (index < 0 || index >= G.lightboxItems.length) return false;
  const item = G.lightboxItems[index];
  G.lightboxIndex = index;
  document.getElementById('lightbox-image').src = item.src;
  document.getElementById('lightbox-meta').textContent = item.meta;
  document.getElementById('lightbox-prev').disabled = index === 0;
  document.getElementById('lightbox-next').disabled = index === G.lightboxItems.length - 1;
  return true;
}

function openLightbox(index) {
  if (!showLightboxItem(index)) return;
  document.getElementById('lightbox').classList.add('open');
  document.getElementById('lightbox').setAttribute('aria-hidden', 'false');
  G.lightboxOpen = true;
}

function moveLightbox(delta) {
  if (!G.lightboxOpen) return;
  showLightboxItem(G.lightboxIndex + delta);
}

function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
  document.getElementById('lightbox').setAttribute('aria-hidden', 'true');
  document.getElementById('lightbox-image').src = '';
  document.getElementById('lightbox-meta').textContent = '';
  G.lightboxOpen = false;
  G.lightboxIndex = -1;
}

document.getElementById('defect-list').addEventListener('click', e => {
  const img = e.target.closest('.defect-preview');
  if (!img) return;
  openLightbox(Number(img.dataset.index));
});

document.getElementById('normal-list').addEventListener('click', e => {
  const img = e.target.closest('.normal-preview');
  if (!img) return;
  openLightbox(Number(img.dataset.index));
});

document.getElementById('lightbox-close').addEventListener('click', closeLightbox);
document.getElementById('lightbox-prev').addEventListener('click', () => moveLightbox(-1));
document.getElementById('lightbox-next').addEventListener('click', () => moveLightbox(1));
document.getElementById('lightbox').addEventListener('click', e => {
  if (e.target.id === 'lightbox') closeLightbox();
});
document.addEventListener('keydown', e => {
  if (!G.lightboxOpen) return;
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowLeft') moveLightbox(-1);
  if (e.key === 'ArrowRight') moveLightbox(1);
});

/* ── Init ───────────────────────────────────────────────────────────────── */
async function init() {
  updateRoiControls();
  if (G.inspector) document.getElementById('inp-inspector').value = G.inspector;
  if (!G.token) {
    show('s-login');
    return;
  }
  try {
    await loadRuntimeConfig();
    show('s-setup');
  } catch (e) {
    if (e.status !== 401) {
      localStorage.removeItem('dc_token');
      G.token = '';
      show('s-login');
      setErr('err-login', 'Không tải được cấu hình hệ thống: ' + e.message);
    }
  }
}

init();
