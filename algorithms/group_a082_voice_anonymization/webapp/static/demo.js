// Demo page: pick input, pick a system, call /api/anonymize, show both signals.
const $ = (id) => document.getElementById(id);
const COLORS = { orig: '#1F5FA8', anon: '#C25B12' };
const DETAIL_NAMES = {
  targets: '目标说话人', weights: '池化权重', length_factor: '长度因子', transcript: '转写',
  piper_speaker: '合成音色', admixture_p: '混合概率 p', drawn: '本句抽到', target: '目标说话人',
};
const state = { input: null, inputName: '', system: null, threshold: 0.25, maxSeconds: 30, recorder: null };

function showError(msg) { const e = $('error'); e.textContent = msg; e.hidden = !msg; }

function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') n.className = v; else if (k.startsWith('on')) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v);
  }
  for (const c of children) n.append(c);
  return n;
}

// A track: play button, title, waveform (click to seek), optional download.
function makeTrack(label, sub, samples, url, color, downloadName) {
  const audio = new Audio(url);
  const canvas = el('canvas', { 'aria-label': label + '波形' });
  const icon = () => audio.paused
    ? '<svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4l13 8-13 8z" fill="#FFF"/></svg>'
    : '<svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 4h4v16H6zM14 4h4v16h-4z" fill="#FFF"/></svg>';
  const play = el('button', { type: 'button', class: 'play', 'aria-label': '播放' + label, style: `background:${color}` });
  play.innerHTML = icon();
  play.addEventListener('click', () => { audio.paused ? audio.play() : audio.pause(); });
  audio.addEventListener('play', () => { play.innerHTML = icon(); play.setAttribute('aria-label', '暂停' + label); });
  audio.addEventListener('pause', () => { play.innerHTML = icon(); play.setAttribute('aria-label', '播放' + label); });
  const redraw = () => drawWave(canvas, samples, color, audio.duration ? audio.currentTime / audio.duration : 0);
  audio.addEventListener('timeupdate', redraw);
  audio.addEventListener('ended', () => { audio.currentTime = 0; redraw(); });
  canvas.addEventListener('click', (e) => {
    if (!audio.duration) return;
    audio.currentTime = (e.offsetX / canvas.clientWidth) * audio.duration; redraw();
  });
  const head = el('div', { class: 'head' }, play, el('div', { class: 'meta' }, el('b', {}, label), el('span', {}, sub)));
  if (downloadName) head.append(el('a', { class: 'btn small', href: url, download: downloadName }, '下载 WAV'));
  const node = el('div', { class: 'track' }, head, canvas);
  requestAnimationFrame(redraw);
  window.addEventListener('resize', redraw);
  return node;
}

function setInput(samples, name) {
  if (samples.length > state.maxSeconds * SR) {
    showError(`音频超过 ${state.maxSeconds} 秒，请截短后再试。`); return;
  }
  showError('');
  state.input = samples; state.inputName = name;
  const box = $('input-track');
  box.replaceChildren(makeTrack('已选输入', `${name} · ${(samples.length / SR).toFixed(1)} s`, samples,
    URL.createObjectURL(encodeWav(samples)), COLORS.orig));
  box.hidden = false;
}

async function loadFile(file) {
  try { setInput(await decodeToMono16k(await file.arrayBuffer()), file.name); }
  catch (e) { showError('无法解码这个文件：' + e.message); }
}

function renderSystems(groups) {
  const fs = $('systems');
  for (const g of groups) {
    const wrap = el('div', {}, el('div', { class: 'group-name' }, g.name));
    for (const it of g.items) {
      const b = el('button', { type: 'button', class: 'sys', 'aria-pressed': 'false', 'data-key': it.key },
        el('span', { class: 'num' }, it.num),
        el('span', {}, el('span', { class: 'name' }, it.name), el('span', { class: 'desc' }, it.desc)));
      b.addEventListener('click', () => selectSystem(it.key));
      wrap.append(b);
    }
    fs.append(wrap);
  }
  const first = groups[0]?.items[0]?.key;
  if (first) selectSystem(first);
  else showError('服务器上没有可用的系统（检查 config.yaml 里的模型路径）。');
}

function selectSystem(key) {
  state.system = key;
  document.querySelectorAll('.sys').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.key === key)));
  $('mixbox').hidden = key !== 'mix';
}

function detailRows(r) {
  const rows = [['系统', r.system_name], ['种子', r.seed]];
  for (const [k, v] of Object.entries(r.details || {})) {
    if (k === 'drawn') { rows.push([DETAIL_NAMES[k], v === 'tts' ? '级联' : 'kNN-VC']); continue; }
    rows.push([DETAIL_NAMES[k] || k, Array.isArray(v) ? v.join(', ') : String(v)]);
  }
  rows.push(['时长（输入 → 输出）', `${r.seconds_in} s → ${r.seconds_out} s`], ['处理时间', `${r.elapsed} s`]);
  return rows.map(([k, v]) => el('div', { class: 'kv' }, el('span', {}, k), el('span', {}, v)));
}

async function run() {
  if (!state.input) { showError('请先上传、录制或选一段示例语音。'); return; }
  showError('');
  const go = $('go');
  go.disabled = true; go.replaceChildren(el('span', { class: 'spinner', 'aria-hidden': 'true' }), ' 处理中…');
  try {
    const fd = new FormData();
    fd.append('audio', encodeWav(state.input), 'input.wav');
    fd.append('system', state.system);
    fd.append('seed', $('seed').value);
    fd.append('p', $('p').value);
    fd.append('similarity', String($('sim').checked));
    const resp = await fetch('api/anonymize', { method: 'POST', body: fd });
    const r = await resp.json();
    if (!resp.ok) throw new Error(r.detail || resp.statusText);
    $('seed').value = r.seed;
    const anonUrl = `api/result/${r.id}/anon.wav`;
    const anonSamples = await decodeToMono16k(await (await fetch(anonUrl)).arrayBuffer());
    $('tracks').replaceChildren(
      makeTrack('原声', `${state.inputName} · ${r.seconds_in} s`, state.input, `api/result/${r.id}/orig.wav`, COLORS.orig),
      makeTrack('匿名化', `${r.system_name} · ${r.seconds_out} s`, anonSamples, anonUrl, COLORS.anon, `anonymized_${r.system}_${r.seed}.wav`));
    const sim = r.similarity;
    $('simcard').hidden = sim === null;
    if (sim !== null) {
      $('simval').textContent = sim.toFixed(3);
      $('simfill').style.width = `${Math.max(0, Math.min(1, sim)) * 100}%`;
      $('simmark').style.left = `${state.threshold * 100}%`;
      $('simnote').textContent = `竖线：${state.threshold}，SpeechBrain 这个模型判定"同一说话人"的默认阈值。原声对原声为 1。`;
    }
    $('details').replaceChildren(...detailRows(r));
    $('spec-orig').src = `api/result/${r.id}/orig.png`;
    $('spec-anon').src = `api/result/${r.id}/anon.png`;
    $('empty').hidden = true; $('out').hidden = false;
  } catch (e) {
    showError('处理失败：' + e.message);
  } finally {
    go.disabled = false; go.textContent = '匿名化';
  }
}

async function init() {
  const drop = $('drop');
  $('file').addEventListener('change', (e) => e.target.files[0] && loadFile(e.target.files[0]));
  ['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(t => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => e.dataTransfer.files[0] && loadFile(e.dataTransfer.files[0]));
  $('p').addEventListener('input', (e) => { $('ptext').textContent = Number(e.target.value).toFixed(3); });
  $('go').addEventListener('click', run);
  $('rec').addEventListener('click', async () => {
    const btn = $('rec');
    try {
      if (!state.recorder) {
        state.recorder = new PcmRecorder(); await state.recorder.start();
        btn.classList.add('recording'); btn.querySelector('span').textContent = '停止录音';
      } else {
        const samples = await state.recorder.stop(); state.recorder = null;
        btn.classList.remove('recording'); btn.querySelector('span').textContent = '录音';
        setInput(samples, '录音');
      }
    } catch (e) { state.recorder = null; showError('无法使用麦克风：' + e.message); }
  });
  try {
    const sys = await (await fetch('api/systems')).json();
    state.threshold = sys.threshold; state.maxSeconds = sys.max_seconds;
    if (!sys.similarity) { $('sim').checked = false; $('sim').parentElement.hidden = true; }
    renderSystems(sys.groups);
    const smp = await (await fetch('api/samples')).json();
    for (const s of smp.samples) {
      $('sources').append(el('button', { type: 'button', class: 'btn small', onclick: async () => {
        try { setInput(await decodeToMono16k(await (await fetch(s.url)).arrayBuffer()), s.name); }
        catch (e) { showError('示例加载失败：' + e.message); }
      } }, '示例：' + s.name));
    }
  } catch (e) { showError('连不上后端：' + e.message); }
}
init();
