// Results page: Table 1, privacy-utility scatter and the admixture comparison, from /api/results.
const GROUP_COLOR = (row) => row === '0' ? '#15202B' : ['1', '2', '3'].includes(row) ? '#1F5FA8'
  : (row === '4' || row.startsWith('mix')) ? '#C25B12' : '#6B4FA0';
const SHORT = (row) => row.startsWith('mix') ? 'p' + row.slice(3).replace('0.', '.') : row;
const fmt = (x) => (x === null || x === undefined) ? null : Number(x).toFixed(2);
let DATA = null;
let VIEW = 'paper';

function cell(text, cls) {
  const td = document.createElement('td');
  td.className = cls || '';
  td.textContent = text;
  return td;
}

function verdictCell(v) {
  const td = document.createElement('td');
  const span = document.createElement('span');
  span.style.cssText = 'display:inline-flex;align-items:center;gap:6px;font-size:13px;color:#3D4B59';
  let color = '#9AA8B6', text = '待测';
  if (v) {
    const keys = Object.entries(v).filter(([k]) => ['EER', 'UAR', 'WER'].includes(k));
    const okAll = keys.every(([, s]) => s === 'ok');
    color = okAll ? '#1D7A46' : '#B7791F';
    text = okAll ? '复现' : keys.filter(([, s]) => s !== 'ok').map(([k, s]) => `${k} ${s}`).join('；');
    if (v.qualitative) text += (text ? '；' : '') + v.qualitative;
  }
  const dot = document.createElement('i');
  dot.className = 'dot'; dot.style.background = color;
  span.append(dot, text);
  td.append(span);
  return td;
}

function renderTable(rows) {
  const body = document.getElementById('rows');
  body.replaceChildren(...rows.map(r => {
    const tr = document.createElement('tr');
    tr.append(cell(r.row.startsWith('mix') ? '1+4' : r.row, 'id'), cell(r.name));
    for (const k of ['EER', 'UAR', 'WER']) {
      tr.append(cell(fmt(r.paper[k]), 'num'));
      const o = r.ours ? fmt(r.ours[k]) : null;
      tr.append(cell(o ?? '待测', 'num' + (o ? '' : ' pending')));
    }
    tr.append(verdictCell(r.verdict));
    return tr;
  }));
}

function point(eer, uar, color, label, ours) {
  const d = document.createElement('div');
  d.className = 'pt' + (ours ? ' ours' : '');
  d.style.left = `${Math.max(0, Math.min(100, eer / 50 * 100))}%`;
  d.style.top = `${Math.max(0, Math.min(100, (75 - uar) / 50 * 100))}%`;
  const i = document.createElement('i');
  i.style.background = color; i.style.boxShadow = `0 0 0 1px ${color}`;
  if (ours) i.style.borderColor = color;
  const b = document.createElement('b');
  b.textContent = label + (ours ? '′' : '');
  d.append(i, b);
  return d;
}

function renderPlot() {
  const plot = document.getElementById('plot');
  const pts = [];
  let anyOurs = false;
  for (const r of DATA.rows) {
    const c = GROUP_COLOR(r.row);
    if (VIEW !== 'ours') pts.push(point(r.paper.EER, r.paper.UAR, c, SHORT(r.row), false));
    if (VIEW !== 'paper' && r.ours && r.ours.EER !== null && r.ours.UAR !== null) {
      pts.push(point(r.ours.EER, r.ours.UAR, c, SHORT(r.row), true)); anyOurs = true;
    }
  }
  plot.replaceChildren(...pts);
  document.getElementById('nodata').hidden = VIEW === 'paper' || anyOurs;
  document.querySelectorAll('.seg button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.view === VIEW)));
}

function bar(width, color, text) {
  const div = document.createElement('div');
  const i = document.createElement('i');
  i.style.width = width; i.style.background = color;
  div.append(i, text);
  return div;
}

function renderMix() {
  const byRow = Object.fromEntries(DATA.rows.map(r => [r.row, r]));
  const k = byRow['1'].paper.EER, t = byRow['4'].paper.EER;
  const sep = (DATA.bonus && DATA.bonus.mix_separated) || {};
  const out = ['0.2', '0.325', '0.4'].map(p => {
    const lin = (1 - p) * k + p * t, act = byRow['mix' + p].paper.EER;
    const w = (x) => `${(x / 50 * 100).toFixed(1)}%`;
    const row = document.createElement('div');
    row.className = 'mixrow';
    const lab = document.createElement('span');
    lab.className = 'mono'; lab.style.cssText = 'font-size:14px;font-weight:500';
    lab.textContent = `p = ${p}`;
    const bars = document.createElement('div');
    bars.className = 'bars';
    bars.append(bar(w(lin), '#9AA8B6', `线性插值 ${lin.toFixed(2)}`), bar(w(act), '#C25B12', `论文实测 ${act.toFixed(2)}`),
      sep[p] !== undefined ? bar(w(sep[p]), '#1D7A46', `攻击者分离后 ${Number(sep[p]).toFixed(2)}`)
                           : bar('2px', '#1D7A46', '攻击者分离后 待测'));
    row.append(lab, bars);
    return row;
  });
  document.getElementById('mix').replaceChildren(...out);
}

document.querySelectorAll('.seg button').forEach(b => b.addEventListener('click', () => { VIEW = b.dataset.view; renderPlot(); }));
fetch('api/results').then(r => r.json()).then(d => { DATA = d; renderTable(d.rows); renderPlot(); renderMix(); })
  .catch(e => { document.getElementById('rows').innerHTML = ''; alert('加载结果失败：' + e.message); });
