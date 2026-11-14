// Read only completed epoch records; no interpolation of missing results.
function renderTrainingCharts() {
  if (!data) return;
  const compare = document.getElementById('compare');
  const selected = compare.value;
  compare.replaceChildren(new Option('不比较', ''));
  for (const row of data.rows) compare.add(new Option(row.name, row.name));
  if (data.rows.some(r => r.name === selected)) compare.value = selected;
  const primary = data.rows.find(r => r.name === document.getElementById('model').value);
  const secondary = data.rows.find(r => r.name === compare.value);
  const runs = [primary, secondary].filter(Boolean);
  const best = document.getElementById('best');
  best.textContent = runs.map(r => r.best_metrics
    ? `${r.name} 已完成路径的最佳验证轮：BER=${r.best_metrics.ber.toPrecision(5)}，SNR=${r.best_metrics.snr.toFixed(3)} dB，NMR=${r.best_metrics.nmr_db.toFixed(3)} dB`
    : `${r.name}：尚无完成路径的最佳指标汇总`).join('\n');
  const container = document.getElementById('charts');
  container.replaceChildren();
  for (const [key, title] of [['ber', '验证 BER ↓'], ['snr', '验证 SNR dB ↑'],
                              ['nmr_db', '验证 NMR dB ↓'], ['loss', '验证 loss（仅同目标可比）']]) {
    const box = document.createElement('div');
    const label = document.createElement('strong'); label.textContent = title; box.append(label);
    const series = runs.map(r => r.history.filter(h => Number.isFinite(h[key])).map(h => [h.epoch+1, h[key]]));
    const points = series.flat();
    if (!points.length) {
      const empty = document.createElement('p'); empty.textContent = '等待完整轮次指标'; box.append(empty);
      container.append(box); continue;
    }
    const xmin = Math.min(...points.map(p => p[0])), xmax = Math.max(xmin+1, ...points.map(p => p[0]));
    const lo = Math.min(...points.map(p => p[1])), hi = Math.max(...points.map(p => p[1]));
    const pad = Math.max((hi-lo)*.1, Math.abs(hi)*.01, 1e-7);
    const ymin=lo-pad, ymax=hi+pad;
    const x=v=>60+(v-xmin)/(xmax-xmin)*360, y=v=>180-(v-ymin)/(ymax-ymin)*150;
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
    svg.setAttribute('viewBox','0 0 450 220'); svg.setAttribute('role','img'); svg.setAttribute('aria-label',title);
    function add(tag, attrs, text) {
      const node=document.createElementNS(svg.namespaceURI,tag);
      for(const [k,v] of Object.entries(attrs)) node.setAttribute(k,String(v));
      if(text!==undefined) node.textContent=text;
      svg.append(node); return node;
    }
    add('path',{d:'M60 25 V180 H425',stroke:'#777',fill:'none'});
    for(const value of [lo, hi]) add('text',{x:4,y:y(value), 'font-size':11}, value.toPrecision(3));
    for(const value of [xmin,xmax]) add('text',{x:x(value),y:202,'font-size':12},value);
    add('text',{x:210,y:218,'font-size':12},'epoch');
    series.forEach((values,index)=>{
      const color=['#1565c0','#d26913'][index];
      add('polyline',{points:values.map(p=>`${x(p[0])},${y(p[1])}`).join(' '),fill:'none',stroke:color,'stroke-width':2});
      for(const point of values) {
        const dot=add('circle',{cx:x(point[0]),cy:y(point[1]),r:3,fill:color});
        const tip=document.createElementNS(svg.namespaceURI,'title');
        tip.textContent=`${runs[index].name} epoch ${point[0]}: ${point[1]}`; dot.append(tip);
      }
    });
    box.append(svg); container.append(box);
  }
}
document.getElementById('model').addEventListener('change',renderTrainingCharts);
document.getElementById('compare').addEventListener('change',renderTrainingCharts);
if(typeof data!=='undefined') renderTrainingCharts();
