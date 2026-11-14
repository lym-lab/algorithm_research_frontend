"""Local training progress and explicit stop/resume controls."""
import argparse
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
from scripts.training_control import TrainingControl

ROOT = Path(__file__).resolve().parents[1]
CONTROL = TrainingControl(ROOT)
TOKEN = secrets.token_urlsafe(32)


def read(path, default):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return default  # A training process may currently be writing this file.


def snapshot():
    paper = ROOT/'runs/paper'
    execution = CONTROL.status()
    live_pids = {pid for p in execution['processes'] for pid in p.get('pids', [p['pid']])}
    rows = []
    live = []
    for path in sorted(paper.rglob('config.json')):
        folder = path.parent
        cfg = read(path, {})
        history = read(folder/'history.json', [])
        latest = history[-1] if history else {}
        result = read(folder/'result.json', {})
        progress = read(folder/'progress.json', {})
        if progress:
            progress = dict(progress, name=str(folder.relative_to(paper)),
                            active=progress.get('pid') in live_pids)
            live.append(progress)
        rows.append(dict(name=str(folder.relative_to(paper)), warmup=cfg.get('warmup_only', False),
                         complete=(folder/'result.json').exists(), epochs=latest.get('epoch', -1)+1,
                         limit=cfg.get('epochs'), history=history, best_metrics=result.get('metrics'),
                         updated=datetime.fromtimestamp((folder/'history.json' if history else path).stat().st_mtime).isoformat(timespec='seconds')))
    return dict(now=datetime.now().isoformat(timespec='seconds'),
                pipeline=read(ROOT/'outputs/pipeline_status.json', {}),
                completed=sum(r['complete'] and not r['warmup'] for r in rows),
                total=len(read(paper/'queue.json', [])), rows=rows, execution=execution,
                epoch_progress=max(live, key=lambda p: (p['active'], p['updated']), default=None))


PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>A051 训练进度</title>
<style>body{font:16px system-ui;margin:32px;background:#f5f7fb;color:#172438}h1{font-size:28px}section{background:white;padding:22px;border-radius:12px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}progress{width:100%;height:24px}.muted{color:#536174}pre{white-space:pre-wrap}</style>
<h1>A051 正式训练进度</h1><p id="connection">正在读取…</p>
<section><h2 id="execution">正在检查训练进程…</h2><p id="cpu"></p>
<button id="stop" onclick="control('stop')">停止训练（未完成轮次重跑）</button>
<button id="start" onclick="control('start')">继续训练</button><p id="action" role="status"></p>
<p>关机前点击停止，等页面确认“训练已停止”再关机。当前仅保存完整 epoch；若第一轮尚未完成，继续时从该轮起点重新开始。重启电脑后双击项目中的“打开训练进度.cmd”，再点击继续。关闭网页不会停止训练。</p></section>
<section><h2 id="count"></h2><progress id="bar" max="60" value="0"></progress><p id="stage"></p>
<p class="muted">每 2 秒核实训练进程及读取记录。CPU 活动按进程树采样，可能超过100%；低活动可能是在等数据，不代表已停止。进程存活不能保证程序未卡住。阶段更新时间不是心跳，停止后可能仍显示上次阶段。</p></section>
<section><h2>当前 epoch 实时进度</h2><p id="epoch-label">等待批次进度…</p>
<progress id="epoch-bar" max="100" value="0"></progress><p id="epoch-detail" aria-live="polite"></p>
<p class="muted">百分比按实际完成批次计算，约每 2 秒更新；训练、验证分别从 0% 到 100%。验证完成并保存断点后，整轮才算完成。进度停留时请检查更新时间。旧训练进程需停止并继续一次才能开始记录批次进度，未完成轮次会重跑。</p></section>
<section><h2>模型与验证集效果</h2><table><thead><tr><th>模型 / 种子 / α</th><th>状态</th><th>已记录轮数</th><th>BER ↓</th><th>SNR dB ↑</th><th>NMR dB ↓</th><th>最近一轮耗时</th><th>指标时间</th></tr></thead><tbody id="rows"></tbody></table>
<p>BER 是误码率，例如 0.01 = 1%。SNR 越高表示波形误差越小；NMR 越低表示噪声相对掩蔽阈值越低。必须一起看，不能只看单项。不同 α 或不同损失的 loss 不宜直接比较。</p>
<p>以下是验证集指标，不是 RWC 最终测试结果。预热不计入 60 个候选完成数；早停使各模型轮数不同，完成比例不能直接换算剩余时间。</p></section>
<section><h2>逐轮记录与曲线</h2><label>主模型 <select id="model"></select></label>
<label>比较模型 <select id="compare"><option value="">不比较</option></select></label>
<p>蓝线为主模型，橙线为比较模型；横轴是完整训练路径的epoch编号，可能从预热之后开始。曲线为各轮验证指标，不能据此宣称最终模型优劣。最终选模使用最低验证loss轮，未必是最后一轮。不同α或损失的loss不宜横向比较。</p>
<div id="charts" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px"></div>
<pre id="best"></pre><pre id="history"></pre></section>
<script>
let data;const el=id=>document.getElementById(id);const fmt=x=>typeof x==='number'?x.toFixed(5):'—';
let busy=false;
async function control(action){busy=true;el('stop').disabled=true;el('start').disabled=true;el('action').textContent=action==='stop'?'正在停止，请等待确认…':'正在恢复…';try{const response=await fetch('/'+action,{method:'POST',headers:{'X-Control-Token':'__TOKEN__'}});const result=await response.json();el('action').textContent=result.message;}catch(e){el('action').textContent='操作未确认：'+e.message+'；请检查实时状态，暂不要关机。'}finally{busy=false;update();}}
function execution(){const state=data.execution;el('execution').textContent=state.running?'训练进程正在运行':'训练进程已停止';el('cpu').textContent=state.processes.map(p=>'PID '+p.pid+'；CPU 活动 '+(p.cpu_percent===null?'等待第二次采样':p.cpu_percent.toFixed(1)+'%')).join('；');el('stop').disabled=busy||!state.running;el('start').disabled=busy||state.running;}
function epochProgress(){const p=data.epoch_progress;const bar=el('epoch-bar');if(!p){bar.value=0;el('epoch-label').textContent='尚无批次记录';el('epoch-detail').textContent=data.execution.running?'当前进程可能仍使用旧代码；重新启动训练后开始显示实际百分比。':'启动训练后显示实际批次百分比。';return;}const phases={training:'训练',validation:'验证',saving:'保存断点',epoch_complete:'本轮已完成',complete:'该模型已完成'};bar.value=p.percent;el('epoch-label').textContent=p.name+' · epoch '+p.epoch+' / '+p.epochs+' · '+(phases[p.phase]||p.phase);const age=Math.max(0,(Date.now()/1000)-p.updated);el('epoch-detail').textContent=(p.active?'':'历史记录（对应训练进程已停止） · ')+p.completed.toLocaleString()+' / '+p.total.toLocaleString()+' 批次 · '+p.percent.toFixed(1)+'% · 本轮已用 '+Math.floor(p.elapsed_seconds/60)+'分'+Math.floor(p.elapsed_seconds%60)+'秒 · 更新于 '+new Date(p.updated*1000).toLocaleTimeString()+'（'+Math.floor(age)+'秒前）';}
function history(){let r=data.rows.find(r=>r.name===el('model').value);el('history').textContent=r&&r.history.length?r.history.map(h=>`epoch ${h.epoch+1} | BER ${fmt(h.ber)} | SNR ${fmt(h.snr)} dB | NMR ${fmt(h.nmr_db)} dB | train loss ${fmt(h.train_loss)} | val loss ${fmt(h.loss)}`).join('\\n'):'尚未完成一轮训练和验证。';}
el('model').onchange=history;
async function update(){try{let response=await fetch('/status',{cache:'no-store'});data=await response.json();el('connection').textContent='读取时间：'+data.now;el('count').textContent=`候选模型完成 ${data.completed} / ${data.total}`;el('bar').max=data.total||60;el('bar').value=data.completed;el('stage').textContent='流水线阶段：'+data.pipeline.stage+'；阶段更新时间：'+data.pipeline.updated;el('rows').replaceChildren();let selected=el('model').value;el('model').replaceChildren();for(let r of data.rows){let h=r.history.at(-1)||{};let tr=document.createElement('tr');for(let v of [r.name,r.complete?'已完成':(r.warmup?'预热，未完成':'未完成'),`${r.epochs} / ${r.limit}`,fmt(h.ber),fmt(h.snr),fmt(h.nmr_db),h.seconds?Math.round(h.seconds)+' 秒':'—',r.updated]){let td=document.createElement('td');td.textContent=v;tr.append(td)}el('rows').append(tr);let option=document.createElement('option');option.value=r.name;option.textContent=r.name;el('model').append(option)}if(data.rows.some(r=>r.name===selected))el('model').value=selected;history()}catch(e){el('connection').textContent='无法读取进度：'+e.message}}
const refresh=update;update=async function(){await refresh();if(data){execution();epochProgress();if(typeof renderTrainingCharts==='function')renderTrainingCharts();}};
update();setInterval(update,2000);
</script><script src="/charts.js"></script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        host = self.headers.get('Host', '')
        origin = self.headers.get('Origin')
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if host not in allowed or (origin and origin != 'http://'+host) or not secrets.compare_digest(self.headers.get('X-Control-Token', ''), TOKEN):
            self.send_error(403)
            return
        if self.path not in ('/start', '/stop'):
            self.send_error(404)
            return
        try:
            message = CONTROL.start() if self.path == '/start' else CONTROL.stop()
            code = 200
        except Exception as exc:
            message, code = str(exc), 500
        payload = json.dumps({'message': message}, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path not in ('/', '/status', '/charts.js'):
            self.send_error(404)
            return
        content = json.dumps(snapshot(), ensure_ascii=False) if self.path == '/status' else PAGE.replace('__TOKEN__', TOKEN)
        if self.path == '/charts.js':
            content = (ROOT/'scripts/progress_charts.js').read_text(encoding='utf-8')
        payload = content.encode('utf-8')
        self.send_response(200)
        kind = 'application/json' if self.path == '/status' else 'text/javascript' if self.path == '/charts.js' else 'text/html'
        self.send_header('Content-Type', kind+'; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7863)
    args = parser.parse_args()
    print(f'Progress page: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
