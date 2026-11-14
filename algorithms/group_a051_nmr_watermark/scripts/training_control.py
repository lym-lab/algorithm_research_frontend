"""Control only this workspace's full pipeline, retaining epoch checkpoints."""
import os
import subprocess
import threading
import time
from pathlib import Path
import psutil


class TrainingControl:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        self.samples = {}
        self.launched = None

    def processes(self):
        matches = []
        for process in psutil.process_iter():
            try:
                command = process.cmdline()
                if any(command[i:i+2] == ['-m', 'scripts.run_full'] for i in range(len(command)-1)) and Path(process.cwd()).resolve() == self.root:
                    matches.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        # Windows venv launcher and its Python child have the same command line.
        ids = {p.pid for p in matches}
        roots = []
        for process in matches:
            try:
                if process.ppid() not in ids:
                    roots.append(process)
            except psutil.NoSuchProcess:
                pass
        return roots

    def status(self):
        with self.lock:
            processes = self.processes()
            details = []
            now = time.monotonic()
            for parent in processes:
                try:
                    tree = [parent, *parent.children(recursive=True)]
                    cpu = sum(p.cpu_times().user+p.cpu_times().system for p in tree)
                    key = (parent.pid, parent.create_time())
                    previous = self.samples.get(key)
                    percent = max(0., 100*(cpu-previous[1])/(now-previous[0])) if previous and now > previous[0] else None
                    self.samples[key] = (now, cpu)
                    details.append(dict(pid=parent.pid, pids=[p.pid for p in tree], cpu_percent=percent, started=parent.create_time()))
                except psutil.NoSuchProcess:
                    continue
            return dict(running=bool(details), processes=details,
                        note='进程存活不等于每时刻都在计算；CPU为进程树采样，不能证明模型收敛。')

    def start(self):
        with self.lock:
            if self.processes() or (self.launched and self.launched.poll() is None):
                return '训练已在运行，没有重复启动。'
            python = self.root/'.venv/Scripts/python.exe'
            if not python.exists():
                raise RuntimeError('未找到项目 Python 环境')
            output = self.root/'outputs'
            output.mkdir(exist_ok=True)
            stamp = time.strftime('%Y%m%d-%H%M%S')
            env = dict(os.environ, PYTHONIOENCODING='utf-8')
            with (output/f'pipeline-{stamp}.log').open('ab') as stdout, (output/f'pipeline-{stamp}.errors.log').open('ab') as stderr:
                self.launched = subprocess.Popen([str(python), '-u', '-m', 'scripts.run_full'], cwd=self.root,
                                 env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            return '已启动恢复流程；从最近完整 epoch 继续，未完成的一轮重跑。'

    def stop(self):
        with self.lock:
            parents = self.processes()
            if not parents:
                return '训练已停止。'
            targets = []
            for parent in parents:
                try:
                    targets.extend(parent.children(recursive=True))
                    targets.append(parent)
                except psutil.NoSuchProcess:
                    pass
            for process in reversed(targets):
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(targets, timeout=10)
            if alive:
                raise RuntimeError('部分训练进程尚未退出，请等待后重试，暂不要关机。')
            return '训练已停止，可以关机。最近完整 epoch 的断点保留，当前未完成的一轮将重跑。'
