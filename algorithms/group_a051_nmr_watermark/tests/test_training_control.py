import shutil
import time
from pathlib import Path
from scripts.training_control import TrainingControl


def test_start_stop_resume_and_workspace_isolation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path/'.venv/Scripts').mkdir(parents=True)
    shutil.copyfile(root/'.venv/Scripts/python.exe', tmp_path/'.venv/Scripts/python.exe')
    shutil.copyfile(root/'.venv/pyvenv.cfg', tmp_path/'.venv/pyvenv.cfg')
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/run_full.py').write_text('import time\nfrom pathlib import Path\nPath("started").write_text("yes")\ntime.sleep(120)\n')
    controller = TrainingControl(tmp_path)
    other = TrainingControl(tmp_path/'unrelated')
    try:
        for _ in range(2):
            controller.start()
            launched = controller.launched.pid
            controller.start()
            assert controller.launched.pid == launched
            for attempt in range(100):
                if controller.status()['running']:
                    break
                time.sleep(.1)
            assert controller.status()['running']
            assert not other.status()['running']
            controller.stop()
            assert not controller.status()['running']
    finally:
        controller.stop()
