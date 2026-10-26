"""Console progress and private command logs for standalone offline installation."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import traceback


class Progress:
    def __init__(self, console=None, interval=15):
        self.console = console if console is not None else sys.stdout
        self.color = self.console.isatty() and 'NO_COLOR' not in os.environ and os.environ.get('TERM') != 'dumb'
        self.interval = interval
        fd, path = tempfile.mkstemp(prefix='neat-core-offline-', suffix='.log')
        self.path = Path(path)
        self.log = os.fdopen(fd, 'w', buffering=1)
        self.lock = threading.Lock()
        self.number = 0
        self.message('INFO', 'Detailed command output: '+str(self.path))

    def message(self, state, text):
        colors = {'RUN': '36', 'OK': '32', 'WAIT': '33', 'FAIL': '31', 'INFO': '36', 'WARN': '33'}
        prefix = '['+state+']'
        if self.color:
            prefix = '\033['+colors[state]+'m'+prefix+'\033[0m'
        text = ''.join(char if char.isprintable() else ' ' for char in str(text))
        with self.lock:
            print(prefix+' '+text, file=self.console, flush=True)

    @contextmanager
    def step(self, label):
        self.number += 1
        label = str(self.number)+'. '+label
        start = time.monotonic()
        stopped = threading.Event()
        def heartbeat():
            while not stopped.wait(self.interval):
                self.message('WAIT', label+f' — still working ({int(time.monotonic()-start)}s)')
        self.message('RUN', label)
        self.log.write('\n=== '+label+' ===\n')
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            yield
        except BaseException:
            traceback.print_exc(file=self.log)
            stopped.set()
            worker.join()
            self.message('FAIL', label+f' ({int(time.monotonic()-start)}s)')
            raise
        else:
            stopped.set()
            worker.join()
            self.message('OK', label+f' ({int(time.monotonic()-start)}s)')
        finally:
            stopped.set()
            worker.join()
            self.log.flush()

    def close(self):
        self.log.close()
