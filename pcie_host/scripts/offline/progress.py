"""Console progress and private command logs for standalone offline installation."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import threading
import time
import traceback

from log_progress import LogProgress, safe_text


class Progress(LogProgress):
    def __init__(self, console=None, interval=None, verbose=False):
        super().__init__(console, verbose=verbose, interval=interval)
        self.console = self.stream
        fd, path = tempfile.mkstemp(prefix='neat-core-offline-', suffix='.log')
        self.path = Path(path)
        self.log_path = self.path
        self.log = os.fdopen(fd, 'w', buffering=1)
        self.lock = threading.Lock()
        self.number = 0
        self.message('INFO', 'Detailed command output: '+str(self.path))

    def message(self, state, text):
        colors = {'RUN': '36', 'OK': '32', 'WAIT': '33', 'FAIL': '31', 'INFO': '36', 'WARN': '33'}
        prefix = '['+state+']'
        if self.color:
            prefix = '\033['+colors[state]+'m'+prefix+'\033[0m'
        text = safe_text(text)
        with self.lock:
            print(prefix+' '+text, file=self.console, flush=True)

    def show(self, message, status='info'):
        self.message('WAIT' if status == 'info' else {'ok': 'OK', 'error': 'FAIL'}[status], message)

    @contextmanager
    def step(self, label):
        self.number += 1
        label = str(self.number)+'. '+label
        start = time.monotonic()
        self.message('RUN', label)
        self.log.write('\n=== '+label+' ===\n')
        try:
            with self.activity(label, panel=True, announce=False):
                try:
                    yield
                except BaseException:
                    traceback.print_exc(file=self.log)
                    self.log.flush()
                    raise
        except BaseException:
            self.message('FAIL', label+f' ({int(time.monotonic()-start)}s)')
            raise
        else:
            self.message('OK', label+f' ({int(time.monotonic()-start)}s)')
        finally:
            self.log.flush()

    def close(self):
        self.log.close()
