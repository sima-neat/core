"""Shared PCIe and standalone terminal progress and bounded log rendering."""
import codecs
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import unicodedata


def safe_text(value):
    """Remote diagnostics must not inject terminal control sequences."""
    return ''.join(c if c.isprintable() else ' ' for c in str(value))


class LogProgress:
    def __init__(self, stream=None, verbose=False, interval=None):
        self.stream = stream or sys.stdout
        self.log_path = None
        self.verbose = verbose
        self.interval = interval
        self.color = self.stream.isatty() and 'NO_COLOR' not in os.environ and os.environ.get('TERM') != 'dumb'

    def show(self, message, status='info'):
        codes = {'info': '36', 'ok': '32', 'error': '31'}
        text = safe_text(message)
        if self.color:
            text = '\033['+codes[status]+'m'+text+'\033[0m'
        print(text, file=self.stream, flush=True)

    @contextmanager
    def stream_log(self):
        """Follow the complete diagnostic log without changing subprocess handling."""
        if not self.verbose:
            yield
            return
        stopped = threading.Event()
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        with Path(self.log_path).open('rb') as source:
            def drain():
                while True:
                    chunk = source.read(65536)
                    if not chunk:
                        break
                    self.stream.write(decoder.decode(chunk))
                    self.stream.flush()

            def follow():
                while not stopped.wait(0.1):
                    drain()

            worker = threading.Thread(target=follow, daemon=True)
            worker.start()
            try:
                yield
            finally:
                stopped.set()
                worker.join()
                drain()
                self.stream.write(decoder.decode(b'', final=True))
                self.stream.flush()

    def tail(self, start, count=10):
        """Read bounded output from this phase, using a separate file descriptor."""
        if self.log_path is None:
            return []
        try:
            with Path(self.log_path).open('rb') as log:
                end = log.seek(0, os.SEEK_END)
                offset = max(start, end - 65536)
                log.seek(offset)
                data = log.read(65536)
            if offset > start:
                data = data.partition(b'\n')[2]
            lines = []
            for line in data.decode('utf-8', errors='replace').splitlines():
                line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line)
                # Expected APT index retries and merged-/usr ownership probes
                # remain in the full log; genuine failures retain their status.
                if re.match(r'^(?:Get:|Ign:|Err:\d+ file:.* \./ Packages|dpkg-query: no path found matching pattern)', line) or line.strip() == 'Method gave a blank filename':
                    continue
                text = safe_text(line).strip()
                if text:
                    lines.append(text)
            return lines[-count:]
        except OSError:
            return ['Detailed output is temporarily unavailable; see the log.']

    @staticmethod
    def clip(text, width):
        result = ''
        used = 0
        for char in safe_text(text):
            size = 0 if unicodedata.combining(char) else (2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1)
            if used + size > width:
                break
            result += char
            used += size
        return result

    @contextmanager
    def activity(self, message, *, panel=False, announce=True):
        """Redraw a bounded log panel without changing command execution."""
        if announce:
            self.show(message)
        if self.verbose:
            yield
            return
        terminal = self.stream.isatty() and os.environ.get('TERM') != 'dumb'
        interval = self.interval if self.interval is not None else (1.0 if terminal else 15.0)
        panel = terminal and self.log_path is not None and panel
        start = Path(self.log_path).stat().st_size if panel else 0
        started = time.monotonic()
        stopped = threading.Event()
        height = 0
        frame = 0

        def refresh():
            nonlocal height, frame
            elapsed = int(time.monotonic() - started)
            text = f"{safe_text(message)} {elapsed // 60:02d}:{elapsed % 60:02d} elapsed"
            if panel:
                try:
                    size = os.get_terminal_size(self.stream.fileno())
                except (OSError, ValueError):
                    size = shutil.get_terminal_size((80, 24))
                if size.columns <= 0 or size.lines <= 0:
                    size = os.terminal_size((80, 24))
                count = max(1, min(10, size.lines - 5))
                lines = self.tail(start, count)
                if not lines:
                    lines = ['Waiting for installation output...']
                title = f'{elapsed // 60:02d}:{elapsed % 60:02d} elapsed | {safe_text(message)}'
                rows = [title] + ['  '+line for line in lines]
                rows += [''] * (max(height, len(rows)) - len(rows))
                if height:
                    print(f'\033[{height}A', end='', file=self.stream)
                for row in rows:
                    print('\r\033[2K'+self.clip(row, max(1, size.columns - 1)), file=self.stream)
                height = len(rows)
                self.stream.flush()
            elif terminal:
                print('\r'+'|/-\\'[frame % 4]+' '+text, end='', file=self.stream, flush=True)
            else:
                self.show(text+'; still running')
            frame += 1

        def heartbeat():
            while not stopped.wait(interval):
                refresh()

        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            yield
        finally:
            stopped.set()
            worker.join()
            if panel:
                if height or Path(self.log_path).stat().st_size > start:
                    refresh()
            elif terminal:
                print(file=self.stream, flush=True)

    def run(self, message, operation, *args):
        with self.activity(message, panel=bool(args and args[-1] == 'install')):
            return operation(*args)


