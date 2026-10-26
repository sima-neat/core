"""Bounded Linux process supervision shared by offline installers."""
import ctypes
import os
import signal
import subprocess
import time


def run_host(args, *, cleanup_grace=30, **kwargs):
    """Own and reap the host process tree before the caller can release its lock."""
    # Linux subreaper adoption lets us wait for grandchildren even if a shell or
    # timeout exits first. No deployment child may create a separate session.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    get_subreaper, set_subreaper = 37, 36  # PR_GET/SET_CHILD_SUBREAPER
    if libc.prctl(get_subreaper, ctypes.byref(previous), 0, 0, 0) or libc.prctl(set_subreaper, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'Cannot supervise host deployment children')
    handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    process = None
    try:
        # Block Ctrl-C across creation so cleanup always has the process-group ID.
        process = subprocess.Popen(args, start_new_session=True, **kwargs)
        signal.signal(signal.SIGINT, handler)
        code = process.wait(timeout=1800)
        if code:
            raise subprocess.CalledProcessError(code, args)
    finally:
        # A second Ctrl-C must not escape cleanup and release the installer lock.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if process is not None:
                def send(sig):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass

                def reaped():
                    process.poll()
                    while True:
                        try:
                            pid, _ = os.waitpid(-process.pid, os.WNOHANG)
                        except ChildProcessError:
                            return True
                        if pid == 0:
                            return False

                send(signal.SIGTERM)
                # Also release stopped children so they can handle termination.
                send(signal.SIGCONT)
                deadline = time.monotonic() + cleanup_grace
                finished = reaped()
                while not finished and time.monotonic() < deadline:
                    time.sleep(0.05)
                    finished = reaped()
                if not finished:
                    send(signal.SIGKILL)
                process.wait()
                # Reap adopted descendants as well as the direct child. Retain
                # the lock while the kernel completes their termination.
                while True:
                    try:
                        os.waitpid(-process.pid, 0)
                    except ChildProcessError:
                        break
        finally:
            libc.prctl(set_subreaper, previous.value, 0, 0, 0)
            signal.signal(signal.SIGINT, handler)

