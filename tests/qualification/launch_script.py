#!/usr/bin/env python3
"""Run a catalogue script unchanged on a chosen backend (README.md, tier T3).

  launch_script.py --backend gstreamer|dataflow [--record <file.json>] -- <script.py> [script args...]

The script file is not edited and no environment variable is involved: this launcher calls
pyneat.set_backend(...) first, as a customer's program would, and then runs the script with
runpy.run_path(script, run_name="__main__") and the script's own arguments (sys.argv as the script
expects it, its directory first on sys.path). Without the launcher a script runs on the default
backend. --record writes what the run did: backend, exit status, wall time, and the exception's
type and text if one escaped (`error` is the text alone, which qualify.py reads as the one
not-carried form), for qualify.py to put beside the other backend's run. The record is written
before the script starts too, so a run that dies without unwinding leaves its backend and command.
"""
import argparse
import json
import os
import runpy
import sys
import time
import traceback


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        sys.exit(__doc__)
    split = argv.index('--')
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=('gstreamer', 'dataflow'), required=True)
    ap.add_argument('--record')
    args = ap.parse_args(argv[:split])
    script, script_args = argv[split + 1], argv[split + 2:]

    import pyneat  # the installed package: the launcher adds nothing to it
    pyneat.set_backend(pyneat.RunBackend.GStreamer if args.backend == 'gstreamer'
                       else pyneat.RunBackend.Dataflow)

    record = {'backend': args.backend, 'script': os.path.abspath(script), 'args': script_args,
              'set_backend': str(pyneat.backend())}
    if args.record:
        with open(args.record, 'w') as f:
            json.dump(record, f, indent=1)
    sys.argv = [script] + script_args
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    t0 = time.monotonic()
    status = 0
    try:
        runpy.run_path(script, run_name='__main__')
    except SystemExit as e:
        status = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if status and not isinstance(e.code, int):  # sys.exit("text"): Python prints the text and exits 1
            print(e.code, file=sys.stderr)
            record['exit_message'] = str(e.code)
    except BaseException as e:  # recorded, and printed as Python prints an uncaught exception
        traceback.print_exc()
        record['exception'] = f'{type(e).__name__}: {e}'
        record['error'] = str(e)
        record['not_carried'] = 'is not carried by the Dataflow backend' in str(e)
        record['traceback'] = traceback.format_exc()[-4000:]
        status = 1
    record['exit_status'] = status
    record['seconds'] = round(time.monotonic() - t0, 3)
    if args.record:
        with open(args.record, 'w') as f:
            json.dump(record, f, indent=1)
    return status


if __name__ == '__main__':
    sys.exit(main())
