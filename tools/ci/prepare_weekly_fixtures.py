#!/usr/bin/env python3
"""Check the weekly stream fixtures before the stream cases run.

The weekly stream job selects a capability profile. Each required stream must
be declared in the environment, or preparation fails instead of letting the
case skip.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def require_streams(env, profile='all'):
    if profile not in ('none', 'http', 'all'):
        raise ValueError('Unknown stream profile')
    if profile == 'none':
        return
    required = [('HTTP', 'MJPEG')]
    if profile == 'all':
        required += [('RTSP', codec) for codec in ('H264', 'H265', 'MJPEG')]
    missing = []
    for protocol, codec in required:
        prefix = f'SIMANEAT_TEST_{protocol}_{codec}'
        if not (env.get(prefix + '_URL') or env.get(prefix + '_URLS')):
            missing.append(prefix + '_URL(S)')
        if codec == 'MJPEG' and not env.get(prefix + '_FPS', '').isdigit():
            missing.append(prefix + '_FPS')
        elif codec == 'MJPEG' and int(env[prefix + '_FPS']) <= 0:
            missing.append(prefix + '_FPS')
    if missing:
        raise ValueError('Required stream fixtures missing: ' + ', '.join(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream-profile', choices=('none', 'http', 'all'), required=True)
    parser.add_argument('--stream-report', type=Path)
    args = parser.parse_args()
    try:
        require_streams(os.environ, args.stream_profile)
        gaps = {'none': ['HTTP MJPEG', 'RTSP H264/H265/MJPEG'], 'http': ['RTSP H264/H265/MJPEG'], 'all': []}[args.stream_profile]
        report = {'profile': args.stream_profile, 'scope': 'codec_stream registrations only',
                  'other_rtsp_workflows_qualified': False, 'excluded_not_qualified': gaps,
                  'prepared_not_executed': args.stream_profile != 'none'}
        if args.stream_report:
            args.stream_report.parent.mkdir(parents=True, exist_ok=True)
            with args.stream_report.open('x') as stream:
                json.dump(report, stream, indent=2)
        print(json.dumps(report))
    except (ValueError, OSError) as exc:
        print(f'PREPARATION_BLOCKED: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
