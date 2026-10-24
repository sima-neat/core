#!/usr/bin/env python3
"""Admit only completed, successful upstream publications from allowed refs."""
import argparse
import json
import os
import re
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError

REPOSITORY = 'sima-neat/core'


def api(path):
    request = Request('https://api.github.com/repos/'+REPOSITORY+'/'+path,
                      headers={'Authorization': 'Bearer '+os.environ['GH_TOKEN'],
                               'Accept': 'application/vnd.github+json'})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def eligible(run, jobs, tag_sha=None, validation_ref=None):
    if (run.get('repository', {}).get('full_name') != REPOSITORY
            or run.get('head_repository', {}).get('full_name') != REPOSITORY
            or run.get('path') != '.github/workflows/vulcan-ci.yml'
            or run.get('event') != 'push'):
        return None
    complete = run.get('status') == 'completed' and run.get('conclusion') == 'success'
    validating_published = (validation_ref and run.get('head_branch') == validation_ref
                            and run.get('status') in ('queued', 'in_progress', 'waiting', 'pending'))
    if not complete and not validating_published:
        return None
    sha, ref = run.get('head_sha', ''), run.get('head_branch', '')
    if not re.fullmatch(r'[0-9a-f]{40}', sha) or not ref or any(c.isspace() for c in ref):
        return None
    branch = ref in ('develop', 'main') or ref.startswith('release-')
    if not branch and tag_sha != sha and ref != validation_ref:
        return None
    # A successful docs-only run has no new native publication to assemble.
    required = ('Publish PCIe Host Package Metadata', 'Publish to Vulcan / Publish Vulcan artifacts')
    if any(not any(j['name'] == name and j.get('conclusion') == 'success' for j in jobs) for name in required):
        return None
    return {'sha': sha, 'ref': ref, 'run_id': run['id']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True, type=int)
    parser.add_argument('--validation-ref', help='allow this same-repository branch for non-publishing validation')
    args = parser.parse_args()
    run = api(f'actions/runs/{args.run_id}')
    jobs = []
    page = 1
    while True:
        rows = api(f'actions/runs/{args.run_id}/attempts/{run["run_attempt"]}/jobs?per_page=100&page={page}')['jobs']
        jobs.extend(rows)
        if len(rows) < 100:
            break
        page += 1
    tag_sha = None
    ref = run.get('head_branch') or ''
    if ref not in ('develop', 'main') and not ref.startswith('release-'):
        try:
            obj = api('git/ref/tags/'+quote(ref, safe=''))['object']
            for _ in range(8):
                if obj['type'] == 'commit':
                    tag_sha = obj['sha']
                    break
                if obj['type'] != 'tag':
                    break
                obj = api('git/tags/'+obj['sha'])['object']
        except HTTPError as exc:
            if exc.code != 404:
                raise
    source = eligible(run, jobs, tag_sha, args.validation_ref)
    if args.validation_ref and source is None:
        raise ValueError('Validation requires successful host/card publications from an eligible same-repository run')
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write('enabled='+str(source is not None).lower()+'\n')
        if source:
            for key, value in source.items():
                output.write(f'{key}={value}\n')
    print(json.dumps(source or {'skipped': 'Requires successful standard CI publications on develop/main/release-* or a matching tag.'}))


if __name__ == '__main__':
    main()
