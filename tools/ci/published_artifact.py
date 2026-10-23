#!/usr/bin/env python3
"""Transport one immutable sima-cli publication; provider tooling owns its contents.

No package installation, repository fallback, lower-dependency resolution or
execution of downloaded code occurs until every publisher resource is verified.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from urllib.parse import quote, urljoin, urlparse
from urllib.request import urlopen

ENVIRONMENT_HOSTS = {'dev': 'artifacts.neat.paconsultings.com',
                     'staging': 'artifacts.stg.neat.sima.ai',
                     'production': 'artifacts.neat.sima.ai'}
HOSTS = set(ENVIRONMENT_HOSTS.values())


class PublicationResolutionError(subprocess.CalledProcessError):
    """The CLI could not resolve a selector; no publication was downloaded."""


def valid_branch(branch):
    """Validate a literal Git branch without Git or checkout-history expansion."""
    if (not isinstance(branch, str) or not branch or branch == 'HEAD'
            or branch.startswith('-') or branch.endswith('.')
            or '..' in branch or '@{' in branch
            or any(ord(char) <= 32 or ord(char) == 127 or char in '~^:?*[\\' for char in branch)):
        return False
    return all(part and not part.startswith('.') and not part.endswith('.lock')
               for part in branch.split('/'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def basename(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._+~\-]*', value):
        raise ValueError('Unsafe published resource name')
    return value


def download(url, output, *, sha256=None, size=None):
    def allowed(value):
        parts = urlparse(value)
        return (parts.scheme == 'https' and parts.hostname in HOSTS
                and not parts.username and not parts.password)
    if not allowed(url):
        raise ValueError('Publication URL is not an approved HTTPS origin')
    with urlopen(url, timeout=120) as source:
        if not allowed(source.url) or urlparse(source.url).hostname != urlparse(url).hostname:
            raise ValueError('Publication redirect left approved HTTPS origin')
        with Path(output).open('xb') as target:
            count = 0
            while block := source.read(1024 * 1024):
                count += len(block)
                if count > (size if size is not None else 1024**3):
                    raise ValueError('Publication exceeds declared size')
                target.write(block)
    if ((size is not None and Path(output).stat().st_size != size)
            or (sha256 is not None and digest(output) != sha256)):
        raise ValueError('Publication resource size/hash differs')


def acquire(cli, component, branch, spec, environment, output, *, package_path="", resource_pattern=None):
    """Resolve once, download by immutable ID, independently verify publisher bytes."""
    environment = {'prd': 'production', 'prod': 'production', 'stg': 'staging'}.get(environment, environment)
    if (not re.fullmatch(r'[a-z][a-z0-9-]*', component)
            or environment not in ENVIRONMENT_HOSTS
            or not valid_branch(branch)
            or not (spec == 'latest' or re.fullmatch(r'[0-9a-f]{12,40}', spec))):
        raise ValueError('Invalid publication selector')
    if package_path and not re.fullmatch(r'[a-z0-9][a-z0-9-]*(/[a-z0-9][a-z0-9-]*)*', package_path):
        raise ValueError('Unsafe publication package path')
    repository = component + ('/' + package_path if package_path else '')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    # Vulcan stores immutable publications under twelve-character commit IDs.
    # Shorten only the lookup key; retain the caller's complete selector below
    # when checking the publisher's full commit, including collision tails.
    lookup_spec = spec if spec == 'latest' else spec[:12]
    # Keep the selected CLI stable across resolution and download. Its updater
    # must not replace that tool or prompt in an unattended acquisition child.
    cli_env = {**os.environ, 'SIMA_CLI_CHECK_FOR_UPDATE': '0'}
    try:
        text = subprocess.check_output([str(cli), 'neat', 'install', '--env', environment,
                                       f'{repository}@{branch}:{lookup_spec}', '--json'], text=True, env=cli_env)
    except subprocess.CalledProcessError as error:
        raise PublicationResolutionError(error.returncode, error.cmd, error.output, error.stderr) from error
    selected = json.loads(text[text.index('{'):])
    revision = selected.get('resolved_spec', '')
    if (selected.get('repository') != component or selected.get('package_path', '') != package_path
            or selected.get('ref') != branch
            or selected.get('environment') != environment
            or not re.fullmatch(r'[0-9a-f]{12,40}', revision)
            or (spec != 'latest' and not (revision.startswith(spec) or spec.startswith(revision)))):
        raise ValueError('sima-cli resolved a different publication')
    url = selected['metadata_url']
    if urlparse(url).hostname != ENVIRONMENT_HOSTS[environment]:
        raise ValueError('Publication URL differs from selected environment')
    output.mkdir(parents=True, exist_ok=False)
    download(urljoin(url, 'manifest.json'), output / 'publication.json')
    publication = json.loads((output / 'publication.json').read_text())
    commit = publication.get('commit', '')
    if (publication.get('repository') != 'sima-neat/' + component
            or publication.get('repository_key') != component
            or publication.get('artifact_folder', '') != package_path
            or publication.get('branch') != branch
            or not re.fullmatch(r'[0-9a-f]{40}', commit) or not commit.startswith(revision)
            or (spec != 'latest' and not commit.startswith(spec))):
        raise ValueError('Publisher identity differs from selected publication')
    entries = {}
    for entry in publication['artifacts']:
        name = entry['path']
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or '..' in path.parts or str(path) != name
                or '\\' in name):
            raise ValueError('Unsafe publisher artifact path')
        if (name in entries or name in ('publication.json', 'acquisition.json')
                or not re.fullmatch(r'[0-9a-f]{64}', entry.get('sha256', ''))
                or type(entry.get('size')) is not int or entry['size'] < 0):
            raise ValueError('Invalid/duplicate publication resource')
        entries[name] = entry
    record = entries['metadata.json']
    download(url, output / 'metadata.json', sha256=record['sha256'], size=record['size'])
    metadata = json.loads((output / 'metadata.json').read_text())
    resources = metadata.get('resources', [])
    if (not isinstance(resources, list) or not resources or len(resources) != len(set(resources))
            or any(name not in entries for name in resources)
            or metadata.get('resources-checksum') != {name: entries[name]['sha256'] for name in resources}):
        raise ValueError('CLI resource inventory differs from publisher')
    # Validate names before asking the CLI to write any selected resources.
    # Nested publisher-only records are allowed, but executable downloads are flat.
    for name in resources:
        basename(name)
    if resource_pattern is None:
        wanted = set(resources) | ({'candidate.json', 'SHA256SUMS'} & set(entries))
        subprocess.run([str(cli), 'neat', 'download', f'{repository}@{branch}:{revision}',
                        '--env', environment, '-d', str(output / 'download')], check=True, stdout=sys.stderr,
                       env=cli_env)
    else:
        # LLiMa publishes build-tree artifacts alongside its root resources. Use
        # its exact declared dev resource, not a made-up metadata-dev variant or
        # a download of the entire publisher tree. Transport still uses sima-cli.
        wanted = {name for name in resources if re.fullmatch(resource_pattern, name)}
        if len(wanted) != 1:
            raise ValueError('Expected exactly one declared development resource')
        for name in wanted:
            basename(name)
            subprocess.run([str(cli), 'download', urljoin(url, quote(name, safe='')), '-d', str(output / 'download')],
                           check=True, stdout=sys.stderr, env=cli_env)
    for name in sorted(wanted):
        basename(name)
        entry = entries[name]
        if name in resources:
            # Generic sima-cli download preserves the URL-encoded basename.
            downloaded_name = quote(name, safe='') if resource_pattern is not None else name
            path = output / 'download' / downloaded_name
            if (path.is_symlink() or not path.is_file() or digest(path) != entry['sha256']
                    or path.stat().st_size != entry['size']):
                raise ValueError('sima-cli download differs from publisher: ' + name)
            shutil.copy2(path, output / name)
        else:
            download(urljoin(url, quote(name, safe='')), output / name, sha256=entry['sha256'], size=entry['size'])
    receipt = {'schema_version': 1, 'component': component, 'environment': environment,
               'branch': branch, 'package_path': package_path, 'commit': commit, 'selector': f'{repository}@{branch}:{revision}',
               'publication_sha256': digest(output / 'publication.json'),
               'resources': {name: entries[name]['sha256'] for name in sorted(wanted | {'metadata.json'})}}
    (output / 'acquisition.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt
