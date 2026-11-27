"""Exercise Core's reviewed label migration without changing GitHub metadata."""

import argparse
import importlib.util
import json
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('migrate_issue_labels', ROOT / 'tools/ci/migrate_issue_labels.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


@pytest.fixture
def setup_migration(monkeypatch, tmp_path):
    state = {
        'labels': {'bug', 'patch-candidate', 'other'},
        'definitions': {'triage', 'patch-candidate'},
        'comments': [], 'calls': [], 'remaining': set(),
    }
    monkeypatch.setenv('GITHUB_TOKEN', 'test-token')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'sima-neat/core')

    def paginated(path, token):
        if path.endswith('/comments'):
            return state['comments']
        if path.endswith('/labels'):
            return [{'name': name} for name in state['definitions']]
        return [{'number': 7, 'title': 'Example',
                 'labels': [{'name': name} for name in state['labels']]}]

    def request(method, path, token, payload=None):
        state['calls'].append((method, path))
        if method == 'GET':
            if '?state=all&labels=' in path:
                label = urllib.parse.unquote(path.split('labels=', 1)[1].split('&')[0])
                return [{'number': 8}] if label in state['remaining'] else []
            return {'labels': [{'name': name} for name in state['labels']]}
        if method == 'POST' and path.endswith('/comments'):
            state['comments'].append({'body': payload['body']})
        elif method == 'POST':
            state['labels'].update(payload['labels'])
        elif method == 'DELETE':
            label = urllib.parse.unquote(path.rsplit('/', 1)[1])
            target = state['labels'] if '/issues/' in path else state['definitions']
            target.discard(label)
        return {}

    monkeypatch.setattr(migration, 'api_paginated', paginated)
    monkeypatch.setattr(migration, 'api_request', request)
    args = argparse.Namespace(config=str(ROOT / '.github/issue-triage/config.json'),
                              assignments=None, dry_run='false')
    return state, args, tmp_path


def test_preserves_release_annotation_and_retries(setup_migration):
    state, args, _ = setup_migration
    migration.migrate_labels(args)
    assert state['labels'] == {'bug', 'other'}
    assert state['definitions'] == set()
    assert 'patch-candidate' in state['comments'][0]['body']
    migration.migrate_labels(args)
    assert len(state['comments']) == 1


def test_dry_run_has_no_mutations(setup_migration, capsys):
    state, args, _ = setup_migration
    args.dry_run = 'true'
    migration.migrate_labels(args)
    assert state['calls'] == []
    assert state['comments'] == []
    assert json.loads(capsys.readouterr().out)['issues'][0]['category'] == 'bug'


def test_ambiguous_categories_require_review_before_writing(setup_migration):
    state, args, _ = setup_migration
    state['labels'] = {'bug', 'feature', 'triage'}
    with pytest.raises(SystemExit, match='assignments'):
        migration.migrate_labels(args)
    assert state['calls'] == []


def test_reviewed_assignment_resolves_ambiguity(setup_migration):
    state, args, tmp_path = setup_migration
    state['labels'] = {'bug', 'feature', 'triage', 'other'}
    assignments = tmp_path / 'assignments.json'
    assignments.write_text(json.dumps({'7': 'feature'}))
    args.assignments = str(assignments)
    migration.migrate_labels(args)
    assert state['labels'] == {'feature', 'other'}


def test_retains_definition_when_label_is_still_used(setup_migration):
    state, args, _ = setup_migration
    state['remaining'] = {'triage'}
    with pytest.raises(SystemExit, match='still used'):
        migration.migrate_labels(args)
    assert 'triage' in state['definitions']
