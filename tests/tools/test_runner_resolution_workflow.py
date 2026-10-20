"""Contract between Core build artifacts and the shared runner resolver."""
from pathlib import Path
import unittest

import yaml


class RunnerResolutionWorkflowTest(unittest.TestCase):
    def test_runner_uses_consumed_internals_manifest_and_preserves_labels(self):
        root = Path(__file__).resolve().parents[2]
        jobs = yaml.safe_load((root / '.github/workflows/vulcan-ci.yml').read_text())['jobs']
        resolver = jobs['resolve-test-runner']
        self.assertIn('build', resolver['needs'])
        self.assertNotIn('resolve-test-runner', jobs['build']['needs'])
        self.assertEqual(resolver['with']['manifest_artifact'], jobs['build']['with']['artifact_name'])
        self.assertEqual(resolver['with']['manifest_path'], 'internals-manifest.json')
        self.assertTrue(resolver['with']['allow-version-fallback'])
        self.assertIn('"rtsp"', resolver['with']['runner-labels'])
        self.assertIn('resolve-vulcan-config.outputs.vulcan_env', resolver['with']['runner-labels'])
        self.assertEqual(resolver['uses'].split('@')[1], resolver['with']['shared_workflow_ref'])
        self.assertIn('outputs.runner_labels', jobs['test-devkit']['runs-on']['labels'])
        self.assertEqual(jobs['test-devkit']['steps'][0]['name'], 'Verify selected DevKit firmware')
