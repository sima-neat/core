"""Core selects Scheduler; the producer owns lower cohort validation."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('build_dependencies', ROOT/'tools/build_dependencies.py')
DEPENDENCIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPENDENCIES)


def source(root, component):
    # Shell-shadow fixture only; no lower implementation is imported.
    (root/'cmake').mkdir(parents=True)
    (root/'CMakeLists.txt').touch()
    (root/'cmake'/('Sima'+component.title()+'Config.cmake.in')).touch()
    (root/'build.sh').touch()


def shell_function(name):
    text=(ROOT/'build.sh').read_text(); start=text.index(f'{name}() {{')
    return text[start:text.index('\n}\n', start)+2]


class BuildDependenciesTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name); self.core=self.root/'core'; self.core.mkdir()

    def producer(self):
        root=self.root/'scheduler'; source(root,'scheduler'); (root/'tools').mkdir()
        (root/'tools/client_dependencies.py').write_text('import sys\nprint("delegated", " ".join(sys.argv[1:]))\nsys.exit(23)\n')
        return root

    def package(self):
        root=self.root/'package'; prefix=root/'usr'
        tool=prefix/'share/SimaScheduler/tools/client_dependencies.py';tool.parent.mkdir(parents=True)
        tool.write_text('print("metadata fixture")\n')
        entry={'kind':'file','sha256':hashlib.sha256(tool.read_bytes()).hexdigest(),'mode':tool.stat().st_mode&0o7777}
        info={'component':'scheduler','installed_files':{tool.relative_to(prefix).as_posix():entry}}
        info['package_sha256']=hashlib.sha256(json.dumps(info,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        (prefix/'share/SimaScheduler/buildinfo.json').write_text(json.dumps(info))
        return root,tool

    def test_selects_source_before_stale_package(self):
        root=self.producer(); (root/'dist/package/usr').mkdir(parents=True)
        self.assertEqual(DEPENDENCIES.scheduler_coordinator(self.core,{}),root/'tools/client_dependencies.py')

    def test_explicit_package_and_receipt_are_authoritative(self):
        self.producer(); root,tool=self.package()
        self.assertEqual(DEPENDENCIES.scheduler_coordinator(self.core,{'NEAT_SCHEDULER_PACKAGE_DIR':str(root)}),tool)
        tool.write_text('different producer')
        with self.assertRaisesRegex(ValueError,'receipt'):
            DEPENDENCIES.scheduler_coordinator(self.core,{'NEAT_SCHEDULER_PACKAGE_DIR':str(root)})

    def test_selected_missing_coordinator_never_falls_back(self):
        self.producer(); broken=self.root/'broken'; source(broken,'scheduler')
        with self.assertRaisesRegex(ValueError,'lacks its client'):
            DEPENDENCIES.scheduler_coordinator(self.core,{'NEAT_SCHEDULER_SOURCE':str(broken)})
        with self.assertRaisesRegex(ValueError,'only one'):
            DEPENDENCIES.scheduler_coordinator(self.core,{'NEAT_SCHEDULER_SOURCE':str(broken),'NEAT_SCHEDULER_PACKAGE_DIR':'unused'})

    def test_missing_scheduler_does_not_fetch(self):
        with self.assertRaisesRegex(ValueError,'TODO: remote artifact'):
            DEPENDENCIES.scheduler_coordinator(self.core,{})

    def test_actual_cli_preserves_arguments_and_failure(self):
        producer=self.producer()
        env=dict(os.environ);env.pop('NEAT_SCHEDULER_PACKAGE_DIR',None);env['NEAT_SCHEDULER_SOURCE']=str(producer)
        args=['prepare','--core',str(self.core),'--output',str(self.root/'out'),'--jobs','3','--clean']
        result=subprocess.run([sys.executable,str(ROOT/'tools/build_dependencies.py'),*args],env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,23,result.stderr)
        self.assertEqual(result.stdout.strip(),'delegated '+' '.join(args))
    def test_shadow_copies_only_selected_sources_and_rewrites_paths(self):
        source(self.root / "external runner", "runner")
        (self.core / "build.sh").touch()
        (self.root / "external runner/build").mkdir()
        (self.root / "external runner/build/stale").touch()
        shadow = self.root / "shadow"
        script = "set -euo pipefail\n" + shell_function("copy_shadow_source") + "\n" + shell_function("prepare_shadow_workspace")
        script += f'\nREPO_ROOT={shlex.quote(str(self.core))}\nNEAT_RUNNER_SOURCE={shlex.quote(str(self.root / "external runner"))}\n'
        script += f'prepare_shadow_workspace {shlex.quote(str(shadow))}\nprintf "%s" "$NEAT_RUNNER_SOURCE"\n'
        result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, str(shadow / "runner"))
        self.assertTrue((shadow / "runner/CMakeLists.txt").is_file())
        self.assertFalse((shadow / "runner/build").exists())
        self.assertFalse((shadow / "internals").exists())

    def test_main_keeps_one_final_install_and_deploy(self):
        main = shell_function("main")
        self.assertEqual(main.count("install_artifacts_into_current_environment_if_requested"), 1)
        self.assertEqual(main.count("deploy_artifacts_to_devkit_if_requested"), 1)
        self.assertNotIn("ensure_neat_internals", main)
        self.assertLess(main.index("build_dependencies.py prepare"), main.index("configure_cmake"))


if __name__ == "__main__":
    unittest.main()
