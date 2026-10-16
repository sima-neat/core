"""Core's external renderer handoff contracts; host fixtures, not renderer execution."""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('core_build_dependencies', ROOT/'tools/build_dependencies.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class BoxrenderDependencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.provider=self.root/'provider';self.provider.mkdir()
        self.prefix=self.root/'scheduler/usr'
        tool=self.prefix/'share/SimaScheduler/tools/boxrender_package.py';tool.parent.mkdir(parents=True);tool.touch()
        self.info={'schema_version':1,'packages':{},'installed_files':{}}
        for role,name in [('runtime','a65apps'),('development','a65apps-dev')]:
            archive=self.provider/(name+'.deb');archive.write_bytes(b'HOST_TRANSIT_FIXTURE_NOT_A_REAL_DEB')
            self.info['packages'][role]={'Package':name,'Version':'2.2.0+test','Architecture':'arm64','archive':archive.name,
                'deb_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
        self.scheduler={'external_dependencies':{'BoxRender':copy.deepcopy(self.info)},
                        'debian_dependencies':[copy.deepcopy(self.info['packages']['runtime'])],
                        'build_configuration':{'SIMA_SCHEDULER_REQUIRE_RENDER_INT32':'ON'}}
        self.env=patch.dict(os.environ,{'NEAT_BOXRENDER_PACKAGE_DIR':str(self.provider)});self.env.start();self.addCleanup(self.env.stop)
        self.cli=patch.object(m.subprocess,'check_output',return_value=json.dumps(self.info));self.call=self.cli.start();self.addCleanup(self.cli.stop)

    def test_uses_installed_verifier_and_composes_separate_pair(self):
        out=self.root/'debs';out.mkdir()
        result=m.compose_external_boxrender(self.scheduler,self.prefix,out)
        self.assertEqual(result['buildinfo'],self.info)
        self.assertEqual({p.name for p in out.iterdir()},{'a65apps.deb','a65apps-dev.deb'})
        command=self.call.call_args.args[0]
        self.assertEqual(command[1],str(self.prefix/'share/SimaScheduler/tools/boxrender_package.py'))
        self.assertIn(str(self.provider),command)

    def test_required_capability_without_consumed_provider_rejected(self):
        del self.scheduler['external_dependencies']['BoxRender']
        with self.assertRaisesRegex(ValueError,'lacks its consumed'):m.compose_external_boxrender(self.scheduler,self.prefix,None)

    def test_float_only_nonproduction_profile_does_not_require_provider(self):
        self.assertIsNone(m.compose_external_boxrender({},self.prefix,None))
        self.call.assert_not_called()

    def test_wrong_provider_identity_rejected(self):
        self.call.return_value=json.dumps({**self.info,'changed':True})
        with self.assertRaisesRegex(ValueError,'differs from'):m.compose_external_boxrender(self.scheduler,self.prefix,None)

    def test_unpinned_runtime_dependency_rejected(self):
        self.scheduler['debian_dependencies'][0]['Version']='old'
        with self.assertRaisesRegex(ValueError,'exact selected'):m.compose_external_boxrender(self.scheduler,self.prefix,None)

    def test_changed_archive_after_verifier_rejected(self):
        out=self.root/'debs';out.mkdir();(self.provider/'a65apps.deb').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed during'):m.compose_external_boxrender(self.scheduler,self.prefix,out)

    def test_lower_package_overlap_rejected(self):
        (self.prefix/'include').mkdir();(self.prefix/'include/shared.h').touch()
        packages={'scheduler':{'prefix':str(self.prefix)},'external_dependencies':{'BoxRender':{'buildinfo':{
            'installed_files':{'usr/include/shared.h':{}}}}}}
        with self.assertRaisesRegex(ValueError,'duplicates'):m.verify_package_ownership(packages)

    def test_source_wrapper_forwards_explicit_provider(self):
        with patch.object(m.subprocess,'run') as run:
            m.build_runtime_source('scheduler',self.root/'src',self.root/'package',self.root/'output',
                {'kernels':{'package':'kernels','buildinfo':{}},'runner':{'package':'runner'}},2,False)
        command=run.call_args.args[0]
        self.assertEqual(command[command.index('--boxrender-package')+1],str(self.provider))


class CoreRenderCapabilityTests(unittest.TestCase):
    def configure(self, definitions, expected):
        with tempfile.TemporaryDirectory() as temporary:
            script=Path(temporary)/'check.cmake'
            script.write_text(definitions + '\ninclude("' + str(ROOT/'cmake/SchedulerRenderCapability.cmake') + '")\n')
            result=subprocess.run(['cmake','-P',str(script)],capture_output=True,text=True)
            self.assertEqual(result.returncode,expected,result.stdout+result.stderr)
            if expected:self.assertIn('typed int32 renderer',' '.join(result.stderr.split()))

    def test_production_profile_accepts_capable_scheduler(self):
        self.configure('set(SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS ON)\nset(SimaScheduler_RENDER_ENABLED ON)\nset(SimaScheduler_RENDER_INT32_CAPABLE ON)',0)

    def test_production_profile_rejects_missing_capability(self):
        self.configure('set(SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS ON)',1)

    def test_nonrendering_profile_remains_supported(self):
        self.configure('set(SIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS OFF)',0)


if __name__=='__main__':unittest.main()
