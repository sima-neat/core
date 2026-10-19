"""Core checks the Scheduler-reported rendering capability."""
from pathlib import Path
import subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[2]

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
