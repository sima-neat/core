#!/usr/bin/env python3
"""Harmless native host processes only; never opens device/driver interfaces."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'tools/hardware_test_supervisor.py'
spec = importlib.util.spec_from_file_location('hardware_test_supervisor', MODULE)
supervisor = importlib.util.module_from_spec(spec); spec.loader.exec_module(supervisor)
PYTHON = str(Path('/usr/bin/python3').resolve())

CHECKER = '''#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--phase');p.add_argument('--owner-json');a=p.parse_args()
s=json.loads(Path(__file__).with_name('check-state.json').read_text())
with Path(__file__).with_name('checker-calls.log').open('a') as f:f.write(a.phase+'\\n')
if s.get('sleep_phase')==a.phase:time.sleep(s.get('sleep_seconds',.1))
owner=json.loads(Path(a.owner_json).read_text())
r={'schema':1,'status':'ready','identity':s['identity'],'checked_at_epoch':time.time()+s.get('age_delta',0)}
if s.get('reject_phase')==a.phase:r['status']='refused'
if a.phase=='retired':r['retirement']={'proven':s.get('retirement',True),'kind':s.get('kind','authoritative-driver-retirement'),'token':'HOST_FIXTURE_ONLY','scope':'whole-owner-process-tree-and-dma','owner':owner}
print(json.dumps(r))
'''


class Lane:
    def __init__(self, deadline=.8, **kwargs):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.state = self.root / 'state'; self.state.mkdir(mode=0o700)
        self.checker = self.root / 'checker.py'; self.checker.write_text(CHECKER); self.checker.chmod(0o755)
        self.identity = dict(admission_id='HOST_FIXTURE_ONLY', boot_id=supervisor.process_identity(os.getpid())['boot_id'],
                             cohort_sha256='fixture', lease_pid=os.getpid(),
                             lease_start_ticks=supervisor.process_identity(os.getpid())['start_ticks'], observer_id='host_fixture')
        self.check = {'identity': self.identity}; self.write_check()
        self.socket = str(self.state / 'supervisor.sock')
        self.config = dict(identity=self.identity, state_dir=str(self.state), socket=self.socket,
                           supervisor_sha256=supervisor.sha(MODULE),
                           checker=dict(command=[str(self.checker)], sha256=supervisor.sha(self.checker), timeout_sec=.5),
                           executables={PYTHON: dict(sha256=supervisor.sha(PYTHON), deadline_sec=deadline, arguments=['-c', "exec(__import__('os').environ['HOST_FIXTURE_CODE'])"])},
                           poll_interval_sec=.005, client_heartbeat_max_age_sec=.5)
        completion_relative = kwargs.pop('completion_relative', None)
        self.config.update(kwargs)
        if completion_relative is not None:
            self.config['executables'][PYTHON]['completion'] = dict(output_dir=str(self.root / completion_relative), expected_files=1)
        self.config_file = self.root / 'config.json'; supervisor.atomic_json(self.config_file, self.config)
        self.log = (self.root / 'daemon.log').open('wb')
        self.daemon = subprocess.Popen([PYTHON, str(MODULE), 'serve', '--config', str(self.config_file)], stdout=self.log, stderr=self.log)
        end = time.monotonic()+3
        while not Path(self.socket).exists():
            if self.daemon.poll() is not None or time.monotonic()>end:
                raise AssertionError((self.root/'daemon.log').read_text())
            time.sleep(.01)

    def write_check(self):
        supervisor.atomic_json(self.root / 'check-state.json', self.check)

    def request(self, code, deadline=.8, **kwargs):
        return dict(command=[PYTHON, '-c', "exec(__import__('os').environ['HOST_FIXTURE_CODE'])"], cwd=str(self.root), env=dict(os.environ, HOST_FIXTURE_CODE=code), deadline_sec=deadline, **kwargs)

    def run(self, code="print('ok')", deadline=.8, **kwargs):
        return supervisor.run_owned(**self.request(code, deadline, **kwargs), socket_path=self.socket)

    def connect(self, request):
        c=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);c.connect(self.socket)
        c.sendall(json.dumps(request).encode()+b'\n');return c

    def close(self):
        # All fixture children are bounded ordinary Python sleeps. Wait for
        # natural exit; cleanup never signals a child or assumes retirement.
        end=time.monotonic()+4
        while time.monotonic()<end:
            try:
                c=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);c.settimeout(.5);c.connect(self.socket)
                c.sendall(b'{"action":"shutdown"}\n');response=json.loads(c.recv(65536));c.close()
                if response.get('status')=='shutdown':break
            except (OSError,ValueError):pass
            time.sleep(.03)
        self.daemon.wait(timeout=3);self.log.close();self.tmp.cleanup()


@unittest.skipUnless(os.geteuid()==0, 'Persistent board supervisor requires root; these fixtures contain no hardware')
class SupervisorTests(unittest.TestCase):
    def setUp(self):self.lanes=[]
    def lane(self, **kw):
        lane=Lane(**kw);self.lanes.append(lane);return lane
    def tearDown(self):
        for lane in self.lanes:
            lane.check.pop('reject_phase',None);lane.check.pop('sleep_phase',None);lane.check.pop('age_delta',None)
            lane.check['identity']=lane.identity;lane.check['retirement']=True;lane.check['kind']='authoritative-driver-retirement';lane.write_check();lane.close()

    def test_normal_completion_and_output(self):
        r=self.lane().run("import sys;print('out');print('err',file=sys.stderr);sys.exit(3)")
        self.assertEqual((r['status'],r['returncode'],r['stdout'],r['stderr']),('completed',3,'out\n','err\n'))
        self.assertTrue(r['retirement_proven']);self.assertIn('start_ticks',r['owner'])

    def test_timeout_keeps_owner_alive_and_refuses_next(self):
        lane=self.lane(deadline=.04)
        r=lane.run("import time;print('before',flush=True);time.sleep(.5);print('late',flush=True)",.04)
        self.assertEqual(r['status'],'timeout');self.assertFalse(r['retirement_proven'])
        self.assertEqual(supervisor.process_identity(r['owner']['pid'])['start_ticks'],r['owner']['start_ticks'])
        rejected=lane.run("open('must-not-launch','w').write('bad')",.04)
        self.assertEqual(rejected['status'],'refused');self.assertFalse((lane.root/'must-not-launch').exists())
        time.sleep(.6);self.assertIn('late',Path(r['stdout_path']).read_text())
        self.assertEqual(json.loads((lane.state/r['run_id']/'result.json').read_text())['status'],'timeout')

    def test_disconnect_retains_real_owner(self):
        lane=self.lane();c=lane.connect(lane.request("import time;open('started','w').write('1');time.sleep(.5)"))
        while not (lane.root/'started').exists():time.sleep(.01)
        c.close()
        while not (lane.state/'fail-stop.json').exists():time.sleep(.01)
        owner=json.loads((lane.state/'fail-stop.json').read_text())['owner']
        self.assertEqual(supervisor.process_identity(owner['pid'])['start_ticks'],owner['start_ticks'])
        self.assertEqual(lane.run()['status'],'refused')

    def test_connected_client_without_heartbeat_failstops(self):
        lane=self.lane(client_heartbeat_max_age_sec=.03)
        c=lane.connect(lane.request("import time;time.sleep(.5)"))
        r=json.loads(c.recv(65536));c.close()
        self.assertEqual(r['status'],'fail_stop');self.assertIn('heartbeat',r['error'])
        self.assertTrue(Path(f"/proc/{r['owner']['pid']}").exists())

    def test_ctest_timeout_owns_only_proxy(self):
        lane=self.lane()
        code="import time;open('ctest-started','w').write('1');time.sleep(.5)"
        env=dict(os.environ, SIMA_HARDWARE_SUPERVISOR_SOCKET=lane.socket, HOST_FIXTURE_CODE=code)
        fixture=lane.root/'CTestTestfile.cmake'
        fixture.write_text('add_test(host_fixture "'+PYTHON+'" "'+str(MODULE)+
                           '" ctest-proxy --test-name host_fixture -- "'+PYTHON+
                           '" -c "exec(__import__(\'os\').environ[\'HOST_FIXTURE_CODE\'])")\n'
                           'set_tests_properties(host_fixture PROPERTIES TIMEOUT 0.2)\n')
        completed=subprocess.run(['ctest','--test-dir',str(lane.root),'--output-on-failure'],
                                 env=env,capture_output=True,text=True,timeout=3)
        self.assertNotEqual(completed.returncode,0)
        self.assertIn('Timeout',completed.stdout)
        end=time.monotonic()+2
        while not (lane.state/'fail-stop.json').exists() and time.monotonic()<end:time.sleep(.005)
        failure=json.loads((lane.state/'fail-stop.json').read_text())
        owner=failure['owner']
        self.assertTrue((lane.root/'ctest-started').exists())
        self.assertEqual(supervisor.process_identity(owner['pid'])['start_ticks'],owner['start_ticks'])
        self.assertEqual(lane.run()['status'],'refused')

    def test_unknown_retirement_does_not_advance(self):
        lane=self.lane();lane.check['retirement']=False;lane.write_check()
        r=lane.run();self.assertEqual(r['status'],'fail_stop');self.assertFalse(r['retirement_proven'])
        self.assertEqual(lane.run()['status'],'refused')

    def test_wrong_retirement_kind_is_not_proof(self):
        lane=self.lane();lane.check['kind']='zero-sampled-jobs';lane.write_check()
        self.assertEqual(lane.run()['status'],'fail_stop')

    def test_changed_lease_boot_observer_or_cohort_refused(self):
        for key in ('lease_start_ticks','boot_id','observer_id','cohort_sha256'):
            with self.subTest(key=key):
                lane=self.lane();lane.check['identity']=dict(lane.identity);lane.check['identity'][key]='changed';lane.write_check()
                self.assertEqual(lane.run("open('unexpected','w').write('bad')")['status'],'fail_stop')
                self.assertFalse((lane.root/'unexpected').exists())

    def test_stale_probe_refused(self):
        lane=self.lane();lane.check['age_delta']=-10;lane.write_check()
        self.assertEqual(lane.run()['status'],'fail_stop')

    def test_second_admission_barrier_blocks_exec_on_disconnect(self):
        lane=self.lane();lane.check.update(sleep_phase='launch',sleep_seconds=.09);lane.write_check()
        c=lane.connect(lane.request("open('unexpected','w').write('bad')"))
        while not list(lane.state.glob('*/owner.json')):time.sleep(.002)
        time.sleep(.12);c.close()
        while not (lane.state/'fail-stop.json').exists():time.sleep(.01)
        self.assertFalse((lane.root/'unexpected').exists())

    def test_probe_crossing_deadline_cannot_late_pass(self):
        lane=self.lane(deadline=.04);lane.check.update(sleep_phase='active',sleep_seconds=.08);lane.write_check()
        r=lane.run("print('finished')",.04)
        self.assertEqual(r['status'],'timeout');self.assertFalse(r['retirement_proven'])

    def test_loader_injection_and_deadline_change_rejected(self):
        lane=self.lane();env=dict(os.environ,LD_PRELOAD='/tmp/evil.so')
        request=lane.request('print(1)');request['env']=env
        r=supervisor.run_owned(**request,socket_path=lane.socket);self.assertEqual(r['status'],'fail_stop')
        self.assertEqual(self.lane().run(deadline=.7)['status'],'fail_stop')

    def test_approved_graceful_stop_requires_exit_and_retirement(self):
        lane=self.lane();policy=dict(signal='SIGTERM',review_receipt_sha256='f'*64)
        lane.config['executables'][PYTHON]['graceful_stop']=policy
        # Config is immutable inside daemon; launch a second fixture using the
        # policy from initial admission, not a mutable post-start control file.
        configured=self.lane(completion_relative='outputs',executables={PYTHON:dict(sha256=supervisor.sha(PYTHON),deadline_sec=.8,graceful_stop=policy,arguments=['-c',"exec(__import__('os').environ['HOST_FIXTURE_CODE'])"])})
        code="import signal,sys,time;signal.signal(signal.SIGTERM,lambda *_:sys.exit(0));__import__('pathlib').Path('outputs').mkdir();open('outputs/result.txt','w').write('ok');time.sleep(.5)"
        r=configured.run(code,completion=dict(output_dir=str(configured.root/'outputs'),expected_files=1))
        self.assertEqual(r['status'],'completed');self.assertTrue(r['completion_condition_met']);self.assertTrue(r['retirement_proven'])

    def test_completion_without_stop_policy_is_not_pass(self):
        lane=self.lane();r=lane.run(completion=dict(output_dir=str(lane.root),expected_files=1))
        self.assertEqual(r['status'],'fail_stop')

    def test_quarantine_log_stops_scheduling(self):
        lane=self.lane();r=lane.run("import time;print('Decoder allocation owner quarantined until platform recovery',flush=True);time.sleep(.3)")
        self.assertEqual(r['status'],'fail_stop');self.assertFalse(r['retirement_proven'])
        self.assertEqual(lane.run()['status'],'refused')

    def test_argument_vector_is_pinned(self):
        lane=self.lane();request=lane.request('print(1)');request['command'] += ['--probe-only']
        self.assertEqual(supervisor.run_owned(**request,socket_path=lane.socket)['status'],'fail_stop')

    def test_injected_pid_reuse_identity_retains_owner(self):
        lane=self.lane()
        config=dict(lane.config,state_dir=str(lane.root/'inline'),socket=str(lane.root/'inline/socket'))
        owned=supervisor.Supervisor(config)
        left,right=socket.socketpair()
        real=supervisor.process_identity
        calls={}
        def changed(pid):
            value=real(pid)
            calls[pid]=calls.get(pid,0)+1
            if calls[pid]>1:value['start_ticks']+=1
            return value
        with mock.patch.object(supervisor,'process_identity',side_effect=changed):
            result=owned.run(lane.request("import time;time.sleep(.3)"),left)
        self.assertEqual(result['status'],'fail_stop')
        self.assertIn('PID/start identity',result['error'])
        self.assertIsNotNone(owned.retained)
        owned.retained.wait()  # bounded harmless fixture natural exit; no signal
        owned.check('retired',owned.root/result['run_id']/'owner.json')
        owned.lock.close();left.close();right.close()

    def test_active_observer_failure_retains_owner(self):
        lane=self.lane();lane.check['reject_phase']='active';lane.write_check()
        result=lane.run("import time;time.sleep(.3)")
        self.assertEqual(result['status'],'fail_stop')
        self.assertTrue(Path(f"/proc/{result['owner']['pid']}").exists())
        self.assertEqual(lane.run()['status'],'refused')

    def test_runner_cleanup_cookie_not_in_hardware_owner(self):
        lane=self.lane();request=lane.request("import os;print(os.environ.get('RUNNER_TRACKING_ID','absent'))")
        request['env']['RUNNER_TRACKING_ID']='HOST_FIXTURE_COOKIE'
        result=supervisor.run_owned(**request,socket_path=lane.socket)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['stdout'],'absent\n')
        executed=json.loads((lane.state/result['run_id']/'request.json').read_text())
        self.assertEqual(executed['safety_env_removed'],['RUNNER_TRACKING_ID'])

    def test_fail_stop_during_final_argument_hash_check_does_not_exec(self):
        lane=self.lane()
        config=dict(lane.config,state_dir=str(lane.root/'inline'),socket=str(lane.root/'inline/socket'))
        owned=supervisor.Supervisor(config);left,right=socket.socketpair()
        real=owned.approved_command;calls=[]
        def race(request):
            value=real(request);calls.append(1)
            if len(calls)==2:owned.fail('HOST_FIXTURE signal during final approval')
            return value
        with mock.patch.object(owned,'approved_command',side_effect=race):
            result=owned.run(lane.request("open('must-not-exec','w').write('bad')"),left)
        self.assertEqual(result['status'],'fail_stop')
        self.assertFalse((lane.root/'must-not-exec').exists())
        owned.retained.wait()
        owned.check('retired',owned.root/result['run_id']/'owner.json')
        owned.lock.close();left.close();right.close()

    def test_supervisor_cancellation_retains_active_child(self):
        lane=self.lane();c=lane.connect(lane.request("import time;open('signal-started','w').write('1');time.sleep(.4)"))
        while not (lane.root/'signal-started').exists():time.sleep(.005)
        os.kill(lane.daemon.pid,__import__('signal').SIGTERM)  # harmless HOST fixture daemon only
        result=json.loads(c.recv(65536));c.close()
        self.assertEqual(result['status'],'fail_stop')
        self.assertTrue(Path(f"/proc/{result['owner']['pid']}").exists())
        self.assertIsNone(lane.daemon.poll())

    def test_process_identity_includes_start_ticks(self):
        identity=supervisor.process_identity(os.getpid())
        self.assertNotEqual(identity,dict(identity,start_ticks=identity['start_ticks']+1))

    def test_control_source_has_no_implicit_owner_cleanup(self):
        text=MODULE.read_text()
        self.assertNotIn('.terminate(',text);self.assertNotIn('child.kill(',text)
        self.assertEqual(text.count('os.kill('),1)
        self.assertIn('EXPLICIT reviewed application stop request',text)
        self.assertNotIn('with subprocess.Popen',text)


if __name__=='__main__':unittest.main()
