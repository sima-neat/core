#!/usr/bin/env python3
"""Keep the native Windows package owned by Windows Installer."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from xml.etree import ElementTree


PACKAGING = Path(__file__).parents[1] / "cmake" / "PackagingWindows.cmake"
ROOT = Path(__file__).parents[1]
WINDOWS_PRODUCER = ROOT.parent / "scripts" / "ci" / "pciehost" / "build_windows.py"
ROLLBACK_FIXTURE = (
    ROOT.parent / "scripts" / "ci" / "pciehost" / "windows_msi_rollback_fixture"
)
SETUP_SCRIPT = ROOT / "scripts" / "windows" / "pcie-setup.ps1"
FIXTURE_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGZpeHR1cmUta2V5LW1hdGVyaWFsLW9ubHk= user@host"


def read_source(path):
  return (ROOT / path).read_text(encoding="utf-8")


def setup_enrolment_command():
  source = SETUP_SCRIPT.read_text(encoding="utf-8")
  return re.search(r"(?ms)^  \$remoteScript = @'\n(.*?)\n'@$", source)[1]


class WindowsPackagingContract(unittest.TestCase):
  def test_windows_genai_adapter_uses_platform_service_provider(self):
    source = (ROOT.parent / "shared" / "pcie_genai" / "ServiceWindows.cpp").read_text(
        encoding="utf-8"
    )
    for symbol in (
        "simaai_svc_open_card", "simaai_svc_subscribe", "simaai_svc_notify",
        "simaai_svc_recv", "simaai_svc_get_file", "simaai_svc_put_file",
        "simaai_svc_handle", "CancelIoEx",
    ):
      self.assertIn(symbol, source)
    self.assertNotIn("provider exposes vision queues only", source)

  def test_windows_runtime_never_loads_or_runs_a_bare_file_name(self):
    # LoadLibraryW and CreateProcessW search the current directory for a bare name, so a
    # file planted there would run. Resolve full paths like sima-pcie-host-env.cmd does.
    service = (ROOT.parent / "shared" / "pcie_genai" / "ServiceWindows.cpp").read_text(
        encoding="utf-8")
    platform = read_source("src/HostPlatform.cpp")
    for source in (service, platform, read_source("src/SshRunnerWindows.cpp")):
      self.assertNotRegex(source, r'(LoadLibrary|CreateProcess)\w*\(\s*L?"')
    self.assertNotIn('return L"SiMaPCIe.dll"', service)
    self.assertIn(r'L"SYSTEM\\CurrentControlSet\\Services\\SiMaMlaDaemon"', service)
    self.assertIn('L"ImagePath", RRF_RT_REG_SZ', service)
    self.assertIn("LoadLibraryW(library_path.c_str())", service)
    self.assertIn("advapi32 bcrypt ole32 shell32", read_source("CMakeLists.txt"))
    resolver = platform.split("std::string open_ssh_executable(", 1)[1].split("\n}\n", 1)[0]
    windows = resolver.split("#ifdef _WIN32\n", 1)[1].split("#else\n", 1)[0]
    self.assertEqual(re.findall(r"(?m)^ *(return [^;]+;)", windows),
                     ["return host_path_string(candidate);"] * 2)
    self.assertIn("throw std::runtime_error(", windows.rsplit("begin = end + 1;", 1)[1])

  def test_msi_has_stable_upgrade_and_install_ownership(self):
    source = PACKAGING.read_text(encoding="utf-8")
    self.assertIn('set(CPACK_GENERATOR "WIX")', source)
    self.assertIn('set(SIMAPCIE_WIX_INSTALL_DIRECTORY "SimaPCIeHost" CACHE STRING', source)
    self.assertIn('set(CPACK_WIX_INSTALL_SCOPE perMachine)', source)
    self.assertIn('set(CPACK_WIX_COMPONENT_INSTALL ON)', source)
    self.assertIn('set(CPACK_WIX_CMAKE_PACKAGE_REGISTRY "SimaPCIeHost")', source)
    self.assertIn('set(CPACK_COMPONENTS_ALL Runtime Development PcieHostTutorials)', source)
    self.assertIn('set(CPACK_COMPONENT_PCIEHOSTTUTORIALS_DEPENDS Runtime)', source)
    self.assertIn('"${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt" COPYONLY', source)
    self.assertNotIn('set(CPACK_GENERATOR "ZIP")', source)
    match = re.search(r'CPACK_WIX_UPGRADE_GUID "([0-9A-F-]+)"', source)
    self.assertIsNotNone(match)
    self.assertRegex(
        match.group(1),
        r"^[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}$",
    )

  def test_cxx_consumers_link_with_the_oldest_supported_toolset(self):
    producer = (WINDOWS_PRODUCER.parent / "run_windows_build.ps1").read_text(encoding="utf-8")
    self.assertIn('$msvcToolset = "14.44"', producer)
    self.assertIn('-DevCmdArguments "-arch=x64 -host_arch=x64 -vcvars_ver=$msvcToolset"', producer)
    self.assertIn('if (-not "$env:VCToolsVersion".StartsWith("$msvcToolset.")) {', producer)
    self.assertIn('string(REGEX MATCH "^[0-9]+[.][0-9]+" SIMAPCIE_MSVC_VERSION',
                  read_source("CMakeLists.txt"))
    self.assertIn(
        'if(MSVC AND _simapcie_msvc_version VERSION_LESS "@SIMAPCIE_MSVC_VERSION@")',
        read_source("cmake/SimaPCIeHostConfig.cmake.in"))
    self.assertIn("#if defined(_MSC_VER) && defined(_DEBUG)\n#error",
                  read_source("include/simaai/neat/pcie/Model.h"))

  @unittest.skipUnless(shutil.which("cmake"), "requires CMake")
  def test_consumer_toolset_check_uses_the_msvc_version_clang_cl_simulates(self):
    config = read_source("cmake/SimaPCIeHostConfig.cmake.in")
    check = config.split("include(CMakeFindDependencyMacro)", 1)[0].replace(
        "@PACKAGE_INIT@", "").replace("@SIMAPCIE_MSVC_VERSION@", "19.44")
    with tempfile.TemporaryDirectory() as directory:
      script = Path(directory) / "check.cmake"
      script.write_text(check, encoding="utf-8")
      for compiler, simulated, accepted in (
          ("19.44.35211", "", True),         # cl.exe of the producer toolset
          ("19.38.33130", "", False),        # an older cl.exe
          ("19.1.5", "19.44", True),         # clang-cl bundled with Visual Studio
          ("20.1.0", "19.38", False),        # newer clang-cl on an older MSVC toolset
      ):
        with self.subTest(compiler=compiler, simulated=simulated):
          result = subprocess.run(
              ["cmake", "-DMSVC=1", f"-DCMAKE_CXX_COMPILER_VERSION={compiler}",
               f"-DCMAKE_CXX_SIMULATE_ID={'MSVC' if simulated else ''}",
               f"-DCMAKE_CXX_SIMULATE_VERSION={simulated}", "-P", str(script)],
              capture_output=True, text=True, check=False)
          self.assertEqual(result.returncode == 0, accepted, result.stderr)
          if not accepted:
            self.assertIn(f"this project uses MSVC {simulated or compiler}",
                          " ".join(result.stderr.split()))

  def test_prebuilt_tutorials_ship_their_opencv_runtime(self):
    tutorials = read_source("tutorials/CMakeLists.txt")
    self.assertIn(
        "get_target_property(SIMAPCIE_OPENCV_RUNTIME opencv_world IMPORTED_LOCATION_RELEASE)",
        tutorials)
    rule = tutorials[tutorials.index('FILES "${CMAKE_CURRENT_BINARY_DIR}/opencv-runtime/'):]
    self.assertLess(rule.index("DESTINATION lib/sima-pcie-host/tutorials"),
                    rule.index("COMPONENT PcieHostTutorials"))
    self.assertIn('DIRECTORY "${OpenCV_INSTALL_PATH}/etc/licenses/"', tutorials)
    producer = WINDOWS_PRODUCER.read_text(encoding="utf-8")
    self.assertIn('"tutorial_024_run_image_mode.exe", "--help"], env=tutorial_environment)',
                  producer)

  def test_installed_tutorials_build_standalone(self):
    producer = WINDOWS_PRODUCER.read_text(encoding="utf-8")
    consumer = producer[producer.index("def check_consumer("):]
    consumer = consumer[:consumer.index('check_consumer(relocated, "relocated")')]
    self.assertIn('"-S", prefix / "share" / "sima-pcie-host" / "tutorials"', consumer)
    self.assertIn('"--target", "tutorial_024_run_image_mode"', consumer)
    self.assertIn('run([binary, "--help"], env=consumer_environment)', consumer)
    tutorials = read_source("tutorials/CMakeLists.txt")
    self.assertTrue(tutorials.startswith(
        "if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)\n"
        "  cmake_minimum_required(VERSION 3.16)"))
    self.assertIn("find_package(SimaPCIeHost REQUIRED CONFIG)", tutorials)
    self.assertIn('PATTERN "CMakeLists.txt"', tutorials)

  def test_msi_first_install_requires_exact_scheduler_msi(self):
    source = PACKAGING.read_text(encoding="utf-8")
    self.assertIn(
        'set(CPACK_WIX_PATCH_FILE "${CMAKE_CURRENT_LIST_DIR}/SchedulerDependency.wxs.xml")',
        source)
    self.assertIn('"-dSimaSchedulerVersion=${SIMAPCIE_SCHEDULER_PACKAGE_VERSION}"', source)
    self.assertIn('"-dSimaSchedulerUpgradeCode=${SIMAPCIE_SCHEDULER_WIX_UPGRADE_CODE}"', source)
    # Packaging without both values fails in cpack, as the DEB path does.
    self.assertIn("Native Core packaging requires its selected Scheduler MSI", source)
    self.assertIn("set(CPACK_PROJECT_CONFIG_FILE", source)

    patch = ElementTree.parse(PACKAGING.parent / "SchedulerDependency.wxs.xml").getroot()
    self.assertEqual(patch.tag, "CPackWiXPatch")
    (fragment,) = patch
    self.assertEqual(fragment.get("Id"), "#PRODUCT")
    upgrade = fragment.find("Upgrade")
    self.assertEqual(upgrade.get("Id"), "$(var.SimaSchedulerUpgradeCode)")
    (version,) = upgrade
    self.assertEqual(version.attrib, {
        "OnlyDetect": "yes",
        "Property": "SIMA_SCHEDULER_MATCHED",
        "Minimum": "$(var.SimaSchedulerVersion)",
        "IncludeMinimum": "yes",
        "Maximum": "$(var.SimaSchedulerVersion)",
        "IncludeMaximum": "yes",
    })
    (condition,) = fragment.findall("Condition")
    # Installed keeps repair, modify and uninstall available.
    self.assertEqual(condition.text, "Installed OR SIMA_SCHEDULER_MATCHED")
    message = condition.get("Message")
    self.assertIn("simaai-scheduler-pcie-host $(var.SimaSchedulerVersion)", message)
    self.assertIn(
        "sima-cli neat install core/pciehost/windows/amd64@v$(var.CPACK_PACKAGE_VERSION)", message)
    # Launch-condition messages are formatted text; brackets would expand.
    self.assertNotRegex(message, r"[\[\]]")

  def test_msi_installs_process_scoped_activation_tools(self):
    cmake = read_source("CMakeLists.txt")
    self.assertIn("scripts/windows/sima-pcie-host-env.cmd", cmake)
    self.assertIn("scripts/windows/sima-pcie-host-run.cmd", cmake)
    self.assertIn("scripts/windows/sima-pcie-host-doctor.cmd", cmake)
    self.assertIn("scripts/windows/pcie-setup.cmd", cmake)
    self.assertIn("scripts/windows/pcie-setup.ps1", cmake)
    self.assertIn("scripts/windows/neat.cmd", cmake)
    windows_block = cmake[cmake.index("if(WIN32)\n  get_target_property"):]
    self.assertIn("${SIMA_NEAT_CORE_ROOT}/scripts/neat_pcie_status.py\n"
                  "    DESTINATION ${CMAKE_INSTALL_DATADIR}/sima-pcie-host", windows_block)
    self.assertIn("COMPONENT Runtime", cmake)
    self.assertIn("SIMAPCIE_NLOHMANN_INCLUDE_ROOT", cmake)
    self.assertIn("nlohmann-json-LICENSE.MIT", cmake)
    self.assertIn("skills/neat-pcie-application-builder/playbook.yml", cmake)
    self.assertIn("skills/neat-pcie-application-builder/targets/", cmake)

    environment = read_source("scripts/windows/sima-pcie-host-env.cmd")
    self.assertIn("SIMA_WINDOWS_PLATFORM_RUNTIME_DIR", environment)
    # sc.exe labels and states are translated with the display language; the registry
    # value name, its type and the PowerShell status name are not.
    self.assertNotIn("sc.exe", environment)
    self.assertIn('reg.exe query "HKLM\\SYSTEM\\CurrentControlSet\\Services\\SiMaMlaDaemon" '
                  '/v ImagePath', environment)
    self.assertIn('do call set "_SIMA_PCIE_HOST_SERVICE=%%J"', environment)
    # A Platform path with spaces is one quoted item of the command line.
    self.assertIn('for %%K in (%_SIMA_PCIE_HOST_SERVICE%) do if not defined '
                  'SIMA_WINDOWS_PLATFORM_RUNTIME_DIR '
                  'set "SIMA_WINDOWS_PLATFORM_RUNTIME_DIR=%%~dpK"', environment)
    # cmd.exe can miss labels in LF-terminated scripts; the scripts use none.
    self.assertNotRegex(environment, r"(?m)^:|call :|goto ")
    # The documented override wins over the GStreamer installer's variable.
    self.assertLess(environment.index('"%GSTREAMER_ROOT_X86_64%"'),
                    environment.index('"%GSTREAMER_1_0_ROOT_MSVC_X86_64%"'))
    self.assertLess(environment.index('"%GSTREAMER_1_0_ROOT_MSVC_X86_64%"'),
                    environment.index('"%ProgramFiles%\\gstreamer\\1.0\\msvc_x86_64"'))
    self.assertIn("CMAKE_PREFIX_PATH", environment)
    self.assertIn("GST_PLUGIN_PATH_1_0", environment)
    # A user's SIMA_PCIE_HOST_LIBRARY_PATH survives, after the installation's bin.
    self.assertIn('if defined SIMA_PCIE_HOST_LIBRARY_PATH (set "SIMA_PCIE_HOST_LIBRARY_PATH='
                  '%SIMA_PCIE_HOST_ROOT%\\bin;%SIMA_PCIE_HOST_LIBRARY_PATH%")', environment)
    self.assertNotIn(r"C:\simaai\build", environment)
    self.assertNotIn("setx", environment.lower())

    python_bootstrap = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in ("python/pyneatpcie/__init__.py", "python/pyneatpcie/_windows_runtime.py")
    )
    self.assertIn("SIMA_WINDOWS_PLATFORM_RUNTIME_DIR", python_bootstrap)
    self.assertIn("GSTREAMER_ROOT_X86_64", python_bootstrap)
    self.assertIn("GSTREAMER_1_0_ROOT_MSVC_X86_64", python_bootstrap)
    self.assertIn("Services\\SiMaMlaDaemon", python_bootstrap)
    self.assertNotIn(r"C:\simaai\build", python_bootstrap)

    doctor = read_source("scripts/windows/sima-pcie-host-doctor.cmd")
    self.assertIn("SiMaMlaDaemon", doctor)
    self.assertNotIn("sc.exe", doctor)
    # The doctor checks the host only; neat reports the cards. No installer adds
    # SIMA_PCIE_HOST_ROOT\bin to PATH, so the hint names the full command.
    self.assertIn('echo ^& "%SIMA_PCIE_HOST_ROOT%\\bin\\neat.cmd"\n', doctor)
    self.assertNotIn("Run neat ", doctor)
    self.assertIn("(Get-Service -Name SiMaMlaDaemon -ErrorAction Stop).Status -ne 'Running'",
                  doctor)
    self.assertIn("gst-inspect-1.0.exe", doctor)

    setup_wrapper = read_source("scripts/windows/pcie-setup.cmd")
    setup_script = read_source("scripts/windows/pcie-setup.ps1")
    self.assertIn("pcie-setup.ps1", setup_wrapper)
    self.assertIn("ExecutionPolicy Bypass", setup_wrapper)
    self.assertNotIn("Set-ExecutionPolicy", setup_wrapper)
    self.assertIn("StrictHostKeyChecking=yes", setup_script)
    self.assertIn("StrictHostKeyChecking=accept-new", setup_script)
    self.assertNotIn("ssh-keyscan.exe", setup_script)
    self.assertNotIn("StrictHostKeyChecking=no", setup_script)
    self.assertNotIn("UserKnownHostsFile=NUL", setup_script)

    producer = WINDOWS_PRODUCER.read_text(encoding="utf-8")
    self.assertIn('install_root / "bin" / "pcie-setup.cmd"', producer)
    self.assertIn('"-Hosts", "10.0.0.2", "-DryRun"', producer)
    self.assertIn('install_root / "bin" / "neat.cmd"', producer)
    self.assertIn('install_root / "share" / "sima-pcie-host" / "neat_pcie_status.py"', producer)
    self.assertIn('"neat-pcie-application-builder" / "playbook.yml"', producer)

    status = read_source("scripts/windows/neat.cmd")
    self.assertIn(r"%NEAT_BIN%..\share\sima-pcie-host\neat_pcie_status.py", status)
    self.assertIn(r"%USERPROFILE%\pyneatpcie\Scripts\python.exe", status)
    self.assertIn("py.exe -3.14", status)
    self.assertIn("--render", status)
    self.assertIn("exit /b 64", status)

  def test_windows_scripts_treat_negative_exit_codes_as_failure(self):
    # "if errorlevel 1" is false for crash codes such as 0xC0000142 (-1073741502).
    scripts = sorted((ROOT / "scripts" / "windows").glob("*.cmd"))
    self.assertTrue(scripts)
    for script in scripts:
      with self.subTest(script=script.name):
        self.assertNotRegex(script.read_text(encoding="utf-8"),
                            r"(?im)^\s*if\s+(not\s+)?errorlevel\s+\d")
    doctor = read_source("scripts/windows/sima-pcie-host-doctor.cmd")
    self.assertEqual(doctor.count('if not "%errorlevel%"=="0" ('), 3)

  def test_windows_scripts_have_pinned_line_endings_and_no_labels(self):
    # The MSI ships the scripts as checked out; pin CRLF, which the hosted Windows
    # build has always produced, instead of relying on core.autocrlf.
    attributes = (ROOT.parent / ".gitattributes").read_text(encoding="utf-8").splitlines()
    for pattern in ("*.bat", "*.cmd", "*.ps1"):
      self.assertIn(f"{pattern} text eol=crlf", attributes)
    # A checkout that overrides the attribute still works: cmd.exe can miss labels
    # in LF-terminated scripts, and the scripts use none.
    for script in sorted((ROOT / "scripts" / "windows").glob("*.cmd")):
      with self.subTest(script=script.name):
        self.assertNotRegex(script.read_text(encoding="utf-8"), r"(?im)^\s*:|call :|goto ")

  def test_windows_scripts_resolve_their_own_directory_before_any_shift(self):
    # A plain shift also moves %0: "neat.cmd --json" then took %~dp0 from the current
    # directory and reported its helper missing everywhere but bin.
    script_name = re.compile(r"%~?[a-z$:]*0")
    for script in sorted((ROOT / "scripts" / "windows").glob("*.cmd")):
      with self.subTest(script=script.name):
        lines = script.read_text(encoding="utf-8").splitlines()
        shifts = [index for index, line in enumerate(lines)
                  if re.search(r"(?i)\bshift\b(?!\s*/)", line)
                  and not line.lstrip().lower().startswith("rem ")]
        if shifts:
          self.assertEqual([line for line in lines[shifts[0]:] if script_name.search(line)],
                           [])
    status = read_source("scripts/windows/neat.cmd")
    self.assertLess(status.index('set "NEAT_BIN=%~dp0"'), status.index("\" shift"))

  def test_windows_scripts_never_run_a_program_from_the_current_directory(self):
    # cmd.exe searches the current directory before PATH for a bare program name.
    for script in sorted((ROOT / "scripts" / "windows").glob("*.cmd")):
      with self.subTest(script=script.name):
        commands = "\n".join(
            line for line in script.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().lower().startswith("rem "))
        prefixes = re.findall(
            r"(?i)([^\s'(\"]*)\b(?:reg|findstr|powershell|where|sc|cmd)\.exe\b", commands)
        self.assertTrue(all(
            prefix in ("%SystemRoot%\\System32\\",
                       "%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\")
            for prefix in prefixes), prefixes)
    status = read_source("scripts/windows/neat.cmd")
    # py.exe and python.exe stay PATH lookups, without the current directory.
    self.assertLess(status.index('set "NoDefaultCurrentDirectoryInExePath=1"'),
                    status.index("py.exe -3.14"))
    self.assertLess(status.index("setlocal"),
                    status.index('set "NoDefaultCurrentDirectoryInExePath=1"'))

  def test_doctor_reports_driver_signing_state_without_failing(self):
    doctor = read_source("scripts/windows/sima-pcie-host-doctor.cmd")
    (line,) = [line for line in doctor.splitlines() if "Confirm-SecureBootUEFI" in line
               and line.startswith(
                   '"%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"')]
    # Runs only when no MLA device works, before the service check, and sets no exit code.
    self.assertIn("ConfigManagerErrorCode=0' -ErrorAction SilentlyContinue) { exit 0 }", line)
    self.assertLess(doctor.index(line), doctor.index("Get-Service -Name SiMaMlaDaemon"))
    self.assertEqual(doctor[doctor.index(line):].count('if not "%errorlevel%"=="0" ('), 2)
    # Non-elevated: the registry boot options replace bcdedit, and a refused
    # Confirm-SecureBootUEFI asks for an elevated check instead of failing.
    self.assertIn("SystemStartOptions -match 'TESTSIGNING'", line)
    self.assertNotIn("bcdedit.exe", line.lower())
    self.assertIn("catch { $secureBoot = 'unknown, run Confirm-SecureBootUEFI from an "
                  "elevated PowerShell to check' }", line)
    self.assertIn("Secure Boot off and test-signing on", line)
    # One cmd.exe-quoted argument: no inner double quotes or %-expansion.
    command = line.split(' -Command "', 1)[1]
    self.assertTrue(command.endswith('"'))
    self.assertNotIn('"', command[:-1])
    self.assertNotIn("%", command)

  def test_setup_sends_only_the_public_key_on_standard_input(self):
    source = SETUP_SCRIPT.read_text(encoding="utf-8")
    self.assertRegex(
        source,
        r"\n    \$publicKey \| & \$ssh -p \$Port -o ConnectTimeout=10 "
        r"-o StrictHostKeyChecking=yes `\n"
        r"        -o \$knownHostsOption \$target \$remoteScript\n")
    self.assertIn('$remoteScript = $remoteScript.Replace("`r", "")', source)
    # Windows PowerShell passes embedded double quotes to ssh.exe unescaped.
    self.assertNotIn('"', setup_enrolment_command())

  @unittest.skipIf(os.name == "nt", "card enrolment command runs in a POSIX shell; executed on Linux")
  def test_setup_enrolment_command_appends_exactly_one_public_key_line(self):
    command = setup_enrolment_command()
    for shell in ("sh", "bash"):
      executable = shutil.which(shell)
      self.assertIsNotNone(executable, shell)
      with self.subTest(shell=shell), tempfile.TemporaryDirectory() as home:
        def enrol(stdin):
          return subprocess.run(
              [executable, "-c", command], input=stdin, capture_output=True, text=True,
              check=False, env={**os.environ, "HOME": home})

        authorized_keys = Path(home) / ".ssh" / "authorized_keys"
        for _ in range(2):
          # PowerShell terminates the piped string with CRLF.
          result = enrol(FIXTURE_KEY + "\r\n")
          self.assertEqual(result.returncode, 0, result.stderr)
          self.assertEqual(authorized_keys.read_text(encoding="utf-8"), FIXTURE_KEY + "\n")
        self.assertEqual(authorized_keys.stat().st_mode & 0o777, 0o600)
        self.assertEqual(authorized_keys.parent.stat().st_mode & 0o777, 0o700)
        for invalid in ("", "umask 077\n", "ssh-ed25519 not-base64!\n", "ssh-ed25519\n"):
          result = enrol(invalid)
          self.assertNotEqual(result.returncode, 0, invalid)
          self.assertEqual(authorized_keys.read_text(encoding="utf-8"), FIXTURE_KEY + "\n")
        self.assertEqual(sorted(path.name for path in authorized_keys.parent.iterdir()),
                         ["authorized_keys"])

  def test_setup_tells_the_user_how_to_enrol_the_key_without_a_password(self):
    source = SETUP_SCRIPT.read_text(encoding="utf-8")
    help_text = source.split("function Write-ManualEnrollmentHelp {", 1)[1].split("\n}\n", 1)[0]
    self.assertIn("$PublicKeyPath to ~/.ssh/authorized_keys for $Target", help_text)
    self.assertIn("serial", help_text)
    self.assertIn(
        "echo '$publicKey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys",
        help_text)
    call = "    Write-ManualEnrollmentHelp -Target $target -PublicKeyPath $publicKeyPath\n"
    for failure in (
        'Write-Warning "Skipped: the key is not enrolled and non-interactive mode cannot prompt."\n',
        'Write-Warning "Public-key enrollment failed for $target."\n',
    ):
      with self.subTest(failure=failure):
        self.assertIn(failure + call + "    $", source)
    # The script never handles the card password itself.
    self.assertNotRegex(source, r"(?i)Read-Host|SecureString|sshpass")

  def test_setup_passes_known_hosts_paths_with_spaces_intact(self):
    source = SETUP_SCRIPT.read_text(encoding="utf-8")
    # OpenSSH splits -o values at whitespace but expands ${VAR} afterwards, so a
    # profile such as C:\Users\Jane Doe must reach ssh.exe through the environment.
    self.assertIn("$env:SIMA_NEAT_PCIE_KNOWN_HOSTS = $knownHostsPath", source)
    self.assertIn("$knownHostsOption = 'UserKnownHostsFile=${SIMA_NEAT_PCIE_KNOWN_HOSTS}'",
                  source)
    self.assertIn("$scanHostsOption = 'UserKnownHostsFile=${SIMA_NEAT_PCIE_SCAN_HOSTS}'",
                  source)
    self.assertEqual(source.count('"-o", $KnownHostsOption') +
                     source.count('"-o", $knownHostsOption') +
                     source.count('"-o", $scanHostsOption') +
                     source.count("-o $knownHostsOption"), 3)
    self.assertNotRegex(source, r"UserKnownHostsFile=\$(?!{)")

  def test_setup_refreshes_the_card_host_key_on_every_run(self):
    source = SETUP_SCRIPT.read_text(encoding="utf-8")
    loop = source.split("foreach ($cardHost in $targets) {", 1)[1]
    # pcie-setup.sh rescans every run; a reflashed or replaced card must be trusted again
    # without losing the saved key when the card does not answer.
    self.assertNotIn("$findExitCode", source)
    probe = loop.index('"-o", $scanHostsOption, $target, "true"')
    retrieved = loop.index("if ($currentEntries.Count -eq 0) {")
    removed = loop.index("& $sshKeygen -R $cardHost -f $knownHostsPath")
    self.assertLess(probe, retrieved)
    self.assertLess(retrieved, removed)
    self.assertIn("Remove-Item -LiteralPath $scanHostsPath", loop)
    self.assertIn("The SSH host key of $cardHost changed; recorded its current key.", loop)
    failure = loop[retrieved:removed]
    self.assertIn("Test-Connection -ComputerName $cardHost -Count 1", failure)
    self.assertIn("-Hosts $cardHost", failure)

  def test_setup_records_host_keys_where_runtime_ssh_reads_them(self):
    source = SETUP_SCRIPT.read_text(encoding="utf-8")
    # Runtime ssh.exe/scp.exe name no UserKnownHostsFile, so OpenSSH reads the
    # profile's known_hosts; -Key C:\keys\card must not move trust beside the key.
    self.assertIn('$knownHostsPath = Join-Path $env:USERPROFILE ".ssh\\known_hosts"', source)
    self.assertEqual(source.count("$knownHostsPath = "), 1)
    for runtime in ("src/RemoteRuntime.cpp", "src/SshRunner.cpp", "src/genai/RemoteSession.cpp"):
      with self.subTest(runtime=runtime):
        self.assertNotIn("UserKnownHostsFile", read_source(runtime))

  @unittest.skipUnless(os.name == "nt", "runs pcie-setup.ps1 with the Windows OpenSSH client")
  def test_setup_with_a_key_outside_the_profile_uses_the_profile_known_hosts(self):
    with tempfile.TemporaryDirectory() as root:
      profile = Path(root) / "profile"
      key = Path(root) / "keys" / "card"
      result = subprocess.run(
          ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
           str(SETUP_SCRIPT), "-Key", str(key), "-Hosts", "10.0.0.2", "-DryRun"],
          capture_output=True, text=True, check=False,
          env={**os.environ, "USERPROFILE": str(profile)})
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      self.assertIn(f"Known hosts: {profile / '.ssh' / 'known_hosts'}", result.stdout)

  def test_setup_key_probes_offer_only_the_neat_key(self):
    source = SETUP_SCRIPT.read_text(encoding="utf-8")
    # A key held by ssh-agent that the card already accepts would otherwise
    # report the Neat key as enrolled.
    probes = re.findall(r'(?s)Invoke-BoundedCommand -Executable \$ssh -Arguments @\((.*?)\)',
                        source, re.IGNORECASE)
    self.assertEqual(len(probes), 2)
    for probe in probes:
      with self.subTest(probe=probe):
        self.assertIn('"-i", ', probe)
        self.assertIn('"-o", "IdentitiesOnly=yes"', probe)

  def test_setup_uses_one_openssh_when_path_holds_several(self):
    source = SETUP_SCRIPT.read_text(encoding="utf-8")
    # Get-Command returns every match; a PC with both the in-box and the
    # Program Files OpenSSH would otherwise store an array as the command.
    self.assertRegex(
        source,
        r"Get-Command \$name -CommandType Application -ErrorAction SilentlyContinue \|\s*"
        r"Select-Object -First 1")

  def test_setup_discovers_cards_only_on_the_platform_pcie_adapter(self):
    source = SETUP_SCRIPT.read_text(encoding="utf-8")
    pattern = re.search(r"\$interface\.Description -notmatch '([^']+)'", source)[1]
    metadata = (ROOT / "scripts" / "create_windows_metadata.py").read_text(encoding="utf-8")
    self.assertIn('"SiMa.ai Virtual Ethernet Adapter"', metadata)
    adapters = {
        "SiMa.ai Virtual Ethernet Adapter": True,
        "SiMa.ai Virtual Ethernet Adapter #2": True,
        "Intel(R) Ethernet Connection (17) I219-LM": False,
        "Hyper-V Virtual Ethernet Adapter": False,
        "WireGuard Tunnel": False,
        "SiMa.ai Virtual Ethernet Adapter (copy)": False,
    }
    for description, is_card in adapters.items():
      with self.subTest(description=description):
        # PowerShell -match is case-insensitive.
        self.assertEqual(bool(re.search(pattern, description, re.IGNORECASE)), is_card)

  def test_setup_prefers_the_openssh_client_the_runtime_uses(self):
    source = SETUP_SCRIPT.read_text(encoding="utf-8")
    runtime = read_source("src/HostPlatform.cpp")
    # Enrollment and runtime must trust the card through the same client.
    self.assertIn('/ L"OpenSSH" / executable', runtime)
    self.assertIn("GetSystemDirectoryW", runtime)
    inbox = source.index('Join-Path ([Environment]::SystemDirectory) "OpenSSH\\$name"')
    self.assertLess(inbox, source.index("Get-Command $name -CommandType Application"))

  def test_setup_drains_redirected_process_output_while_command_runs(self):
    setup_script = read_source("scripts/windows/pcie-setup.ps1")
    bounded_command = setup_script.split(
        "function Invoke-BoundedCommand {", 1
    )[1].split("function Test-KeyAccess {", 1)[0]

    process_started = bounded_command.index("if (-not $process.Start())")
    stdout_drain = bounded_command.index(
        "$standardOutputDrain = $process.StandardOutput.ReadToEndAsync()"
    )
    stderr_drain = bounded_command.index(
        "$standardErrorDrain = $process.StandardError.ReadToEndAsync()"
    )
    bounded_wait = bounded_command.index(
        "if (-not $process.WaitForExit($TimeoutMilliseconds))"
    )
    process_kill = bounded_command.index("$process.Kill()")
    killed_process_exit = bounded_command.index("$process.WaitForExit()")
    drain_completion = bounded_command.index(
        "[System.Threading.Tasks.Task]::WaitAll("
    )

    self.assertLess(process_started, stdout_drain)
    self.assertLess(process_started, stderr_drain)
    self.assertLess(stdout_drain, bounded_wait)
    self.assertLess(stderr_drain, bounded_wait)
    self.assertLess(bounded_wait, drain_completion)
    self.assertLess(process_kill, drain_completion)
    self.assertLess(killed_process_exit, drain_completion)
    self.assertIn(
        "[System.Threading.Tasks.Task[]]@($standardOutputDrain, $standardErrorDrain)",
        bounded_command,
    )
    self.assertNotIn("StandardOutput.ReadToEnd()", bounded_command)
    self.assertNotIn("StandardError.ReadToEnd()", bounded_command)
    self.assertNotIn(".Result", bounded_command)
    self.assertNotIn("Write-", bounded_command)
    self.assertIn(
        "ExitCode = if ($timedOut) { $null } else { $process.ExitCode }",
        bounded_command,
    )
    self.assertIn("TimedOut = $timedOut", bounded_command)

  def test_ci_rollback_fixture_fails_after_windows_installer_owns_state(self):
    wxs = (ROLLBACK_FIXTURE / "rollback_fixture.wxs").read_text(encoding="utf-8")
    probe = (ROLLBACK_FIXTURE / "rollback_probe.c").read_text(encoding="utf-8")
    producer = WINDOWS_PRODUCER.read_text(encoding="utf-8")
    self.assertIn('Execute="deferred"', wxs)
    self.assertIn('After="WriteRegistryValues"', wxs)
    self.assertNotIn('After="InstallFiles"', wxs)
    self.assertIn('Id="RollbackSentinel"', wxs)
    self.assertIn('Id="RollbackFailureProbe"', wxs)
    self.assertIn('Key="Software\\SiMa.ai\\NeatPcieRollbackFixture"', wxs)
    self.assertIn("require_sibling_sentinel", probe)
    self.assertIn("require_registry_sentinel", probe)
    self.assertIn('"schema_version=1\\r\\nfile=present\\r\\nregistry=present\\r\\n"', probe)
    self.assertIn("return ERROR_INSTALL_FAILURE", probe)
    self.assertIn("qualify_failed_msi_transaction", producer)
    self.assertIn('"msi-transaction-rollback.json"', producer)
    self.assertNotIn("windows_msi_rollback_fixture", read_source("CMakeLists.txt"))

  def test_installed_package_smoke_covers_genai_cpp_and_python_apis(self):
    producer = WINDOWS_PRODUCER.read_text(encoding="utf-8")
    self.assertIn('PCIE_HOST / "tests" / "installed_genai_consumer"', producer)
    self.assertIn('"installed_genai_consumer.exe"', producer)
    self.assertIn("assert p.genai.GenAIModel", producer)
    self.assertIn("assert p.genai.GenerationRequest", producer)
    self.assertIn("assert p.genai.ConnectionOptions", producer)

  def test_installed_skill_preserves_qualified_windows_genai_api(self):
    skill_root = ROOT / "skills" / "neat-pcie-application-builder"
    skill = (skill_root / "SKILL.md").read_text(encoding="utf-8")
    validation = (skill_root / "references" / "build-and-validation.md").read_text(
        encoding="utf-8"
    )

    self.assertNotIn("Linux-host-only", skill)
    self.assertIn("release-qualified Platform", skill)
    self.assertIn("SiMaPCIe.dll", skill)
    self.assertIn("matching host/card packages", skill)
    self.assertNotIn("requires a Linux host", validation)
    self.assertIn("release-qualified Platform", validation)
    self.assertIn("connected-card acceptance", validation)

  def test_windows_genai_media_defaults_follow_platform_contract(self):
    header = (
        ROOT / "include" / "simaai" / "neat" / "pcie" / "genai" / "GenAIModel.h"
    ).read_text(encoding="utf-8")
    host_platform = (ROOT / "src" / "HostPlatform.cpp").read_text(encoding="utf-8")
    media_stage = (ROOT / "src" / "genai" / "MediaStage.cpp").read_text(
        encoding="utf-8"
    )
    cmake = read_source("CMakeLists.txt")
    consumer = (ROOT / "tests" / "installed_genai_consumer" / "main.cpp").read_text(
        encoding="utf-8"
    )
    producer = WINDOWS_PRODUCER.read_text(encoding="utf-8")

    self.assertIn("host_common_application_data_directory()", header)
    self.assertIn('media_directory = "/srv/simaai/data"', header)
    self.assertIn("SHGetKnownFolderPath(FOLDERID_ProgramData", host_platform)
    self.assertNotIn('_wgetenv(L"PROGRAMDATA")', host_platform)
    self.assertIn("bcrypt ole32 shell32", cmake)
    self.assertIn("SHGetKnownFolderPath(FOLDERID_ProgramData", consumer)
    self.assertNotIn(r"C:\ProgramData\SiMa.ai\data", producer)
    self.assertIn(r"%ProgramData%\SiMa.ai\simaai-mla-daemon.conf", media_stage)
    self.assertIn("missing or is not a directory", media_stage)
    self.assertIn("noncanonical or traverses a symlink", media_stage)


if __name__ == "__main__":
  unittest.main()
