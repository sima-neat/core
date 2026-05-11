// src/pipeline/internal/DispatcherRecovery.cpp
#include "pipeline/internal/DispatcherRecovery.h"

#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace simaai::neat::pipeline_internal {
namespace {

int run_cmd(const char* cmd) {
  const int rc = std::system(cmd);
  if (rc != 0) {
    std::fprintf(stderr, "[WARN] dispatcher_recovery: command failed rc=%d: %s\n", rc, cmd);
  }
  return rc;
}

bool has_stuck_remoteproc_stop_task() {
  // If a prior "echo stop > .../remoteproc*/state" is stuck in kernel D-state,
  // any further recovery attempt is unsafe and will not recover in userspace.
  return run_cmd("sh -c \"ps -eo stat=,cmd= | grep -E '^D[[:space:]].*remoteproc[01]/state' "
                 "| grep -v grep >/dev/null\"") == 0;
}

bool rpmsg_channels_idle() {
  // Require no active RPMsg users before hard remoteproc reset.
  // fuser returns 0 when at least one process uses the path.
  return run_cmd(
             "printf '%s\\n' edgeai | sudo -S -p '' sh -c 'for p in /dev/rpmsg[0-9]* "
             "/tmp/rpmsg_lock /tmp/rpmsg_lock_rpmsg*; do "
             "[ -e \"$p\" ] || continue; "
             "if fuser -s \"$p\" 2>/dev/null; then echo busy:$p; exit 1; fi; "
             "done; exit 0'") == 0;
}

void append_note(SessionReport* report, const std::string& line) {
  if (!report)
    return;
  if (!report->repro_note.empty()) {
    report->repro_note += "\n";
  }
  report->repro_note += line;
}

} // namespace

bool match_dispatcher_unavailable(const std::string& message) {
  return message.find("Unable to connect to the server from dispatcher") != std::string::npos;
}

bool is_dispatcher_unavailable(const SessionReport& report) {
  return simaai::neat::error_codes::is_dispatcher_unavailable(report.error_code);
}

bool attempt_dispatcher_recovery(SessionReport* report, bool auto_recover) {
  append_note(report, "DispatcherUnavailable: auto recovery will run and you should retry.");
  if (!auto_recover) {
    append_note(
        report,
        "DispatcherUnavailable: auto recovery disabled; run the recovery commands and retry.");
    return false;
  }

  const bool hard_reset_requested = env_truthy("SIMA_NEAT_RECOVERY_HARD_RESET");
  const bool unsafe_hard_reset = env_truthy("SIMA_NEAT_RECOVERY_ALLOW_UNSAFE_RESET");
  bool do_hard_reset = hard_reset_requested;

  std::fprintf(stderr, "[WARN] dispatcher_recovery: attempting recovery (services + MLA init); "
                       "hard remoteproc reset is %s.\n",
               do_hard_reset ? "enabled" : "disabled");

  bool ok = true;
  // Stop userspace services before touching remoteproc to avoid races.
  run_cmd("printf '%s\n' edgeai | sudo -S -p '' systemctl stop simaai-appcomplex.service");
  run_cmd("printf '%s\n' edgeai | sudo -S -p '' systemctl stop simaai-pipeline-manager.service");
  run_cmd("printf '%s\n' edgeai | sudo -S -p '' systemctl stop rctd.service");
  run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'pkill -f mla_rt_service.py >/dev/null 2>&1 "
          "|| true'");

  if (do_hard_reset && has_stuck_remoteproc_stop_task()) {
    append_note(report,
                "DispatcherUnavailable: found stuck remoteproc stop task (kernel D-state). "
                "Skipping hard reset to avoid wedge; reboot recommended.");
    do_hard_reset = false;
  }
  if (do_hard_reset && !unsafe_hard_reset && !rpmsg_channels_idle()) {
    append_note(report,
                "DispatcherUnavailable: RPMsg nodes are in use; skipping hard remoteproc reset "
                "for safety. Set SIMA_NEAT_RECOVERY_ALLOW_UNSAFE_RESET=1 to override.");
    do_hard_reset = false;
  }

  if (do_hard_reset) {
    append_note(report, "DispatcherUnavailable: hard remoteproc reset enabled.");

    // Disable recovery so remoteproc stop actually takes effect.
    ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo disabled > "
                   "/sys/class/remoteproc/remoteproc0/recovery'") == 0);
    ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo disabled > "
                   "/sys/class/remoteproc/remoteproc1/recovery'") == 0);

    // Stop remoteprocs (ignore transient stop error; validate via offline wait).
    run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo stop > "
            "/sys/class/remoteproc/remoteproc0/state || true'");
    run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo stop > "
            "/sys/class/remoteproc/remoteproc1/state || true'");
    ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'for i in $(seq 1 50); do [ "
                   "\"$(cat /sys/class/remoteproc/remoteproc0/state)\" != \"running\" ] && exit 0; "
                   "sleep 0.1; done; exit 1'") == 0);
    ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'for i in $(seq 1 50); do [ "
                   "\"$(cat /sys/class/remoteproc/remoteproc1/state)\" != \"running\" ] && exit 0; "
                   "sleep 0.1; done; exit 1'") == 0);

    // Start remoteprocs and wait for them to report running.
    ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo start > "
                   "/sys/class/remoteproc/remoteproc1/state'") == 0);
    ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo start > "
                   "/sys/class/remoteproc/remoteproc0/state'") == 0);
    ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'for i in $(seq 1 50); do [ "
                   "\"$(cat /sys/class/remoteproc/remoteproc1/state)\" = \"running\" ] && exit 0; "
                   "sleep 0.1; done; exit 1'") == 0);
    ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'for i in $(seq 1 50); do [ "
                   "\"$(cat /sys/class/remoteproc/remoteproc0/state)\" = \"running\" ] && exit 0; "
                   "sleep 0.1; done; exit 1'") == 0);

    // Re-enable recovery after successful restart.
    ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo enabled > "
                   "/sys/class/remoteproc/remoteproc0/recovery'") == 0);
    ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo enabled > "
                   "/sys/class/remoteproc/remoteproc1/recovery'") == 0);
  } else {
    append_note(
        report,
        "DispatcherUnavailable: skipping hard remoteproc reset (safe mode). "
        "Enable SIMA_NEAT_RECOVERY_HARD_RESET=1 for explicit hard reset.");
  }

  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'sleep 1'") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'for i in $(seq 1 50); do ls "
                 "/sys/bus/rpmsg/devices/*rpmsg_ns* >/dev/null 2>&1 && exit 0; sleep 0.1; done; "
                 "exit 1'") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'for i in $(seq 1 120); do "
                 "n=$(ls /dev/rpmsg[0-9]* 2>/dev/null | wc -l); "
                 "[ \"$n\" -ge 1 ] && exit 0; sleep 0.1; done; exit 1'") == 0);

  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'for rp in "
                 "/sys/class/remoteproc/remoteproc0 "
                 "/sys/class/remoteproc/remoteproc1; do "
                 "echo \"$rp: $(cat $rp/name) state=$(cat $rp/state)\"; done'") == 0);

  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' /usr/bin/init_mla_memory.sh") == 0);
  ok &= (run_cmd(
             "printf '%s\n' edgeai | sudo -S -p '' systemctl restart simaai-appcomplex.service") ==
         0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' systemctl restart "
                 "simaai-pipeline-manager.service") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' systemctl restart rctd.service") == 0);

  if (ok) {
    append_note(report, "DispatcherUnavailable: auto recovery completed; retry the command.");
  } else {
    append_note(report,
                "DispatcherUnavailable: auto recovery failed (sudo may require a password). "
                "Run the recovery commands manually and retry.");
  }
  return ok;
}

} // namespace simaai::neat::pipeline_internal
