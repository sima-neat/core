// src/pipeline/internal/DispatcherRecovery.cpp
#include "pipeline/internal/DispatcherRecovery.h"

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

  std::fprintf(stderr, "[WARN] dispatcher_recovery: attempting remoteproc reset + MLA init; "
                       "retry the command after recovery.\n");

  bool ok = true;
  // Stop userspace services before touching remoteproc to avoid races.
  run_cmd("printf '%s\n' edgeai | sudo -S -p '' systemctl stop simaai-appcomplex.service");
  run_cmd("printf '%s\n' edgeai | sudo -S -p '' systemctl stop simaai-pipeline-manager.service");
  run_cmd("printf '%s\n' edgeai | sudo -S -p '' systemctl stop rctd.service");
  run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'pkill -f mla_rt_service.py >/dev/null 2>&1 "
          "|| true'");

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
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'for i in $(seq 1 50); do [ \"$(cat "
                 "/sys/class/remoteproc/remoteproc0/state)\" != \"running\" ] && exit 0; sleep "
                 "0.1; done; exit 1'") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'for i in $(seq 1 50); do [ \"$(cat "
                 "/sys/class/remoteproc/remoteproc1/state)\" != \"running\" ] && exit 0; sleep "
                 "0.1; done; exit 1'") == 0);

  // Start remoteprocs and wait for them to report running.
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo start > "
                 "/sys/class/remoteproc/remoteproc1/state'") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo start > "
                 "/sys/class/remoteproc/remoteproc0/state'") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'for i in $(seq 1 50); do [ \"$(cat "
                 "/sys/class/remoteproc/remoteproc1/state)\" = \"running\" ] && exit 0; sleep 0.1; "
                 "done; exit 1'") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'for i in $(seq 1 50); do [ \"$(cat "
                 "/sys/class/remoteproc/remoteproc0/state)\" = \"running\" ] && exit 0; sleep 0.1; "
                 "done; exit 1'") == 0);

  // Re-enable recovery after successful restart.
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo enabled > "
                 "/sys/class/remoteproc/remoteproc0/recovery'") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'echo enabled > "
                 "/sys/class/remoteproc/remoteproc1/recovery'") == 0);

  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'sleep 1'") == 0);
  ok &= (run_cmd("printf '%s\n' edgeai | sudo -S -p '' sh -c 'for i in $(seq 1 50); do ls "
                 "/sys/bus/rpmsg/devices/*rpmsg_ns* >/dev/null 2>&1 && exit 0; sleep 0.1; done; "
                 "exit 1'") == 0);

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
