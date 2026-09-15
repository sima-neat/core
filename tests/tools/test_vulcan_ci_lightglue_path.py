import subprocess
from pathlib import Path


def test_lightglue_executable_path_fits_runtime_limit():
    workflow = (Path(__file__).resolve().parents[2] / ".github/workflows/vulcan-ci.yml").read_text()
    prepare = workflow.split("- name: Prepare LightGlue NEAT-68 regression model", 1)[1]
    assignments = "\n".join(
        line.strip() for line in prepare.split("MODEL_SPEC=", 1)[0].splitlines()
        if line.strip().startswith(("TMP_ROOT=", "MODEL_ROOT="))
    )
    model_root = subprocess.check_output(
        ["/bin/bash", "-eu", "-c", assignments + '\nprintf "%s" "$MODEL_ROOT"'],
        env={
            "RUNNER_TEMP": "/media/nvme/actions-runner-sima-neat-modalix-linux-arm64-r9l8xx/_work/_temp",
            "GITHUB_RUN_ID": "34969312937",
            "GITHUB_RUN_ATTEMPT": "3",
        },
        text=True,
    )
    # Nested fixture, seven-digit worker PID, and Core's 16-digit package hash.
    executable = (
        f"{model_root}/modalix_int8_neat68_regression/.lightglue-extract-auto/"
        "proc_4194304/pkg_0123456789abcdef/p/share/"
        "quantized_lightglue_superpoint_600_stage1_mla.elf"
    )
    assert len(executable.encode("utf-8")) < 256, executable
