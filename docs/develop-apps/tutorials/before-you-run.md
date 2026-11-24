---
title: Tutorial Setup
description: Choose an execution environment, download the tutorials, and prepare model archives
sidebar_position: 2
slug: /tutorials/before-you-run
---

# Tutorial Setup

Complete this setup once before starting a tutorial. Choose the environment
that matches the tutorial category; the Neat Library and PCIe bundles are not
interchangeable.

## 1. Choose your environment

| Tutorial category | Run on | Python environment |
| --- | --- | --- |
| Models & Inference, Graphs & Pipelines, Cameras & Streaming, GenAI | Modalix DevKit or the environment named by the tutorial | `~/pyneat` |
| PCIe Co-Processing | Host connected to a Modalix PCIe Card | `~/pyneatpcie` |

PCIe tutorials run on the host, not inside the SDK container or directly on
the card.

## 2. Set up Neat Library tutorials

Make sure the [Neat Library is installed](/getting-started/neat-library/install-or-update/),
then run these commands from the directory where you want the tutorial bundle:

<ShellCommand prompt="sdk|devkit">
sima-cli neat install core@v%%release.core%% -t extras
cd sima-neat-*-Linux-extras
</ShellCommand>

For Python tutorials running directly on a DevKit, activate PyNeat and verify
the import:

<ShellCommand prompt="devkit">
source ~/pyneat/bin/activate
python3 -c "import pyneat; print('pyneat ready')"
</ShellCommand>

## 3. Set up PCIe tutorials

First [install and verify the PCIe host package](/getting-started/neat-library/pcie-host/).
Then download the tutorial bundle for the Ubuntu version running on the host.
Run the command from the directory where you want the bundle.

On Windows, the host package already includes the tutorials. Set them up as
described in
[Run applications on Windows](/getting-started/neat-library/pcie-host/#run-applications-on-windows).

**Ubuntu 22.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu22/amd64@v%%release.core%% -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

**Ubuntu 24.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu24/amd64@v%%release.core%% -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

Verify PCIe PyNeat:

<ShellCommand prompt="pcie-host">
source ~/pyneatpcie/bin/activate
python3 -c "import pyneatpcie; print('pyneatpcie ready')"
</ShellCommand>

## 4. Prepare model archives

Download the models that the tutorials use from the
[Model Browser](/tools/model-browser/) and extract each archive from its ZIP
file:

<ShellCommand prompt="sdk|devkit|pcie-host">
sima-cli models download --id resnet_50 --variant w8a8 --output models --force
sima-cli models download --id yolov8s --variant w8a8 --output models --force
unzip -o models/resnet_50/w8a8/*.zip resnet_50_w8a8_mpk.tar.gz
unzip -o models/yolov8s/w8a8/*.zip yolov8s_w8a8_mpk.tar.gz
</ShellCommand>

If the download is a `<id>_<variant>_mpk.tar.gz` file instead of a ZIP file, as
for a model last built before Model Browser packaged models as ZIP files, copy
that file from `models/<id>/<variant>` instead of extracting it. If `unzip` is
not installed, run `python3 -m zipfile -e models/yolov8s/w8a8/*.zip .` and the
same for `resnet_50` instead; it extracts every file in the ZIP file into the
current directory.

Neat Library tutorials accept `--model`, so you can pass the extracted archive
directly. PCIe tutorials use fixed filenames in the PCIe extras root:

| PCIe tutorial | Required model file |
| --- | --- |
| Run Your First Model over PCIe | `yolo_v8s_mpk.tar.gz` |
| Run PCIe Inference Async | `yolo_v8s_mpk.tar.gz` |
| Run Multiple Models | `resnet_50_mpk.tar.gz` and `yolo_v8s_mpk.tar.gz` |

For the PCIe tutorials, run the commands in the PCIe extras root and rename the
archives:

<ShellCommand prompt="pcie-host">
mv resnet_50_w8a8_mpk.tar.gz resnet_50_mpk.tar.gz
mv yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
</ShellCommand>

On Windows, run these PowerShell commands in your copy of the tutorials:

```powershell
sima-cli models download --id resnet_50 --variant w8a8 --output models --force
sima-cli models download --id yolov8s --variant w8a8 --output models --force
Expand-Archive models\resnet_50\w8a8\*.zip -DestinationPath models\resnet_50\w8a8 -Force
Expand-Archive models\yolov8s\w8a8\*.zip -DestinationPath models\yolov8s\w8a8 -Force
Copy-Item models\resnet_50\w8a8\resnet_50_w8a8_mpk.tar.gz resnet_50_mpk.tar.gz
Copy-Item models\yolov8s\w8a8\yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

## 5. Verify paths and expected output

Run tutorial commands from the extracted extras root. Confirm that it contains
the build helper, prebuilt C++ programs, and tutorial source:

<ShellCommand prompt="sdk|pcie-host">
test -x build.sh
ls lib/*/tutorials/
ls share/*/tutorials/
</ShellCommand>

- Prebuilt C++ programs are under `lib/<package>/tutorials/`.
- C++ and Python source is under `share/<package>/tutorials/`.
- `./build.sh --list-targets` lists the C++ programs you can rebuild.
- Successful C++ tutorials finish with `[OK]`; Python tutorials print a compact
  result such as `top1=...`, `completed=...`, or `detections=...`.

On Windows, the host package does not install `build.sh`. Run the tutorials as
described in
[Run applications on Windows](/getting-started/neat-library/pcie-host/#run-applications-on-windows).

If a tutorial reports a missing file, first check the current directory and the
model filename. For more help, see [Troubleshooting](/reference/troubleshooting/).
