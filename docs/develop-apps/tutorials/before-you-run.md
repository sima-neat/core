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

<ShellCommand prompt="sdk-or-devkit">
sima-cli neat install core -t extras
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

**Ubuntu 22.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu22/amd64 -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

**Ubuntu 24.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu24/amd64 -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

Verify PCIe PyNeat:

<ShellCommand prompt="pcie-host">
source ~/pyneatpcie/bin/activate
python3 -c "import pyneatpcie; print('pyneatpcie ready')"
</ShellCommand>

## 4. Prepare model archives

Use Model Zoo to download the model named by the tutorial. For example:

<ShellCommand prompt="sdk-devkit-or-pcie-host">
sima-cli modelzoo get resnet_50
sima-cli modelzoo get yolo_v8s
</ShellCommand>

Neat Library tutorials accept `--model`, so you can pass the downloaded archive
directly. PCIe tutorials use fixed filenames in the PCIe extras root:

| PCIe tutorial | Required model file |
| --- | --- |
| Run Your First Model over PCIe | `yolo_v8s_mpk.tar.gz` |
| Run PCIe Inference Async | `yolo_v8s_mpk.tar.gz` |
| Run Multiple Models | `resnet_50_mpk.tar.gz` and `yolo_v8s_mpk.tar.gz` |

Model Zoo output names and locations can vary. If necessary, copy the archives
into the PCIe extras root using the required names:

<ShellCommand prompt="pcie-host">
cp /absolute/path/to/downloaded-resnet-archive.tar.gz resnet_50_mpk.tar.gz
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
</ShellCommand>

## 5. Verify paths and expected output

Run tutorial commands from the extracted extras root. Confirm that it contains
the build helper, prebuilt C++ programs, and tutorial source:

<ShellCommand prompt="sdk-or-pcie-host">
test -x build.sh
ls lib/*/tutorials/
ls share/*/tutorials/
</ShellCommand>

- Prebuilt C++ programs are under `lib/<package>/tutorials/`.
- C++ and Python source is under `share/<package>/tutorials/`.
- `./build.sh --list-targets` lists the C++ programs you can rebuild.
- Successful C++ tutorials finish with `[OK]`; Python tutorials print a compact
  result such as `top1=...`, `completed=...`, or `detections=...`.

If a tutorial reports a missing file, first check the current directory and the
model filename. For more help, see [Troubleshooting](/reference/troubleshooting/).
