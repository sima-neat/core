---
title: Before You Run Tutorials
description: Prerequisites, model archives, command contexts, and quick checks for Neat tutorials
sidebar_position: 2
slug: /tutorials/before-you-run
---

# Before You Run Tutorials

Use this page once before your first tutorial. Neat Library tutorials and PCIe
host tutorials execute in different environments and use different packages,
Python environments, and extras bundles.

## Pick the execution environment

| Tutorial category | Run on | Python | C++ package | Extras paths |
| --- | --- | --- | --- | --- |
| Models & Inference, Graphs & Pipelines, Cameras & Streaming, GenAI | Modalix DevKit or the context named by the tutorial | `~/pyneat` | `SimaNeat` | `lib/sima-neat`, `share/sima-neat` |
| PCIe Co-Processing | x86 host connected to a Modalix PCIe Card | `~/pyneatpcie` | `SimaPCIeHost` | `lib/sima-pcie-host`, `share/sima-pcie-host` |

Do not run a PCIe host tutorial inside the SDK container or directly on the
card. Do not use the target-side core extras bundle for a PCIe tutorial.

## Prepare target-side Neat tutorials

Install Neat and download the core extras bundle:

<ShellCommand prompt="sdk-or-devkit">
sima-cli neat install core -t extras
cd sima-neat-*-Linux-extras
</ShellCommand>

For Python tutorials running directly on a DevKit, activate PyNeat:

<ShellCommand prompt="devkit">
source ~/pyneat/bin/activate
python3 -c "import pyneat; print('pyneat ready')"
</ShellCommand>

Target-side model tutorials accept `--model`, so you can pass the actual model
archive path when it differs from the example path.

## Prepare PCIe host tutorials

Install the host package and extract its separate extras archive by following
[Install PCIe Host](/getting-started/neat-library/pcie-host/). Enter the
extracted directory and verify PyNeat PCIe:

<ShellCommand prompt="pcie-host">
cd pciehost-install/sima-pcie-host-*-Linux-amd64-extras
source ~/pyneatpcie/bin/activate
python3 -c "import pyneatpcie; print('pyneatpcie ready')"
</ShellCommand>

PCIe tutorials intentionally accept only `--card N`. To keep their commands
short, they require these exact model filenames in the extras root:

| Model | Required PCIe tutorial filename |
| --- | --- |
| ResNet-50 | `resnet_50_mpk.tar.gz` |
| YOLOv8s | `yolo_v8s_mpk.tar.gz` |

Download the models you need:

<ShellCommand prompt="pcie-host">
sima-cli modelzoo get resnet_50
sima-cli modelzoo get yolo_v8s
</ShellCommand>

Model Zoo output names and locations can vary. If either archive was written
elsewhere or uses another name, copy it into the PCIe extras root using the
required filename:

<ShellCommand prompt="pcie-host">
cp /absolute/path/to/downloaded-resnet-archive.tar.gz resnet_50_mpk.tar.gz
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
test -f resnet_50_mpk.tar.gz
test -f yolo_v8s_mpk.tar.gz
</ShellCommand>

Only prepare the model required by the tutorial you are running. Tutorial 026
uses both models.

## Run from the extras root

Relative model and asset paths are intentional. From the correct extras root:

- Python source is under `share/<package>/tutorials/`.
- Prebuilt C++ programs are under `lib/<package>/tutorials/`.
- `./build.sh --list-targets` lists rebuildable C++ tutorial targets.

If an example reports a missing file, first confirm that the shell is in the
correct extras root and that the exact required model filename exists there.

## Know what success looks like

A successful tutorial prints a compact result such as `top1=...`,
`completed=1000`, or `detections=...`. C++ tutorials also print an `[OK]` line.
Timing and detection counts can vary by model package, host, card, and input,
but C++ and Python should report the same tensor contracts for the same mode.

For target-side source-tree asset overrides, see
[Tutorial Assets and Model Archives](/reference/tutorial-assets). For
symptom-first help, see [Troubleshooting](/reference/troubleshooting).
