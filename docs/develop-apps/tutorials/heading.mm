## Before You Start

These tutorials run in two different environments. Choose the path for the
category you want to follow before copying any commands.

### Neat Library tutorials on a DevKit

The Models & Inference, Graphs & Pipelines, Cameras & Streaming, and GenAI
categories use the target-side Neat Library. Install and download the core
tutorial bundle:

```bash
sima-cli neat install core -t extras
cd sima-neat-*-Linux-extras
```

The bundle contains prebuilt programs under `lib/sima-neat/tutorials/` and
source under `share/sima-neat/tutorials/`. Activate `~/pyneat` for Python, and
link copied C++ examples with `SimaNeat::sima_neat`.

### PCIe Co-Processing tutorials on an x86 host

The PCIe category runs on the x86 host connected to a Modalix PCIe Card. Follow
[Install PCIe Host](/getting-started/neat-library/pcie-host/) to install the
host packages and extract the separate PCIe host extras archive, then enter its
root:

```bash
cd pciehost-install/sima-pcie-host-*-Linux-amd64-extras
```

That bundle contains prebuilt programs under `lib/sima-pcie-host/tutorials/`
and source under `share/sima-pcie-host/tutorials/`. Activate `~/pyneatpcie` for
Python, and link copied C++ examples with
`SimaPCIeHost::sima_neat_pcie_host`.

## Quick Preflight

- Run commands from the extras root specified by the tutorial.
- Prepare the required model archive using the exact filename shown by the
  tutorial.
- Confirm the relevant Python environment imports `pyneat` or `pyneatpcie`
  before running a Python example.

For model naming and command-context details, see
[Before You Run Tutorials](/tutorials/before-you-run).
