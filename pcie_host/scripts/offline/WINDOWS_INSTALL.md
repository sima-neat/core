# Install the offline Neat PCIe bundle on Windows

This bundle installs the Neat PCIe host package on a Windows 11 x64 host
without internet access: the Scheduler and Core MSIs, the `pyneatpcie` wheel
and its Python dependencies, and the `sima-cli` that installs them. It also
holds the Neat Library package for the cards. It is the same release as the
online package, with the same files.

It does not install the SiMa Platform 3.0 Windows package, its drivers, or
card firmware. Install the Platform package first, as described in the PCIe
Driver Installation guide.

Copy the complete downloaded folder to a local disk of the host. The installer
writes its Windows Installer logs beside the packages, so the folder must be
writable. Run every command below in a PowerShell window that is not started as
administrator.

## 1. Install the prerequisites

In the downloaded folder, which holds this bundle and two installers, run the
GStreamer installer and choose a setup type that includes the development
files. Then run the Python installer with its default options:

```powershell
Start-Process -Wait (Get-Item .\gstreamer-1.0-msvc-x86_64-*.exe).FullName
Start-Process -Wait (Get-Item .\python-3.14.*-amd64.exe).FullName
```

Close PowerShell and open a new window, so that it sees the settings of both
installers. Skip a prerequisite that is already installed at the same version.

## 2. Install sima-cli from the bundle

Change to the bundle's `neat-offline` folder:

```powershell
Set-Location .\neat-pcie-offline-installer-*\neat-offline
```

If the folder holds only the `.tar.gz` archive, extract it first with
`tar -xzf (Get-Item .\neat-pcie-offline-installer-*.tar.gz).FullName`, then
run `Set-Location .\neat-offline`.

Install `sima-cli` into its own environment. pip installs only the files in the
bundle and checks each one against its SHA-256:

```powershell
py -3.14 -m venv "$env:USERPROFILE\.sima-cli-env"
& "$env:USERPROFILE\.sima-cli-env\Scripts\python.exe" -m pip install --no-index --find-links .\sima-cli --require-hashes -r .\sima-cli\requirements.txt
$cli = "$env:USERPROFILE\.sima-cli-env\Scripts\sima-cli.exe"
```

If PowerShell does not find `py`, use the Python you installed instead, for
example `& "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe" -m venv "$env:USERPROFILE\.sima-cli-env"`.

## 3. Install the host package

```powershell
& $cli install -m .\metadata.json
```

Windows asks once for administrator approval for the two MSI packages. Before
either MSI runs, `sima-cli` checks Windows OpenSSH, the Platform service,
GStreamer and Python, and checks every file of the bundle against its SHA-256.
A missing or changed file stops the installation before anything is installed.
The installation ends by setting up SSH to the card; OpenSSH asks once for the
password of the card's `sima` account.

The wheel is installed into `%USERPROFILE%\pyneatpcie`, with `numpy` and
`opencv-python` at the versions in the bundle. Use `--python-executable <path>`
to install into another environment, or `--native-only` for a C++-only
installation.

Check the installation:

```powershell
$venv = "$env:USERPROFILE\pyneatpcie"
$root = "$env:ProgramFiles\SimaPCIeHost"
& "$root\bin\sima-pcie-host-doctor.cmd"
& "$venv\Scripts\python.exe" -m pip check
& "$venv\Scripts\python.exe" -c "import pyneatpcie as p; print(p.__version__, p.Model)"
& "$root\bin\neat.cmd"
```

## 4. Install Neat on each card

Run these commands in the same `neat-offline` folder. Enter the password of the
card's `sima` account when the card installer asks. For another card, replace
`10.0.0.2` with its address, `10.0.N.2`.

```powershell
$bundle = (Get-Item .\card\neat-core-offline-installer.tar.gz).FullName
scp -i "$env:USERPROFILE\.ssh\sima_neat_pcie_ed25519" $bundle sima@10.0.0.2:/var/tmp/neat-core-offline-installer.tar.gz
ssh -t -i "$env:USERPROFILE\.ssh\sima_neat_pcie_ed25519" sima@10.0.0.2 "cd /var/tmp && rm -rf neat-core-offline && tar -xzf neat-core-offline-installer.tar.gz && cd neat-core-offline && ./install.sh && ./install.sh --verify && cd /var/tmp && rm -rf neat-core-offline neat-core-offline-installer.tar.gz"
& "$env:ProgramFiles\SimaPCIeHost\bin\neat.cmd"
```

The card installer needs no internet access. `neat.cmd` shows the Neat release
on the host and the card; they must be the same.

## Repair, upgrade and remove

Run the install command of step 3 again to repair the installation. To upgrade,
run it from the `neat-offline` folder of a newer bundle. Windows Installer does
not replace a package with an older version.

To remove the host package, use the version that `& $cli packages list` shows:

```powershell
& $cli packages remove gh:sima-neat/core/pciehost/windows/amd64 --version <version> --yes
```

Other `sima-cli` commands look for a newer `sima-cli` when they start and
continue when they cannot reach the internet. Run
`$env:SIMA_CLI_CHECK_FOR_UPDATE = "0"` first to skip that check.
