# Offline PCIe discovery qualification (draft)

Related: [Core #1132](https://github.com/sima-neat/core/issues/1132).

This directory contains a **read-only evidence collector** and the draft
[offline package assembler](ASSEMBLY.md).
The collector itself never installs or authorizes changes. The assembler bundles
a separate [installation entry point](INSTALL.md) that preflights the prepared
system, authenticates SSH, and deploys Neat to the host and all cards sequentially.
Existing host/card install lifecycles are reused; firmware, driver, networking and
SSH provisioning remain separate system prerequisites. Physical qualification is
still pending.

## Source findings

- The [sima-cli discovery client](https://github.com/sima-neat/sima-cli/blob/53b36826573b77832875ae38fcd40fe64f5153ef/sima_cli/discover/discover.py)
  sends the exact UDP payload `DISCOVER` to `239.255.42.1:50000`, binds source
  port `60000`, sets multicast TTL to 1, and parses JSON replies. This verifies
  the existing **client convention**, not responder availability in Platform 3.0.
- The client scans general interfaces and deduplicates by IP. Its advertised
  `ip`/`mac` fields do not establish a PCI-function-to-physical-card identity.
  The collector preserves the observed source IP separately from the untrusted
  reply. It never uses an advertised IP as an installation target.
- Inspected Platform host-driver archive `sima_pcie_host_pkg.sh`, SHA-256
  `9995721520d6a3522ab73ef564b4e6b553531cc4f1b4d6610f5d652d6f43953b`:
  `drivers/host/sima_mla_net_drv.c`, `si_mla_net_init`, lines 467 and 506,
  associates the netdev with `mla->pdev->dev` using `SET_NETDEV_DEV`.
  Lines 512–514 construct its MAC from `card_num` followed by `SIMA0`.
  This supports inspecting PCI-owned sysfs netdevs; it does **not** establish
  a persistent serial identity. The archive's Platform 3.0 provenance remains
  unverified and must be checked against the actual supported driver.
- Core's existing `pcie-setup.sh` infers endpoints from `10.0.N.1` addresses.
  That heuristic cannot prove complete attached-card discovery.

## Single-card hardware inspection: 2026-10-23 (Pacific)

Read-only inspection of a user-provided host established:

| Component | Observed value |
| --- | --- |
| Host | Ubuntu 24.04.4 amd64, kernel `7.0.0-38-generic` |
| Loaded driver | `simaai_mla_drv`, version `develop_B114` |
| Card | PCI `0000:01:00.0`, vendor/device `1f06:2123`, revision 02 |
| Host PCIe netdev | `veth-simaai0`, `10.0.0.1/24` |
| Card PCIe netdev | `veth-simaai`, `10.0.0.2/24` |
| Card firmware | eLxr/agate Modalix `3.0.0_develop_B2121`, build date `20261014` |
| Responder package | `simaai-discovery`, `3.0.0~git202610140145.4df44d0-2121` |

The collector ran from stdin without installing files or modifying services.
It inventoried the card correctly and exited 1 with zero multicast responses.
SSH inspection used the host's existing authorized card key.

### Confirmed responder implementation and configuration blocker

`/usr/libexec/simaai/simaai-discoveryd` has SHA-256
`0af5d37badadb168217369d366c8daaedac779f44d062c6f9aed1c5eb3a78f6a`.
The installed source confirms group `239.255.42.1`, UDP port `50000`, a
case-insensitive `DISCOVER` request, and a unicast JSON reply to the request's
source address/port. Response keys are `ip`, `mac`, `hostname`, `device`, and
`version`. The MAC is the configured interface's MAC; no persistent board serial
is returned. Source port 60000 is a client convention, not a responder requirement.

The service is enabled and active, but `/etc/default/simaai-discovery` sets
`DISCOVERY_IFACE=end0`. This card has `end3` and `veth-simaai`, **no `end0`**.
The daemon waits for a valid configured interface address before opening its
socket. `ss -ulnp` showed no UDP 50000 listener and `/proc/net/igmp` showed no
discovery-group membership. Thus service-active status alone is not readiness.

Step-1 provisioning must establish discovery on the PCIe interface. Merely
pointing this single-interface service at Ethernet would still not satisfy the
PCIe discovery requirement. Decide whether Platform will provide a PCIe-specific
instance or another supported multi-interface configuration; do not silently
replace an existing Ethernet discovery role. No configuration was changed and
no service was restarted during this inspection. A successful PCIe exchange is
still pending.

### Confirmed attachment and runtime device-index mapping

- `/sys/class/net/veth-simaai0/device` resolves to PCI `0000:01:00.0`.
- `/sys/class/sima_mla_c0/sima_mla_c0/dev_info` reports
  `PCIe Slot: 0000:01:00.0`, explicitly linking runtime device index 0 to the
  same PCI function without assuming sort order or deriving it from an IP.
- Installed `simaai_mla_drv_common.c` assigns `card_num` from a free driver slot
  and creates `sima_mla_c%d`. Installed `simaai_mla_net_drv.c` uses
  `SET_NETDEV_DEV` and derives the host MAC from `card_num` plus `SIMA0`.
  The host MAC was `00:53:49:4d:41:30`; the card PCIe MAC was
  `00:53:49:4d:41:31`. Neither is evidence of persistent physical identity.
- Installed source hashes: `simaai_mla_net_drv.c`:
  `7c14be9895f777d6580ce757f9692283d3bbcfdf7f659aa237b3a33df5f8ae94`;
  `simaai_mla_drv_common.c`:
  `72888dc91c2fec33369d10b038dc342e0cdc2a9dc5fed52c44270616251867b9`.

This verifies one live attachment/runtime-device mapping and the installed
responder implementation. It does not establish a persistent serial identity,
successful discovery, scheduler inference routing, or multi-card qualification.

## Collect evidence on a provisioned Linux PCIe host

Prerequisites: Python 3, iproute2, driver-loaded PCIe network interfaces with
one IPv4 address each, and privileges for Linux `SO_BINDTODEVICE`. Firmware
and networking must already be provisioned. No sima-cli or pip packages are needed.

```sh
sudo python3 pcie_host/scripts/offline/discover.py > discovery-evidence.json
```

The tool inventories **all SiMa PCI functions**, including unsupported device IDs
and functions without network interfaces. It does not yet classify functions as
qualified HHHL cards. It probes their directly associated netdevs sequentially
with bounded time/output; there is no MAC/name/IP-range fallback or LAN scan.
Only one netdev and one IPv4 address per function are accepted for this initial
qualification tool. Missing/ambiguous replies, malformed packets, missing netdevs,
permission errors, and inventory changes produce nonzero exit status. Healthy
functions are still inspected after another function fails.

A zero exit code means evidence collection completed; **it is not installation
approval**. JSON always contains `installation_authorized: false` and
`identity_status: unverified`. Replies are untrusted diagnostic data. Review
collected device/network identifiers before sharing the report.

## Required verification before implementing installation targeting

1. Obtain the actual Platform 3.0 responder source/service and exact firmware and
   host-driver versions. Confirm enablement, interface binding, response schema,
   and reply destination behavior against the client convention above.
2. On single-card and multi-card hosts, correlate each PCI BDF and sysfs netdev
   with a responder endpoint and an independently read, authenticated board
   identity. Confirm how the inference scheduler's card index maps to that same
   device; do not assume it equals PCI enumeration order.
3. Verify which PCI IDs/functions constitute each supported HHHL card and whether
   multifunction devices require grouping. A BDF is an attachment identifier for
   a run, not a serial number stable across replacement or reboot.
4. Exercise missing responders, duplicate identities, unrelated responses, and
   topology changes. Confirm that incomplete discovery cannot yield success.
5. Record the evidence and select the initial supported host/firmware combination.

Single-card inspection found the configuration blocker above. Successful PCIe
discovery, multi-card qualification, and complete stable identity mapping remain
pending. Until those gates pass, this collector must not feed package or
SSH installation commands.

## Subsequent implementation

Reuse existing host/card installers with isolated offline dependency inputs. The
small orchestrator will preflight host and every card before mutation, install
host software once, then install **and verify each card sequentially**. Individual
card failures will not skip remaining cards while host/inventory remain valid.
Reruns will inspect actual installed state and always repeat inference verification.
The package assembler collects dependencies and validates package transactions in
network-disabled containers. Integration with the existing installers, bundle
publication, package recovery, and offline hardware qualification remain tracked
in #1132; this draft does not close that issue.

## Tests

```sh
python3 -m pytest -q tests/tools/test_offline_pcie_discovery.py
```

These host-only tests exercise synthetic sysfs and sockets. They do not qualify
firmware, responder behavior, physical identity, or offline installation.
