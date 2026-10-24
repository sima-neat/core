# Offline PCIe discovery qualification (draft)

Related: [Core #1132](https://github.com/sima-neat/core/issues/1132).

This directory starts the offline bundle work with a **read-only evidence collector**.
It does not install packages, change networking, provision SSH keys, authenticate
card identity, or establish that a host/card combination is supported. Existing
host and card installers are unchanged.

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

No physical qualification or complete stable identity mapping has been performed
for this draft. Until those gates pass, this collector must not feed package or
SSH installation commands.

## Subsequent implementation

Reuse existing host/card installers with isolated offline dependency inputs. The
small orchestrator will preflight host and every card before mutation, install
host software once, then install **and verify each card sequentially**. Individual
card failures will not skip remaining cards while host/inventory remain valid.
Reruns will inspect actual installed state and always repeat inference verification.
Dependency closure, bundle publication, package recovery, and offline hardware
qualification remain tracked in #1132; this draft does not close that issue.

## Tests

```sh
python3 -m pytest -q tests/tools/test_offline_pcie_discovery.py
```

These host-only tests exercise synthetic sysfs and sockets. They do not qualify
firmware, responder behavior, physical identity, or offline installation.
