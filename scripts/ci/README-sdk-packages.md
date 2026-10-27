# SDK package setup in CI

The shared `vulcan-build.yml@main` exports `SDK_PACKAGE_HELPER` from its own
revision into `RUNNER_TEMP`. Core calls it only after validating the SDK
container/image identity. It configures the SDK host package database before
`build.sh`, retrying recognized lock contention for 120 seconds by default.
Other failures remain immediately fatal. Diagnostics run inside that container;
the helper never removes locks, kills owners, or manages the target sysroot.

See the [shared contract](https://github.com/sima-neat/.github/tree/main/actions/setup-sdk-package-helper)
for timeout settings and validation. Package-setup success is separate from
Core build, hardware and publication acceptance.
