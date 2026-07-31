---
title: sccache Cheatsheet
description: Use and troubleshoot the Neat compiler cache locally and in Vulcan
sidebar_position: 2
slug: /develop-apps/contribute/sccache
---

# Neat `sccache` Cheatsheet

Neat uses [`sccache`](https://github.com/mozilla/sccache) as the C and C++
compiler launcher. It caches compiler outputs, not final Neat packages, test
results, dependency downloads, or Docker images.

For most developers there is nothing to install or configure: use `build.sh`
normally and inspect the cache statistics printed at the end.

## At a Glance

| Build | Cache levels | Writes | Survives `--clean` |
|---|---|---|---|
| Local | User-local disk | Local disk | Yes |
| Vulcan `develop` or `main` | Runner-local disk, then protected-branch S3 | Local disk and its protected S3 namespace | S3 does |
| Vulcan feature branch push | Runner-local disk, then branch S3 | Local disk and its isolated branch namespace | Until branch deletion |
| Vulcan tag or non-direct ref | Runner-local disk, then closest protected S3 | Local disk only | No persistent runner state |

The supported entry point is always:

```bash
./build.sh <options>
```

Do not invoke `sccache` as a replacement for the compiler or add launcher
options manually. `build.sh` supplies both CMake launchers:

```text
CMAKE_C_COMPILER_LAUNCHER
CMAKE_CXX_COMPILER_LAUNCHER
```

## Local Builds

### Normal use

Caching is enabled in `auto` mode:

```bash
./build.sh --dev-only
./build.sh --all --clean
```

The default cache location and limit are:

```text
~/.cache/sima-neat/sccache
10 GiB
```

The cache is outside `build/`. Deleting `build/` or passing `--clean` does not
delete cached compiler outputs.

If `sccache` is not on `PATH`, `build.sh` downloads the pinned release into:

```text
${XDG_CACHE_HOME:-$HOME/.cache}/sima-neat/tools/sccache/<version>/
```

The archive is verified against the SHA-256 recorded in
`scripts/configure_sccache.sh`. Linux and macOS on arm64 and x86-64 are
supported.

### Common controls

```bash
# Explicitly require sccache. Fail the build if it cannot be configured.
SIMANEAT_SCCACHE=on ./build.sh --all

# Disable caching for a reproducibility comparison.
SIMANEAT_SCCACHE=off ./build.sh --all --clean

# Put the local cache on a larger or faster volume.
SCCACHE_DIR=/mnt/nvme/sccache ./build.sh --all

# Change the local cache limit.
SCCACHE_CACHE_SIZE=20G ./build.sh --all
```

`SIMANEAT_SCCACHE=auto` is the default. In this mode, bootstrap failure produces
a warning and the build continues without caching. `on` makes that failure
fatal.

### Inspect or clear the local cache

Use the same binary selected by `build.sh`, or `sccache` when it is on `PATH`:

```bash
sccache --show-stats
sccache --zero-stats
sccache --show-adv-stats
```

To reclaim space, stop the server and remove only the configured cache
directory:

```bash
sccache --stop-server
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/sima-neat/sccache"
```

Confirm the resolved path before deleting it. Do not delete the entire user
cache directory.

## Vulcan Cloud Builds

Vulcan supplies the same local disk cache plus an encrypted S3 level:

```text
s3://sima-neat-compiler-cache-production/
  core/
    sccache-v1/
      <architecture>/
        <sdk-cache>/
          <build-mode>/
            develop/
              branches/<encoded-feature-branch>/
            main/
              branches/<encoded-feature-branch>/
```

For example:

```text
core/sccache-v1/arm64/sdk-develop/standard/develop/
```

The namespace deliberately includes:

- `sccache-v1`: cache schema, allowing an intentional global reset
- architecture: prevents arm64 and x86-64 compiler outputs from mixing
- SDK cache identity: prevents incompatible SDK/toolchain outputs from mixing
- build mode: keeps standard and fuzz instrumentation separate
- protected base branch: prevents `develop` and `main` from writing the same namespace

The S3 bucket is private, encrypted with its own KMS key, and separate from the
artifact bucket. It has no CloudFront distribution because compiler cache
objects are private and disposable. Objects expire automatically after 45
days.

### Branch access

| Git ref | OIDC role | S3 mode |
|---|---|---|
| Exact `refs/heads/develop` | Protected writer | `READ_WRITE` in `develop/` |
| Exact `refs/heads/main` | Protected writer | `READ_WRITE` in `main/` |
| Direct feature-branch push | Branch writer | `READ_WRITE` below `<base>/branches/<branch>/` |
| Tag or non-direct ref | Reader | `READ_ONLY` from the selected protected baseline |

On its first build, a feature branch copies the cache from the closest protected
Git ancestor (`develop` or `main`) into its own namespace. The build then reads
and writes only that branch namespace. Subsequent builds reuse it until GitHub's
branch-deletion event removes every architecture, SDK, and build-mode namespace
for that branch. Feature branches cannot write into either protected cache.

Automatic ancestry detection compares the merge-base distance to `develop` and
`main`. A reusable or manually dispatched workflow can set
`cache_base_branch=develop|main` when an exceptional branch needs an explicit
baseline. AWS credentials are short-lived GitHub OIDC credentials; no
long-lived AWS key is stored in GitHub or the SDK container.

An empty protected namespace is expected before its first writable protected
branch build. A feature branch can still populate its own namespace, but it
will receive no initial cache hits when its selected protected baseline is
empty.

## Reading Build Statistics

Every cached build ends with output similar to:

```text
Compile requests                    623
Cache hits                          619
Cache misses                          4
Cache hits rate                   99.36 %
Cache timeouts                        0
Cache read errors                     0
Cache write errors                    0
Compilations                          4
```

Interpret the important fields as follows:

| Field | Meaning |
|---|---|
| Compile requests | Compiler invocations seen by `sccache` |
| Cache hits | Requests restored without running the compiler |
| Cache misses | Requests that required compilation |
| Compilations | Compiler processes actually executed |
| Non-cacheable calls | Invocations that `sccache` intentionally bypassed |
| Read/write errors | Cache backend failures; investigate when nonzero on local or writable builds |
| Cache location | Active backend, such as local disk or multi-level |

A low hit rate is normal for the first build of a new toolchain, SDK cache,
architecture, build mode, or materially changed source tree. Judge the cache
with a second build of the same commit and configuration.

When any level is read-only, `sccache` v0.16 can report attempted stores as
write errors even though the read-only build succeeds. This applies to tags and
other non-direct contexts. Direct feature-branch pushes should report
`READ_WRITE`; investigate write errors in those builds.

## Quick Verification

### Verify local reuse

Run the same clean build twice:

```bash
SIMANEAT_SCCACHE=on SIMANEAT_SCCACHE_ZERO_STATS=ON \
  ./build.sh --dev-only --clean --no-dist

SIMANEAT_SCCACHE=on SIMANEAT_SCCACHE_ZERO_STATS=ON \
  ./build.sh --dev-only --clean --no-dist
```

The second run should have a much higher hit rate. Exact numbers vary with
source changes, compiler probes, generated files, and the selected build.

### Verify Vulcan configuration

In the GitHub Actions build log, look for:

```text
sccache enabled: sccache <version>
sccache local cache: <path> (<limit>)
sccache remote cache: s3://<bucket>/<prefix> (READ_ONLY|READ_WRITE)
```

Then verify the final statistics contain no unexpected read, write, timeout, or
cache errors.

## Troubleshooting

### `sccache` is not enabled

- Confirm the build uses `build.sh`.
- Check that `SIMANEAT_SCCACHE` is not `off`.
- Rerun with `SIMANEAT_SCCACHE=on` so bootstrap errors become fatal.
- Confirm `curl`, `tar`, and either `sha256sum` or `shasum` are available.

### The second local build still misses

- Confirm both builds use the same compiler, SDK, build mode, and flags.
- Confirm `SCCACHE_DIR` resolves to the same persistent directory.
- Look for generated inputs containing timestamps or changing absolute paths.
- Check `Non-cacheable calls` and `Unsupported compiler calls`.
- Confirm the cache was not evicted by `SCCACHE_CACHE_SIZE`.

### Vulcan shows zero remote hits

- Confirm the selected `develop` or `main` baseline has been populated.
- For a feature branch, confirm the log reports the expected base branch and
  its encoded branch-specific prefix.
- Compare architecture, SDK cache identity, and build mode.
- Confirm the log shows the expected bucket and prefix.
- Treat a cold namespace as normal; compare two identical builds.

### S3 startup fails with `AccessDenied`

The cache roles require prefix-scoped `s3:ListBucket` for the
`.sccache_check` probe, plus object permissions:

- reader: `GetObject`
- writer: `GetObject` and `PutObject`

Both roles also require the corresponding KMS permissions. Do not add S3 or KMS
permissions to the EC2 runner role as a workaround; fix the GitHub OIDC cache
role in Vulcan.

### Bypass the cache while diagnosing a compiler failure

```bash
SIMANEAT_SCCACHE=off ./build.sh --all --clean
```

If the failure remains, it is not caused by a cached compiler output.

## Ownership and Source of Truth

| Concern | Source |
|---|---|
| Local bootstrap, version, checksums, cache defaults | `scripts/configure_sccache.sh` |
| CMake launcher integration and statistics | `build.sh` |
| Branch role selection and cache namespace | `.github/workflows/vulcan-ci.yml` |
| Reusable Vulcan workflow inputs and OIDC setup | `sima-neat/.github` |
| S3, KMS, lifecycle, and cache IAM roles | `sima-neat/vulcan` |

When changing the cache schema, toolchain compatibility boundary, or access
model, update all affected repositories and this page together.
