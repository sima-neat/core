---
title: DevKit Containers
description: Install Docker on a Modalix DevKit, the dk container commands, Ubuntu containers, and troubleshooting
sidebar_position: 11
---

# DevKit Containers

This page is the reference for running containers on a Modalix DevKit with
platform version 3.0. For the overview and the end-to-end workflow, start with
[Develop Containerized Apps](/getting-started/dev-environment/containerized-apps/).

## Accelerator Access {#accelerators}

Neat applications can use the MLA inside a container. The container must have:

- Access to the required devices, such as `/dev/mla`, `/dev/cvu`, and
  `/dev/dma_heap/linux,cma`, passed with Docker's `--device` option.
- Compatible Neat libraries, GStreamer plugins, and platform user-space
  libraries, either packaged in the image or mounted from the DevKit.
- A compatible DevKit platform and a Neat Core version that supports container
  platform detection. The application must match the runtime's ABI.

On an up-to-date DevKit, run the workload with the DevKit user's UID and GID
using `--user UID:GID`, with permission to access the mapped devices. Once Docker
access is configured, MLA device access does not require application `sudo` or
`--privileged`. The current first-party launcher also uses no unconfined security
overrides. Administrator access for installing Docker and configuring the host is
separate from application device permissions.

The DevKit still provides the kernel drivers and firmware. Passing devices alone
does not make an arbitrary Ubuntu image accelerator-ready. DSP workloads require
their own compatible runtime and device configuration; this guide does not
establish DSP support for arbitrary container images.

For a complete device and library configuration, see the first-party
[Parallel Apps in Docker example](https://github.com/sima-neat/apps/tree/develop/examples/docker/parallel-apps).
Use the prerequisites for the selected example and platform version.

## Install Docker on the DevKit {#install}

Install Docker from the Palette SDK or with the standalone installer. A fresh
installation using either method below will:

- Install Docker CE (`docker-ce`, `docker-ce-cli`, and `containerd.io`) for
  ARM64 from Docker's Debian `trixie` repository.
- Store Docker's data in `/data/docker`, and containerd's data in
  `/data/containerd`, so that images do not fill the root file system.
- Turn on IP forwarding, which container networking needs.
- Start the `docker` and `containerd` services, and start them at every boot.
- Add the DevKit user to the `docker` group, so that the user can run `docker`
  without `sudo`.

Both need Internet access on the DevKit.

### From the Palette SDK {#install-sdk}

Pair the DevKit with `sima-cli sdk setup --devkit {devkit-ip}` first. Pairing
configures the passwordless `sudo` access needed for installation. Then, in the
SDK shell, run:

<ShellCommand prompt="sdk">
dk container setup
</ShellCommand>

Answer `y` when it asks to set up Docker on the DevKit. To skip the question,
run `dk container setup --yes`. If Docker already works on the DevKit, the
command reuses it. If an installation is interrupted, run the command again;
it resumes.

Other `dk container` commands also offer to install Docker when it is missing.

### Manual Installation on the DevKit {#install-manual}

Use the SDK's standalone installer when the Palette SDK is not available. On a
ARM64 Modalix DevKit, you need Internet access and a user
account with sudo access. Through SSH or the serial console, download the
installer from the SDK repository and run it as your normal user:

<ShellCommand prompt="devkit">
curl --fail --location --output install-devkit-docker.sh \
  https://raw.githubusercontent.com/sima-neat/sdk/main/scripts/install-devkit-docker.sh
bash install-devkit-docker.sh install
</ShellCommand>

The standalone `install` mode prompts for your sudo password when needed. It
uses the same Docker setup and interruption recovery as `dk container setup`,
including storage configuration and restoring the eLxr package mirror. Existing
Docker configuration is preserved. If interrupted, rerun the same command.

Log out and log in again to activate Docker group membership, then
[check the installation](#check).

## dk container Commands {#dk-container}

Run these commands in the Palette SDK shell. They connect to the paired DevKit
over SSH and run Docker there.

| Command | What it does | Needs the local registry |
| --- | --- | --- |
| `dk container setup [--yes]` | Installs Docker on the DevKit, or keeps a working installation. | No |
| `dk container deploy <image> [options] [-- command]` | Downloads the image from the local registry to the DevKit, then starts it. | Yes |
| `dk container run <image> [options] [-- command]` | Starts an image that is already on the DevKit. | Yes |
| `dk container pull <image>` | Downloads the image from the local registry to the DevKit. | Yes |
| `dk container images` | Lists the images on the DevKit that came from the local registry. | Yes |
| `dk container list` | Lists all containers on the DevKit, running or stopped. | No |
| `dk container logs <container> [options]` | Shows a container's output. Add `--follow` to keep watching. | No |
| `dk container stop <container>` | Stops a container. | No |
| `dk container remove <container> [options]` | Removes a container. `rm` also works. | No |
| `dk container help` | Shows the command summary. | No |

How the commands treat their arguments:

- **Image names.** Give the image name without the registry address, for
  example `my-app:latest`. `dk` adds the DevKit's address of the local registry.
  So `deploy`, `run`, and `pull` work only with images from the local registry.
- **Options.** Arguments after the image name and before `--` are `docker run`
  options, such as `--detach`, `--name`, `--rm`, or `--network host`. Arguments
  after `--` are the command to run in the container.
- **Architecture.** `deploy` and `run` refuse an image that is not ARM64.

For example, to start an image and override its command:

<ShellCommand prompt="sdk">
dk container run my-app:latest --rm -- sh -c 'uname -m'
</ShellCommand>

## Run an Ubuntu Container {#ubuntu}

Ubuntu images from Docker Hub run on the DevKit's Arm CPUs. To try one, run this
command on the DevKit:

<ShellCommand prompt="devkit">
docker run --rm ubuntu:24.04 cat /etc/os-release
</ShellCommand>

It prints `PRETTY_NAME="Ubuntu 24.04.5 LTS"` or a later point release. For an
interactive shell, run `docker run --rm -it ubuntu:24.04 bash`. Inside the
container, `apt-get` installs Ubuntu packages for ARM64.

The container shares the DevKit's Linux kernel; it is not a separate operating
system installation. Files you change in the container are lost when you remove
it. Use `-v` to mount a DevKit directory into the container for files you want
to keep.

To ship an application on Ubuntu, start its `Dockerfile` with
`FROM ubuntu:<release>`, and build and deploy it from the SDK as shown in
[Develop Containerized Apps](/getting-started/dev-environment/containerized-apps/).
`dk container run ubuntu:24.04` does not work, because `dk` looks for the image
in the local registry.

The Ubuntu commands above demonstrate CPU execution only. For Neat accelerator
workloads, follow [Accelerator Access](#accelerators); changing the base image
alone does not supply the required devices and runtime libraries.

## Check the Installation {#check}

On the DevKit:

<ShellCommand prompt="devkit">
docker info --format '{{.DockerRootDir}}'
docker run --rm hello-world
</ShellCommand>

The first command prints `/data/docker`. The second prints `Hello from Docker!`.

## Free Space on /data {#storage}

Docker keeps images and containers in `/data`, which is about 3 GB. Check the
free space on the DevKit:

<ShellCommand prompt="devkit">
df -h /data
</ShellCommand>

To free space, remove the containers and images you no longer need:

<ShellCommand prompt="devkit">
docker container prune
docker image prune -a
</ShellCommand>

`docker image prune -a` removes every image that no container uses.

## Troubleshooting {#troubleshooting}

| Message | Fix |
| --- | --- |
| `The local container registry is not configured for this SDK shell.` | Run `sima-cli sdk setup --devkit {devkit-ip}` on the host, answer `y` to the container registry question, then open a new SDK shell. |
| `Docker installation requires passwordless sudo` | Pair again with `sima-cli sdk setup --devkit {devkit-ip}`, or use the [standalone installer](#install-manual). |
| `Image ... is for amd64, not ARM64.` | Rebuild the image with `docker buildx build --platform linux/arm64 ... --push`. |
| `Image ... is not available on the DevKit.` | Use `dk container deploy` or `dk container pull` to download it first. |
| `permission denied while trying to connect to the Docker daemon socket` on the DevKit | Log out and log in again after Docker setup, so that your user is in the `docker` group. |
| `exec format error` while building | A `RUN` step in the `Dockerfile` needs ARM64 emulation on an `x86_64` host. Install it on the host with `docker run --privileged --rm tonistiigi/binfmt --install arm64`, then build again. |
