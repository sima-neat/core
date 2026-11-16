---
title: DevKit Containers
description: Install Docker on a Modalix DevKit, the dk container commands, Ubuntu containers, and troubleshooting
sidebar_position: 11
---

# DevKit Containers

This page is the reference for running containers on a Modalix DevKit with
platform version 3.0. For the overview and the end-to-end workflow, start with
[Develop Containerized Apps](/getting-started/dev-environment/containerized-apps/).

:::warning Containers cannot use the MLA or DSP
Using the MLA or the DSP from a container, including an Ubuntu container, is not
supported. Applications in a container run on the DevKit's Arm CPUs only. Run
Neat applications that use the MLA or the DSP directly on the DevKit.
:::

## Install Docker on the DevKit {#install}

The DevKit does not ship with Docker. Install it from the Palette SDK, or by
hand on the DevKit. Both methods:

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
gives the DevKit user passwordless `sudo`, which the installation needs. Then,
in the SDK shell, run:

<ShellCommand prompt="sdk">
dk container setup
</ShellCommand>

Answer `y` when it asks to set up Docker on the DevKit. To skip the question,
run `dk container setup --yes`. If Docker already works on the DevKit, the
command keeps it. If the installation is interrupted, run the command again; it
resumes.

Any other `dk container` command also offers to install Docker when it is
missing.

### By Hand on the DevKit {#install-manual}

Use this method if you do not use the Palette SDK. Run all commands below on
the DevKit, through SSH or the serial console. `sudo` asks for your password.

#### 1. Move the eLxr Package Mirror List Aside

You put it back in step 9.

<ShellCommand prompt="devkit">
sudo mkdir -p /root/apt-disabled
sudo mv /etc/apt/sources.list.d/0000mirror.list /root/apt-disabled/
</ShellCommand>

#### 2. Keep Docker's Data in /data

<ShellCommand prompt="devkit">
sudo mkdir -p /etc/docker /data/docker
echo '{"data-root": "/data/docker", "firewall-backend": "nftables"}' | sudo tee /etc/docker/daemon.json
</ShellCommand>

#### 3. Turn On IP Forwarding

<ShellCommand prompt="devkit">
printf 'net.ipv4.ip_forward=1\nnet.ipv6.conf.all.forwarding=1\n' | sudo tee /etc/sysctl.d/99-docker-forward.conf
sudo sysctl --system
</ShellCommand>

#### 4. Add Docker's Package Repository

<ShellCommand prompt="devkit">
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo 'deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian trixie stable' | sudo tee /etc/apt/sources.list.d/docker.list
</ShellCommand>

#### 5. Install Docker

<ShellCommand prompt="devkit">
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
</ShellCommand>

#### 6. Move containerd's Data to /data

<ShellCommand prompt="devkit">
sudo systemctl stop docker.socket docker containerd
sudo mkdir -p /data/containerd
sudo sh -c 'mv /var/lib/containerd/* /data/containerd/ 2>/dev/null; true'
</ShellCommand>

#### 7. Mount /data/containerd at Every Boot

<ShellCommand prompt="devkit">
echo '/data/containerd /var/lib/containerd none bind 0 0' | sudo tee -a /etc/fstab
sudo mount /var/lib/containerd
</ShellCommand>

#### 8. Start the Services

<ShellCommand prompt="devkit">
sudo systemctl enable containerd docker
sudo systemctl restart containerd docker
</ShellCommand>

#### 9. Put the eLxr Package Mirror List Back

<ShellCommand prompt="devkit">
sudo mv /root/apt-disabled/0000mirror.list /etc/apt/sources.list.d/
</ShellCommand>

#### 10. Add Your User to the docker Group

<ShellCommand prompt="devkit">
sudo usermod -aG docker "$USER"
</ShellCommand>

Log out and log in again for the group change to take effect.

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

:::warning No MLA or DSP in Ubuntu
An application in an Ubuntu container cannot use the MLA or the DSP. Use Ubuntu
containers for software that runs on the Arm CPUs only.
:::

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
| `Docker installation requires passwordless sudo` | Pair the DevKit again with `sima-cli sdk setup --devkit {devkit-ip}`, or install Docker [by hand](#install-manual). |
| `Image ... is for amd64, not ARM64.` | Rebuild the image with `docker buildx build --platform linux/arm64 ... --push`. |
| `Image ... is not available on the DevKit.` | Use `dk container deploy` or `dk container pull` to download it first. |
| `permission denied while trying to connect to the Docker daemon socket` on the DevKit | Log out and log in again after Docker setup, so that your user is in the `docker` group. |
| `exec format error` while building | A `RUN` step in the `Dockerfile` needs ARM64 emulation on an `x86_64` host. Install it on the host with `docker run --privileged --rm tonistiigi/binfmt --install arm64`, then build again. |
