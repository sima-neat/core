---
title: Develop Containerized Apps
description: Build container images in the Palette SDK and run them on a Modalix DevKit
sidebar_position: 7
---

A container packages an application together with the libraries it needs, so
the application runs on the DevKit without installing those libraries on the
DevKit itself. With the Palette SDK, you build an ARM64 container image on your
host, push it to a local container registry, and start it on the DevKit with
`dk container`.

## When to Use a Container

Use a container for software that runs on the DevKit's Arm CPUs, for example:

- A service or tool that needs libraries that eLxr does not provide.
- An application built for another Linux distribution, such as Ubuntu. For
  example, a ROS 2 application built for a different ROS 2 release than the one
  available for eLxr.
- Several applications that need different versions of the same library.

:::note Accelerator access requires configuration
Neat applications can use the MLA from a container when the required device
nodes and compatible runtime libraries are available. An ordinary Ubuntu image
does not provide those dependencies. See
[Accelerator Access](/reference/devkit-containers/#accelerators) for requirements.
:::

## How It Works

The workflow uses the same three places as the rest of the Palette SDK:

| Where | What happens |
| --- | --- |
| Host | `sima-cli sdk setup --devkit` pairs the DevKit with the SDK and sets up a local container registry on the host. |
| SDK container | You build an ARM64 image with `docker buildx` and push it to the local registry. |
| DevKit | `dk container deploy` downloads the image from the local registry and starts it with Docker. |

The first `dk container` command offers to install Docker if it is missing.

## Before You Start

You need:

- Palette SDK %%release.sdk_series%% or later on the host.
- A Modalix DevKit with platform version 3.0, reachable from the host. To check
  the version, run `cat /etc/buildinfo` on the DevKit.
- Internet access on the DevKit, to download Docker.
- Free space on `/data` on the DevKit, where Docker keeps its images and
  containers. `/data` is about 3 GB, so keep images small.

## Build and Run a Container App

### 1. Pair the DevKit with the Local Registry

On the host, pair the DevKit with the SDK:

<ShellCommand prompt="host">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

When setup asks **Set up a local container registry so the DevKit can download
images built in the SDK?**, answer `y`. Pairing configures the passwordless
`sudo` access needed to install Docker. Then open a new SDK shell:

<ShellCommand prompt="host">
sima-cli sdk neat
</ShellCommand>

If you paired the DevKit earlier without the registry, run the setup command
again and answer `y`.

### 2. Set Up Docker on the DevKit

In the SDK shell, run:

<ShellCommand prompt="sdk">
dk container setup
</ShellCommand>

Answer `y` when it asks to set up Docker on the DevKit. If Docker already works
on the DevKit, the command reuses it. For what the installation changes, see
[Install Docker on the DevKit](/reference/devkit-containers/#install).

### 3. Build and Push an Image

Create a directory with a `Dockerfile` for your application. For example, a
`Dockerfile` that prints a message on Ubuntu 24.04:

```dockerfile
FROM ubuntu:24.04
CMD ["sh", "-c", "echo Hello from the DevKit"]
```

In the SDK shell, from that directory, build the image for ARM64 and push it to
the local registry:

<ShellCommand prompt="sdk">
docker buildx build --platform linux/arm64 -t "$SIMA_CONTAINER_REGISTRY/my-app:latest" --push .
</ShellCommand>

The DevKit runs only ARM64 images. Always build with
`--platform linux/arm64`.

### 4. Run the Image on the DevKit

In the SDK shell, download the image to the DevKit and start it:

<ShellCommand prompt="sdk">
dk container deploy my-app:latest --detach --name my-app
</ShellCommand>

You do not need the registry address in the image name; `dk` adds it. Options
after the image name are `docker run` options.

The example exits after printing its message. Read the output and remove the
stopped container:

<ShellCommand prompt="sdk">
dk container logs my-app --follow
dk container remove my-app
</ShellCommand>

For a long-running application, use `dk container logs my-app --follow` to watch
its output and `dk container stop my-app` before removing it.

## Next Steps

- For every `dk container` command, installing Docker by hand, running an Ubuntu
  container, and troubleshooting, see
  [DevKit Containers](/reference/devkit-containers/).
- For how the SDK and the DevKit share files and run commands, see
  [DevKit Sync](/getting-started/dev-environment/devkit-sync/).
