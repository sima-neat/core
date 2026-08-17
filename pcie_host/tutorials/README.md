# PCIe Host Tutorials

These tutorials run on an x86 host connected to a Modalix PCIe Card. They use
the installed `SimaPCIeHost` C++ package or the `pyneatpcie` Python wheel; they
do not use the target-side Neat runtime API.

The numbered chapters progress through:

1. synchronous YOLOv8s tensor, image, and image-plus-boxdecode modes;
2. asynchronous YOLOv8s image detection and throughput measurement;
3. ResNet-50 and YOLOv8s running concurrently on two PCIe queues.

YOLOv8s chapters use a detection-oriented scene. The Labrador image is reserved
for ResNet-50 classification in the multi-queue chapter.

## Packaged layout

The PCIe host extras archive is relocatable, but it does not contain an
enclosing directory. Create a destination directory before extracting it:

```bash
mkdir -p sima-pcie-host-extras
tar -xzf sima-pcie-host-<version>-Linux-<arch>-extras.tar.gz \
  -C sima-pcie-host-extras
cd sima-pcie-host-extras
```

The extracted layout is:

```text
./
├── build.sh
├── lib/sima-pcie-host/tutorials/      # prebuilt C++ binaries
└── share/sima-pcie-host/tutorials/    # C++, Python, docs, and assets
```

From the extracted extras root, list or rebuild the C++ tutorials with:

```bash
./build.sh --list-targets
./build.sh --target tutorial_024_run_tensor_mode
```
