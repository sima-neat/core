## Prebuilt Binaries and Source Code

When prompted during installation of NEAT, select **SiMa NEAT extras** (press `Space` to toggle), then continue installation.

The extras package includes prebuilt tutorials/tests plus source code, and is automatically unpacked under the NEAT installation folder.

Prebuilt layout:

```text
.
├── lib
│   └── sima-neat
│       ├── tests
│       └── tutorials
└── share
    └── sima-neat
        ├── tests
        └── tutorials
```

- `lib/...`: prebuilt binaries
- `share/...`: source code

## Build Tutorials From Source

From the tutorial source directory, use `build.sh`:

```bash
./build.sh --list-targets
./build.sh
```

Build a single tutorial target:

```bash
./build.sh --target tutorial_v2_015_graph_model_hybrid
```

<p class="tutorial-grid-intro">Use these tutorials in order. Each card links to a chapter with concept-first guidance and matching C++ and Python implementation.</p>
