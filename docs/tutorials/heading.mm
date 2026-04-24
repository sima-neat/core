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

## How to Run Tutorials

Before running anything, make sure NEAT is installed — `sima-neat.deb` is the library, `sima-neat-extras.deb` adds the prebuilt tutorial binaries and source.

Set the install root once per shell:

```bash
export NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
```

**On the DevKit** — activate the `pyneat` venv before running Python tutorials:

```bash
source ~/pyneat/bin/activate
```

**On the eLxr SDK** — prefix every command with `dk` (forwards to the paired DevKit). No venv activation needed on the SDK side.

### Python

Python tutorials are interpreted — no build step. Run the shipped script directly:

```bash
python3 $NEAT_EXTRAS_ROOT/share/sima-neat/tutorials/<chapter>/<chapter_name>.py --args
```

The `.py` file runs from any location. Copy it elsewhere if you want to modify it, and the same command works against the copy.

### C++ — run the prebuilt

Every chapter ships a compiled binary under `$NEAT_EXTRAS_ROOT/lib/sima-neat/tutorials/`:

```bash
$NEAT_EXTRAS_ROOT/lib/sima-neat/tutorials/tutorial_v2_<chapter_name> --args
```

### C++ — compile a copy yourself {#compile-a-copy-yourself}

Want to modify a chapter's C++ source, or reuse it inside your own project? You do not need `sima-neat-extras.deb` for this path — only `sima-neat.deb` (which provides `libsima_neat.so` and `SimaNeatConfig.cmake`).

Copy the `.cpp` into a new folder. Drop in this minimal `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_chapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(my_chapter <chapter_name>.cpp)
target_link_libraries(my_chapter PRIVATE SimaNeat::sima_neat)
```

Build and run:

```bash
cmake -S . -B build && cmake --build build -j
./build/my_chapter --args
```

`find_package(SimaNeat REQUIRED CONFIG)` auto-resolves headers, library, and dependencies from the installed NEAT — no hardcoded paths.

For the full template with SYSROOT handling (cross-builds from inside the eLxr SDK container), see the [Hello NEAT Minimal Example](/getting-started/minimal_example).

<p class="tutorial-grid-intro">Use these tutorials in order. Each card links to a chapter with concept-first guidance and matching C++ and Python implementation.</p>
