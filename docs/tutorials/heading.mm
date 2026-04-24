## What Gets Installed

Running `sima-cli install neat` installs **two** things:

1. **The NEAT library** (the `sima-neat` package). Goes system-wide via `apt`. You never interact with it directly — your code links against it through CMake's `find_package(SimaNeat CONFIG)`.

2. **The tutorials** (the **SiMa NEAT extras** option, opt-in during install). Does **not** go into system paths. It extracts as a self-contained folder in your **current working directory**, looking like this:

   ```text
   sima-neat-0.0.0+<branch>-<sha>-Linux-extras/
   ├── build.sh                      ← build helper (auto-detects extras layout)
   ├── lib/
   │   └── sima-neat/
   │       └── tutorials/            ← prebuilt C++ binaries (tutorial_*)
   └── share/
       └── sima-neat/
           └── tutorials/            ← source folders (.cpp, .py, README.md)
   ```

The folder name includes the NEAT version, branch, and commit hash. For example, if you ran the installer from `~/neat/`, the folder ends up at `~/neat/sima-neat-0.0.0+<branch>-<sha>-Linux-extras/`.

When prompted during install, select **SiMa NEAT extras** (press `Space` to toggle the checkbox) to receive this folder. If you skip it, only the library gets installed and the tutorials are not on disk.

## How to Run Tutorials

Everything happens inside the extras folder. `cd` into it first:

```bash
cd sima-neat-*-Linux-extras
```

Your shell's current directory is now the root of the extras tree. All tutorial commands below use paths relative to here.

### Verify the install

Both lists should print the same set of chapter names. If either is empty, re-run `sima-cli install neat` and make sure **SiMa NEAT extras** is selected.

```bash
ls lib/sima-neat/tutorials/ | grep '^tutorial_'
ls share/sima-neat/tutorials/ | grep -E '^0[0-9]{2}_'
```

Activate the `pyneat` virtual environment before running any Python tutorial:

```bash
source ~/pyneat/bin/activate
```

### Python

Python tutorials are interpreted — there's nothing to compile. Run the script directly:

```bash
python3 share/sima-neat/tutorials/<chapter>/<chapter_name>.py --args
```

Copy the `.py` anywhere on disk if you want to modify it — the same command works from any location.

### C++ — run the prebuilt

Every chapter ships as a compiled binary. Run it straight from `lib/`:

```bash
./lib/sima-neat/tutorials/tutorial_<chapter_name> --args
```

### C++ — build from source with `build.sh` {#build-from-source}

If you want to recompile a chapter (to tweak it, or because the shipped binary does not match your runtime), run the bundled helper:

```bash
./build.sh --list-targets
./build.sh --target tutorial_<chapter_name>
./build/tutorials-standalone/tutorial_<chapter_name> --args
```

`build.sh` auto-detects `SimaNeatConfig.cmake` from the installed NEAT package, invokes CMake for the whole `tutorials/` tree (or a single target), and writes the binaries under `build/tutorials-standalone/`. No flags required for a stock eLxr SDK or on-device apt install.

### C++ — integrate NEAT into your own project {#compile-a-copy-yourself}

Copying a chapter's `.cpp` into your own codebase? Drop this minimal `CMakeLists.txt` alongside it — no extras folder required, only the base `sima-neat` package (which provides `libsima_neat.a` and `SimaNeatConfig.cmake`):

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

`find_package(SimaNeat REQUIRED CONFIG)` auto-resolves headers, library, and dependencies from the installed NEAT — no hardcoded paths, no extras folder required.

For the full template with SYSROOT handling (cross-builds from inside the eLxr SDK container), see the [Hello NEAT Minimal Example](/getting-started/minimal_example).

<p class="tutorial-grid-intro">Use these tutorials in order. Each card links to a chapter with concept-first guidance and matching C++ and Python implementation.</p>
