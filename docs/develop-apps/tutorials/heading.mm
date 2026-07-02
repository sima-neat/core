## Before You Start

:::note Packaged tutorial bundle
Before running the packaged tutorials, make sure the Neat Library is installed,
then download the tutorial bundle:

```bash
sima-cli neat install core -t extras
```

The extras package contains the tutorial source and prebuilt tutorial binaries.
:::

The extras package gives you a self-contained tutorial folder with:

- prebuilt C++ tutorial binaries under `lib/sima-neat/tutorials/`
- C++ and Python tutorial source under `share/sima-neat/tutorials/`
- a `build.sh` helper for downloading tutorial models or rebuilding C++ examples

The tutorials do not install into system paths. They are extracted into your
current working directory as `sima-neat-<version>-Linux-extras/`. The folder
name includes the Neat version, branch, and commit hash.

```text
sima-neat-<version>-Linux-extras/
├── build.sh
├── lib/sima-neat/tutorials/      # prebuilt C++ binaries
└── share/sima-neat/tutorials/    # C++ and Python source folders
```

## Quick Preflight

Before you run a chapter, check three things:

- Run the command from the directory shown in the tutorial.
- For model-backed tutorials, pass the real `.tar.gz` model archive path with
  `--model <path>` when the default path does not exist.
- For Python tutorials on a DevKit, activate PyNeat first with
  `source ~/pyneat/bin/activate`.

If paths still do not line up, see [Tutorial Assets and Model Archives](/reference/tutorial-assets).

## Run a Tutorial

First enter the extracted extras folder:

```bash
cd sima-neat-*-Linux-extras
```

Your shell's current directory is now the root of the extras tree. Then run a
tutorial in the language you want to use.

### Python

Activate PyNeat on the DevKit, then run the tutorial script:

```bash
source ~/pyneat/bin/activate
python3 share/sima-neat/tutorials/<chapter>/<chapter_name>.py --args
```

Python tutorials are interpreted, so there is nothing to compile. You can copy
the `.py` file anywhere on disk if you want to modify it.

### C++

Run the prebuilt tutorial binary:

```bash
./lib/sima-neat/tutorials/tutorial_<chapter_name> --args
```

To rebuild a C++ tutorial from source:

```bash
./build.sh --list-targets
./build.sh --target tutorial_<chapter_name>
./build/tutorials-standalone/tutorial_<chapter_name> --args
```

`build.sh` auto-detects `SimaNeatConfig.cmake` from the installed Neat Library
and writes rebuilt binaries under `build/tutorials-standalone/`.

Some MPK tutorials need Model Zoo artifacts. `build.sh` downloads those required
models automatically before a C++ build. GenAI tutorials use LLiMa model
directories and are skipped by this automatic Model Zoo download path. To
download MPK tutorial models without rebuilding:

```bash
./build.sh --download-models-only
```

By default, tutorial models are prepared under `/tmp`. To use another download
root, add `--model-target-folder <path>`.

## Verify the Extras Folder

Both lists should print the same tutorial chapter names. If either list is
empty, download the extras package:

```bash
sima-cli neat install core -t extras
```

```bash
ls lib/sima-neat/tutorials/ | grep '^tutorial_'
ls share/sima-neat/tutorials/ | grep -E '^0[0-9]{2}_'
```

## Use a Tutorial in Your Own C++ Project

If you copy a tutorial `.cpp` file into your own codebase, you do not need the
extras folder anymore. You only need the installed `sima-neat` release artifacts,
which provide `SimaNeatConfig.cmake` and the Neat libraries.

Create a minimal `CMakeLists.txt` next to your source file:

```cmake title="CMakeLists.txt" {7,10}
cmake_minimum_required(VERSION 3.16)
project(my_chapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(my_chapter <chapter_name>.cpp)
target_link_libraries(my_chapter PRIVATE SimaNeat::sima_neat)
```

`find_package(SimaNeat REQUIRED CONFIG)` locates the installed Neat Library, and
`target_link_libraries(... SimaNeat::sima_neat)` brings in Neat's libraries,
headers, and transitive dependencies.

Build and run:

```bash
cmake -S . -B build && cmake --build build -j
./build/my_chapter --args
```

For a fuller template with SDK cross-build handling, see
[Hello Neat](/develop-apps/hello-neat/minimal).
