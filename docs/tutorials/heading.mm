## Prebuilt Binaries and Source Code

When prompted during installation of NEAT, select **SiMa NEAT extras** (press `Space` to toggle), then continue installation.

The extras package includes prebuilt tutorial binaries plus source code, installed under `/usr/` on the DevKit:

- `/usr/lib/sima-neat/tutorials/` — prebuilt C++ tutorial binaries (`tutorial_v2_*`).
- `/usr/share/sima-neat/tutorials/` — tutorial source code (`.cpp`, `.py`, `README.md`).

## How to Run Tutorials

### Verify your install

Both lists below should print 19 entries. If either is empty, re-run `sima-cli install neat` and select **SiMa NEAT extras**.

```bash
ls /usr/lib/sima-neat/tutorials/ | grep '^tutorial_v2_'
ls /usr/share/sima-neat/tutorials/ | grep -E '^0[0-9]{2}_'
```

Activate the `pyneat` virtual environment before running any Python tutorial:

```bash
source ~/pyneat/bin/activate
```

### Python

Python tutorials are interpreted — no build step. Run the script directly:

```bash
python3 /usr/share/sima-neat/tutorials/<chapter>/<chapter_name>.py --args
```

Copy the `.py` anywhere on disk if you want to modify it — the same command works from any location.

### C++ — run the prebuilt

```bash
/usr/lib/sima-neat/tutorials/tutorial_v2_<chapter_name> --args
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
