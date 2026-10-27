## Helpers and other stuff for Build process

Root contents:
- `build-settings.json`: static settings for build process, including version of the dependencies. Used by CMake, Github Workflows and others. `Build.cxx` is the C++ standard of the dependencies (Conan `compiler.cppstd`), `Build.cxx-luxcore` the one of LuxCore code.
- `ccache.conf`: configuration for ccache. Used by Github Workflows

Folder contents:
- `cmake`: additional modules for CMake. Used by CMakeLists.txt.
- `conan`: project's conan recipe. Used by `make` helpers (see below)
- `debug`: (handcrafted) scripts for build debugging. Run on command line
- `luxmake`: underlying Python scripts of `make` wrapper - enabling `make deps`, `make <target>`, `make wheel`, `make clean` etc. Called by `GNUMakefile` and `make.bat`. `make wheel` builds the pyluxcore wheel with scikit-build-core (`pyproject.toml`) and repairs it (`luxmake/wheel.py`); the CI wheel builder (cibuildwheel) uses the same backend and calls the same repair (`make repair-wheel`).
