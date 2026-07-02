# Building godot-ml

## Prerequisites

- Python + [SCons](https://scons.org/)
- A C++20-capable compiler (MSVC 2019+, GCC 10+, Clang 10+)
- [vcpkg](https://vcpkg.io/) installed and `VCPKG_ROOT` pointing to it

## 1. Clone

The repo uses git submodules for `godot-cpp` and the ONNX proto definitions, so clone recursively:

```bash
git clone --recursive git@github.com:FrancoYudica/godot-ml.git
cd godot-ml
```

## 2. Install dependencies via vcpkg

The project ships a `vcpkg.json` manifest. Running `vcpkg install` from the project root reads it and installs protobuf and Abseil locally into `vcpkg_installed/`. Nothing goes into your global vcpkg directory.

Static linking is required. The GDExtension must be a single self-contained binary with no external runtime DLLs to ship alongside it.

```bash
# Windows
vcpkg install --triplet x64-windows-static

# Linux
vcpkg install --triplet x64-linux

# macOS
vcpkg install --triplet arm64-osx
```

Windows needs the explicit `-static` triplet since vcpkg's default `x64-windows` triplet links
dynamically. On Linux and macOS, the plain triplet (`x64-linux`, `arm64-osx`) already builds
static libraries by default -- vcpkg doesn't ship a separate `-static` variant for those
platforms.

This step takes a while the first time since both libraries are compiled from source. Subsequent runs are instant.

## 3. Build

```bash
cd src
scons platform=windows   # or linux / macos
```

For a release build:

```bash
scons platform=windows target=template_release
```

The extension is written to `demo/addons/ml/bin/`. The build picks up the vcpkg install automatically -- no extra flags needed as long as `VCPKG_ROOT` is set.

If you need a non-default triplet or vcpkg location, pass them explicitly:

```bash
scons platform=windows vcpkg_triplet=x64-windows-static vcpkg_root=C:/dev/vcpkg
```
