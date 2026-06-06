# godot-ml
A GDExtension-based neural inference backend for Godot 4.x, utilizing Compute Shaders for zero-copy GPU processing.


## Features
* **Asynchronous Execution**: Inference runs on the RenderingDevice without blocking the main thread.
* **Zero-Copy Pipeline**: (WIP) Texture-to-Buffer GPU-only data paths.
* **Modular Architecture**: Easily extensible `InputHandler/OutputHandler` and `Operator` classes.
* **ONNX Support**: Native parsing of ONNX models into compute-compatible graphs.

## Building

**Prerequisites**
- Python + [SCons](https://scons.org/)
- A C++20-capable compiler (MSVC 2019+, GCC 10+, Clang 10+)
- [vcpkg](https://vcpkg.io/) installed and `VCPKG_ROOT` pointing to it

**1. Clone**

The repo uses git submodules for `godot-cpp` and the ONNX proto definitions, so clone recursively:

```bash
git clone --recursive git@github.com:FrancoYudica/godot-ml.git
cd godot-ml
```

**2. Install dependencies via vcpkg**

The project ships a `vcpkg.json` manifest. Running `vcpkg install` from the project root reads it and installs protobuf and Abseil locally into `vcpkg_installed/` — nothing goes into your global vcpkg directory.

Static linking is required. The GDExtension must be a single self-contained binary with no external runtime DLLs to ship alongside it.

```bash
# Windows
vcpkg install --triplet x64-windows-static

# Linux
vcpkg install --triplet x64-linux-static

# macOS
vcpkg install --triplet arm64-osx-static
```

This step takes a while the first time since both libraries are compiled from source. Subsequent runs are instant.

**3. Build**

```bash
cd src
scons platform=windows   # or linux / macos
```

Or the following for release
```bash
scons platform=windows target=template_release
```

The extension is written to `demo/addons/ml/bin/`. The build picks up the vcpkg install automatically — no extra flags needed as long as `VCPKG_ROOT` is set.

If you need a non-default triplet or vcpkg location you can pass them explicitly:

```bash
scons platform=windows vcpkg_triplet=x64-windows-static vcpkg_root=C:/dev/vcpkg
```