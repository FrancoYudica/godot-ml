# Building godot-infer

## Prerequisites

- [CMake](https://cmake.org/) 3.21+
- Visual Studio 2019+ (MSVC, C++20-capable)
- [vcpkg](https://vcpkg.io/) installed, with `VCPKG_ROOT` pointing to it (see step 2)

> **Platform support**: this build currently only covers **Windows**. Linux is supported by the CMake build itself but isn't walked through here yet. macOS/iOS/Android haven't been ported.

These steps use PowerShell.

## 1. Clone

The repo uses git submodules for `godot-cpp` and the ONNX proto definitions, so clone recursively:

```powershell
git clone --recursive git@github.com:FrancoYudica/godot-infer.git
cd godot-infer
```

## 2. Install vcpkg and set VCPKG_ROOT

If you don't already have vcpkg, clone and bootstrap it anywhere on your machine (it doesn't need to live inside this repo):

```powershell
git clone https://github.com/microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat
```

Then point `VCPKG_ROOT` at that folder, **persistently** (not just for the current shell session), so every future terminal and the CMake commands below all pick it up automatically:

```powershell
[System.Environment]::SetEnvironmentVariable("VCPKG_ROOT", "C:\path\to\vcpkg", "User")
```

(GUI alternative: Start menu -> search "Environment Variables" -> *Edit the system environment variables* -> *Environment Variables...* -> under *User variables*, *New...* -> Name `VCPKG_ROOT`, Value the full path to your vcpkg folder.)

Open a **new** PowerShell window afterward, then sanity-check it's set:

```powershell
echo $env:VCPKG_ROOT
```

## 3. Install dependencies via vcpkg

The project ships a `vcpkg.json` manifest. Running `vcpkg install` from the project root reads it and installs protobuf and Abseil locally into `vcpkg_installed/`. Nothing goes into your global vcpkg directory.

Static linking is required. The GDExtension must be a single self-contained binary, with no external runtime DLLs to ship alongside it. Windows needs the explicit `-static` triplet since vcpkg's default `x64-windows` triplet links dynamically.

```powershell
vcpkg install --triplet x64-windows-static
```

This step takes a while the first time since both libraries are compiled from source. Subsequent runs are instant.

## 4. Build

Two independent choices affect what you get, and both are always spelled out explicitly in the commands below. Don't rely on defaults.

- **`-DGODOTCPP_TARGET`**: `template_debug` or `template_release`. Picks which Godot API variant the extension is built against, and determines the output filename (`lib_godot_infer.windows.template_debug.*` vs `...template_release.*`). Godot's exporter looks for this specifically, a "Release" export in the Godot editor loads the `template_release`-named binary.
- **`--config Debug` / `--config Release`**: how *your own* C++ code is compiled, covering optimizations, debug symbols, and the MSVC runtime. Entirely independent of `GODOTCPP_TARGET` above, you can build `template_debug` optimized, or `template_release` unoptimized with debug symbols, if you have a reason to.

### Development build (editor / testing)

```powershell
cmake -S src -B build -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows-static -DGODOTCPP_TARGET=template_debug
cmake --build build --config Debug
```

### Shipping build (Release export)

**Both** `-DGODOTCPP_TARGET=template_release` at configure time **and** `--config Release` at build time are required. `--config Release` alone is not enough, since `GODOTCPP_TARGET` still defaults to `template_debug` and controls which binary variant (and filename) gets produced:

```powershell
cmake -S src -B build -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows-static -DGODOTCPP_TARGET=template_release
cmake --build build --config Release
```

The extension is written to `demo/addons/godot_infer/bin/`.

Notes:
- There's no default triplet. `-DVCPKG_TARGET_TRIPLET` is required.
- If you switch `GODOTCPP_TARGET` (e.g. from `template_debug` to `template_release`) or otherwise change configure-time options on an existing `build/` directory, re-run the configure command again before building. Reusing a stale `build/` directory configured with different options is the most common cause of confusing linker errors (mismatched MSVC runtime, wrong binary variant, etc.).
- `CMakeLists.txt` pins the MSVC runtime to match vcpkg's static-triplet libs for whichever CMake config you pick (`/MT` for Release, `/MTd` for Debug), working around a godot-cpp CMake quirk where `linux.cmake`/`windows.cmake` disagree on the default.
