# godot-ml
A high-performance, GDExtension-based neural inference backend for Godot 4.x, utilizing Compute Shaders for zero-copy GPU processing.


## Features
* **Asynchronous Execution**: Inference runs on the RenderingDevice without blocking the main thread.
* **Zero-Copy Pipeline**: (WIP) Texture-to-Buffer GPU-only data paths.
* **Modular Architecture**: Easily extensible `InputHandler/OutputHandler` and `Operator` classes.
* **ONNX Support**: Native parsing of ONNX models into compute-compatible graphs.

## Building
Ensure you have `scons` and `protoc` (Protobuf Compiler) installed.

```bash
# Clone with submodules
git clone --recursive git@github.com:FrancoYudica/godot-ml.git

# Build the extension
scons platform=windows  # or linux/macos
```