# godot-ml

<p align="center">
  <a href="https://godotengine.org/download/windows/">
      <img alt="Static Badge" src="https://img.shields.io/badge/Godot-4.5%2B-blue">
  </a>
  
  <a href="https://github.com/FrancoYudica/godot-ml/actions/workflows/tests.yml">
    <img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/FrancoYudica/godot-ml/tests.yml?label=tests">
  </a>

  <a href="LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/FrancoYudica/godot-ml">
  </a>
</p>

GPU-accelerated neural network inference for Godot, built as a GDExtension.

<img width="800" height="450" alt="mnist-sample" src="https://github.com/user-attachments/assets/661b9e6c-d636-4db7-958c-6d9de76a4bc2" />

## What is godot-ml?

godot-ml is an ONNX inference engine that runs directly inside Godot's rendering pipeline. You bring a trained model exported as [`.onnx`](https://onnx.ai/), godot-ml loads it, compiles the compute graph, and runs it on the GPU alongside your game's rendering, with no CPU-GPU transfers and no external runtime dependencies.

This is an inference engine, not a training framework. It is built on Godot's `RenderingDevice` API and supports every platform that supports compute shaders: Windows (Vulkan/D3D12), Linux (Vulkan), macOS (Metal), and more.

## Features

- **GPU-native execution**: inference runs in the same Vulkan/D3D12/Metal context as rendering. No context switches, no external runtimes, no CPU-GPU round-trips.
- **ONNX compatible**: loads models exported directly from PyTorch or TensorFlow. No manual conversion steps.
- **Dynamic input resolutions**: input dimensions are resolved at inference time, so screen resolution changes require no model reload.
- **100% GDScript API**: register models, queue inferences, and read results without writing any C++.
- **Native engine I/O**: accepts `Texture2D` inputs and writes outputs straight back to a texture, keeping all data on the GPU.
- **Multiple concurrent models**: each registered model gets its own GPU resources and can run within the same frame.
- **Async by design**: inference tasks complete via a `completed` signal, the main thread is never blocked.
- **Numerically validated**: every operator is tested against ONNX Runtime as reference.

## Documentation

- [Building from source](docs/building.md)
- [Development setup](docs/development.md)
