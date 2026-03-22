```bash
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
./onnx_parser_test ../path/to/color_mlp_baseline.onnx

```