#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Input { float data[]; }
src;
layout(set = 0, binding = 1) buffer Output { float data[]; }
dst;

layout(push_constant) uniform PushConstants {
  uint rows; // input rows M
  uint cols; // input cols N
}
pc;

void main() {
  uint row = gl_GlobalInvocationID.x; // 0..M-1
  uint col = gl_GlobalInvocationID.y; // 0..N-1
  if (row >= pc.rows || col >= pc.cols)
    return;
  // Transpose: in[row, col] -> out[col, row]
  dst.data[col * pc.rows + row] = src.data[row * pc.cols + col];
}
