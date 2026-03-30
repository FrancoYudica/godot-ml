#[compute]
#version 450

// BUFFERS
layout(set = 0, binding = 0, std430) restrict readonly buffer InputBuffer {
  float A[]; // (M, K)
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer OutputBuffer {
  float out_data[]; // (M, K)
};

layout(push_constant) uniform PushConstants {
  uint M; // number of rows    (pixels)
  uint K; // number of columns (features)
};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
  uint m = gl_GlobalInvocationID.x;

  if (m >= M)
    return;

  // max(0, feature)
  for (uint k = 0; k < K; k++) {
    out_data[m * K + k] = max(0.0, A[m * K + k]);
  }
}