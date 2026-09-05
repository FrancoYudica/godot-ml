#[compute]
#version 450

layout(set = 0, binding = 0, std430) restrict readonly buffer InputBuffer {
  float A[];
};
layout(set = 0, binding = 1, std430) restrict readonly buffer WeightBuffer {
  float B[];
};
layout(set = 0, binding = 2, std430) restrict readonly buffer BiasBuffer {
  float C[];
};
layout(set = 0, binding = 3, std430) restrict writeonly buffer OutputBuffer {
  float out_data[];
};

layout(push_constant) uniform PushConstants {
  uint M;
  uint N;
  uint K;
  float alpha;
  float beta;
};

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
  uint m = gl_GlobalInvocationID.x;
  uint n = gl_GlobalInvocationID.y;

  if (m >= M || n >= N)
    return;

  float sum = 0.0;
  for (uint k = 0; k < K; k++) {
    sum += A[m * K + k] * B[n * K + k];
  }

  out_data[m * N + n] = alpha * sum + beta * C[n];
}