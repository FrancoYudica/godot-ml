#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer ColBuffer {
  float col_data[]; // [out_H*out_W, in_C*kH*kW]  (gemm output)
};
layout(set = 0, binding = 1, std430) buffer OutputBuffer {
  float out_data[]; // [batch, out_C, out_H, out_W]
};

layout(push_constant) uniform PushConstants {
  uint in_width;   // low-res width  (col columns)
  uint in_height;  // low-res height (col rows)
  uint out_width;  // high-res width
  uint out_height; // high-res height
  uint out_channels;
  uint kernel_size;
  uint pad;
  uint stride;
}
pc;

void main() {
  uint out_x = gl_GlobalInvocationID.x;
  uint out_y = gl_GlobalInvocationID.y;
  uint oc = gl_GlobalInvocationID.z;

  if (out_x >= pc.out_width || out_y >= pc.out_height || oc >= pc.out_channels)
    return;

  float sum = 0.0;

  // Which input patches contribute to this output pixel?
  for (uint ky = 0; ky < pc.kernel_size; ky++) {
    for (uint kx = 0; kx < pc.kernel_size; kx++) {
      // Reverse map: which in_x, in_y produced this out_x, out_y via this
      // kernel pos?
      int in_x_raw = int(out_x) + int(pc.pad) - int(kx);
      int in_y_raw = int(out_y) + int(pc.pad) - int(ky);

      // Must be divisible by stride
      if (in_x_raw % int(pc.stride) != 0)
        continue;
      if (in_y_raw % int(pc.stride) != 0)
        continue;

      int in_x = in_x_raw / int(pc.stride);
      int in_y = in_y_raw / int(pc.stride);

      if (in_x < 0 || in_x >= int(pc.in_width) || in_y < 0 ||
          in_y >= int(pc.in_height))
        continue;

      uint patch_idx = uint(in_y) * pc.in_width + uint(in_x);
      uint elem_idx =
          oc * (pc.kernel_size * pc.kernel_size) + ky * pc.kernel_size + kx;

      uint col_idx =
          patch_idx * (pc.out_channels * pc.kernel_size * pc.kernel_size) +
          elem_idx;

      sum += col_data[col_idx];
    }
  }

  uint out_idx = (out_y * pc.out_width + out_x) * pc.out_channels + oc;
  out_data[out_idx] = sum;
}