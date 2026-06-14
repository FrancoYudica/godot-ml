#[compute]
#version 450

#include "common.glsl.inc"
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer ColBuffer {
  float col_data[]; // [out_H*out_W, in_C*kH*kW]  (gemm output)
};
layout(set = 0, binding = 1, std430) buffer OutputBuffer {
  float out_data[]; // [batch, out_C, out_H, out_W]
};

layout(push_constant) uniform PushConstants {
  uint in_height;
  uint in_width;
  uint out_channels;
  uint out_height;
  uint out_width;
  uint kernel_h;
  uint kernel_w;
  uint pad_top;
  uint pad_left;
  uint stride_y;
  uint stride_x;
  uint dilation_y;
  uint dilation_x;
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
  for (uint ky = 0; ky < pc.kernel_h; ky++) {
    for (uint kx = 0; kx < pc.kernel_w; kx++) {
      // Reverse map
      int in_x_raw =
          int(out_x) + int(pc.pad_left) - int(kx) * int(pc.dilation_x);
      int in_y_raw =
          int(out_y) + int(pc.pad_top) - int(ky) * int(pc.dilation_y);

      // Must be divisible by stride
      if (in_x_raw % int(pc.stride_x) != 0)
        continue;
      if (in_y_raw % int(pc.stride_y) != 0)
        continue;

      int in_x = in_x_raw / int(pc.stride_x);
      int in_y = in_y_raw / int(pc.stride_y);
      if (in_x < 0 || in_x >= int(pc.in_width) || in_y < 0 ||
          in_y >= int(pc.in_height))
        continue;

      uint patch_idx = uint(in_y) * pc.in_width + uint(in_x);
      uint elem_idx = oc * (pc.kernel_h * pc.kernel_w) + ky * pc.kernel_w + kx;

      uint col_idx =
          patch_idx * (pc.out_channels * pc.kernel_h * pc.kernel_w) + elem_idx;

      sum += col_data[col_idx];
    }
  }

  uint out_idx = chw_index(pc.out_height, pc.out_width, oc, out_y, out_x);
  out_data[out_idx] = sum;
}