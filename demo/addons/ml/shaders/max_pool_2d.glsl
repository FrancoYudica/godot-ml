#[compute]
#version 450

#include "common.glsl.inc"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Input { float data[]; }
input_tensor;
layout(set = 0, binding = 1) buffer Output { float data[]; }
output_tensor;

layout(push_constant) uniform PushConstants {
  uint batch_size;
  uint channels;
  uint in_height;
  uint in_width;
  uint out_height;
  uint out_width;
  uint kernel_h;
  uint kernel_w;
  uint padding_top;
  uint padding_left;
  uint stride_y;
  uint stride_x;
  uint dilation_y;
  uint dilation_x;
}
pc;

void main() {
  uint out_x = gl_GlobalInvocationID.x;
  uint out_y = gl_GlobalInvocationID.y;
  uint bc = gl_GlobalInvocationID.z; // Batch * Channels

  if (out_x >= pc.out_width || out_y >= pc.out_height ||
      bc >= (pc.batch_size * pc.channels))
    return;

  float max_val = -3.402823466e+38;

  uint start_x = out_x * pc.stride_x;
  uint start_y = out_y * pc.stride_y;

  for (uint ky = 0; ky < pc.kernel_h; ky++) {
    for (uint kx = 0; kx < pc.kernel_w; kx++) {
      int sample_x =
          int(start_x) + int(kx) * int(pc.dilation_x) - int(pc.padding_left);
      int sample_y =
          int(start_y) + int(ky) * int(pc.dilation_y) - int(pc.padding_top);

      if (sample_x >= 0 && sample_x < int(pc.in_width) && sample_y >= 0 &&
          sample_y < int(pc.in_height)) {
        uint input_idx = chw_index(pc.in_height, pc.in_width, bc,
                                   uint(sample_y), uint(sample_x));
        max_val = max(max_val, input_tensor.data[input_idx]);
      }
    }
  }

  output_tensor.data[chw_index(pc.out_height, pc.out_width, bc, out_y, out_x)] =
      max_val;
}