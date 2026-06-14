#[compute]
#version 450

#include "common.glsl.inc"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Flattened Tensors
layout(set = 0, binding = 0) buffer Input { float data[]; }
input_tensor;
layout(set = 0, binding = 1) buffer Weights { float data[]; }
weights;
layout(set = 0, binding = 2) buffer Biases { float data[]; }
biases;
layout(set = 0, binding = 3) buffer Output { float data[]; }
output_tensor;

layout(push_constant) uniform PushConstants {
  uint batch_size;
  uint in_channels;
  uint in_height;
  uint in_width;
  uint out_channels;
  uint out_height;
  uint out_width;
  uint kernel_h;
  uint kernel_w;
  uint padding_top;
  uint padding_left;
  uint stride_y;
  uint stride_x;
}
pc;

void main() {
  // These IDs represent the coordinates in the OUTPUT tensor
  uint out_x = gl_GlobalInvocationID.x;
  uint out_y = gl_GlobalInvocationID.y;
  uint batch_idx = gl_GlobalInvocationID.z;

  if (out_x >= pc.out_width || out_y >= pc.out_height ||
      batch_idx >= pc.batch_size)
    return;

  // Batch offset: same total elements per batch regardless of layout
  uint image_size = pc.in_channels * pc.in_width * pc.in_height;
  uint batch_offset = batch_idx * image_size;

  // Map output coordinate to the top-left of the input window
  uint start_x = out_x * pc.stride_x;
  uint start_y = out_y * pc.stride_y;

  for (uint oc = 0; oc < pc.out_channels; oc++) {
    float sum = 0.0;

    for (uint ky = 0; ky < pc.kernel_h; ky++) {
      for (uint kx = 0; kx < pc.kernel_w; kx++) {

        // Sample relative to the window start
        int sample_x = int(start_x) + int(kx) - int(pc.padding_left);
        int sample_y = int(start_y) + int(ky) - int(pc.padding_top);

        if (sample_x >= 0 && sample_x < int(pc.in_width) && sample_y >= 0 &&
            sample_y < int(pc.in_height)) {

          for (uint ic = 0; ic < pc.in_channels; ic++) {
            uint input_idx =
                batch_offset + chw_index(pc.in_height, pc.in_width, ic,
                                         uint(sample_y), uint(sample_x));
            // Weights are OIHW: oc*(IC*KH*KW) + ic*(KH*KW) + ky*KW + kx
            uint weight_idx =
                oc * (pc.in_channels * pc.kernel_w * pc.kernel_h) +
                ic * (pc.kernel_w * pc.kernel_h) + ky * pc.kernel_w + kx;

            sum += input_tensor.data[input_idx] * weights.data[weight_idx];
          }
        }
      }
    }

    uint output_idx =
        batch_idx * (pc.out_channels * pc.out_height * pc.out_width) +
        chw_index(pc.out_height, pc.out_width, oc, out_y, out_x);
    output_tensor.data[output_idx] = sum + biases.data[oc];
  }
}