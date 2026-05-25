#[compute]
#version 450

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

  uint batch = bc / pc.channels;
  uint channel = bc % pc.channels;

  // Initialize max_val to a very small number
  float max_val = -3.402823466e+38;

  uint start_x = out_x * pc.stride_x;
  uint start_y = out_y * pc.stride_y;

  uint in_image_size = pc.in_width * pc.in_height;
  uint batch_channel_offset = (batch * pc.channels + channel) * in_image_size;

  // Sliding Window
  for (uint ky = 0; ky < pc.kernel_h; ky++) {
    for (uint kx = 0; kx < pc.kernel_w; kx++) {
      int sample_x =
          int(start_x) + int(kx) * int(pc.dilation_x) - int(pc.padding_left);
      int sample_y =
          int(start_y) + int(ky) * int(pc.dilation_y) - int(pc.padding_top);

      if (sample_x >= 0 && sample_x < int(pc.in_width) && sample_y >= 0 &&
          sample_y < int(pc.in_height)) {

        uint input_idx = batch_channel_offset +
                         (uint(sample_y) * pc.in_width + uint(sample_x));
        max_val = max(max_val, input_tensor.data[input_idx]);
      }
    }
  }

  // Write result
  uint out_image_size = pc.out_width * pc.out_height;
  uint output_idx = (batch * pc.channels + channel) * out_image_size +
                    (out_y * pc.out_width + out_x);

  output_tensor.data[output_idx] = max_val;
}