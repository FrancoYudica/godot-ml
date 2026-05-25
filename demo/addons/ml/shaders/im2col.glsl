#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Input { float data[]; }
input_tensor;
layout(set = 0, binding = 1) buffer Output { float data[]; }
output_tensor;

layout(push_constant) uniform PushConstants {
  uint in_width;
  uint in_height;
  uint in_channels;
  uint kernel_w;
  uint kernel_h;
  uint padding_left;
  uint padding_top;
  uint stride_x;
  uint stride_y;
  uint out_width;
  uint out_height;
  uint dilation_x;
  uint dilation_y;
}
pc;

void main() {
  // Each thread writes one element of the col matrix
  // col shape: [out_h * out_w, in_c * kh * kw]
  uint total =
      pc.out_height * pc.out_width * pc.in_channels * pc.kernel_w * pc.kernel_h;
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= total)
    return;

  // Decompose flat index back into (patch_row, col_elem)
  uint patch_size = pc.in_channels * pc.kernel_w *
                    pc.kernel_h;     // amount of elements per patch/frame
  uint elem_idx = idx % patch_size;  // which input element within the patch
  uint patch_idx = idx / patch_size; // which output element

  // Maps patch idx to output (x, y) coordinates
  uint out_x = patch_idx % pc.out_width;
  uint out_y = patch_idx / pc.out_width;

  // Decompose elem_idx into (ic, ky, kx)
  uint kx = elem_idx % pc.kernel_w;
  uint ky = (elem_idx / pc.kernel_w) % pc.kernel_h;
  uint ic = elem_idx / (pc.kernel_w * pc.kernel_h);

  int sample_x =
      int(out_x * pc.stride_x) + int(kx * pc.dilation_x) - int(pc.padding_left);
  int sample_y =
      int(out_y * pc.stride_y) + int(ky * pc.dilation_y) - int(pc.padding_top);

  float val = 0.0; // zero padding

  // If inside the image
  if (sample_x >= 0 && sample_x < int(pc.in_width) && sample_y >= 0 &&
      sample_y < int(pc.in_height)) {

    // Gets element index within tensor data
    uint input_idx =
        (uint(sample_y) * pc.in_width + uint(sample_x)) * pc.in_channels + ic;
    val = input_tensor.data[input_idx];
  }

  output_tensor.data[idx] = val;
}