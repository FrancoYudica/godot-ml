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
  uint kernel_size;
  uint pad;
  uint stride_x;
  uint stride_y;
  uint out_width;
  uint out_height;
}
pc;

void main() {
  // Each thread writes one element of the col matrix
  // col shape: [out_h * out_w, in_c * kh * kw]
  uint total = pc.out_height * pc.out_width * pc.in_channels * pc.kernel_size *
               pc.kernel_size;
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= total)
    return;

  // Decompose flat index back into (patch_row, col_elem)
  uint patch_size = pc.in_channels * pc.kernel_size *
                    pc.kernel_size;  // amount of elements per patch/frame
  uint elem_idx = idx % patch_size;  // which input element within the patch
  uint patch_idx = idx / patch_size; // which output element

  // Maps patch idx to output (x, y) coordinates
  uint out_x = patch_idx % pc.out_width;
  uint out_y = patch_idx / pc.out_width;

  // Decompose elem_idx into (ic, ky, kx)
  uint kx = elem_idx % pc.kernel_size;
  uint ky = (elem_idx / pc.kernel_size) % pc.kernel_size;
  uint ic = elem_idx / (pc.kernel_size * pc.kernel_size);

  int sample_x = int(out_x * pc.stride_x) + int(kx) - int(pc.pad);
  int sample_y = int(out_y * pc.stride_y) + int(ky) - int(pc.pad);

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