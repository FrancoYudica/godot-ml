#[compute]
#version 450

#include "common.glsl.inc"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) writeonly uniform image2D target_image;

layout(set = 0, binding = 1) buffer InputTensor { float data[]; }
input_tensor;

layout(push_constant) uniform PushConstants {
  uint width;
  uint height;
  uint channel_count;
}
pc;

void main() {
  uvec3 id = gl_GlobalInvocationID;

  if (id.x >= pc.width || id.y >= pc.height) {
    return;
  }

  ivec2 texel_coord = ivec2(id.xy);

  vec4 out_color = vec4(0.0, 0.0, 0.0, 1.0);

  if (pc.channel_count >= 1)
    out_color.r =
        input_tensor.data[chw_index(pc.height, pc.width, 0u, id.y, id.x)];
  if (pc.channel_count >= 2)
    out_color.g =
        input_tensor.data[chw_index(pc.height, pc.width, 1u, id.y, id.x)];
  if (pc.channel_count >= 3)
    out_color.b =
        input_tensor.data[chw_index(pc.height, pc.width, 2u, id.y, id.x)];
  if (pc.channel_count >= 4)
    out_color.a =
        input_tensor.data[chw_index(pc.height, pc.width, 3u, id.y, id.x)];

  imageStore(target_image, texel_coord, out_color);
}