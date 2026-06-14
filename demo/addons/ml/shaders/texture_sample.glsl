#[compute]
#version 450

#include "common.glsl.inc"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D input_texture;

layout(set = 0, binding = 1) buffer OutputTensor { float data[]; }
output_tensor;

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

  // Sample the texture (normalized coordinates for sampler2D)
  vec2 uv = (vec2(id.xy) + vec2(0.5)) / vec2(pc.width, pc.height);
  vec4 color = texture(input_texture, uv);

  output_tensor.data[chw_index(pc.height, pc.width, 0u, id.y, id.x)] = color.r;
  if (pc.channel_count > 1)
    output_tensor.data[chw_index(pc.height, pc.width, 1u, id.y, id.x)] =
        color.g;
  if (pc.channel_count > 2)
    output_tensor.data[chw_index(pc.height, pc.width, 2u, id.y, id.x)] =
        color.b;
  if (pc.channel_count > 3)
    output_tensor.data[chw_index(pc.height, pc.width, 3u, id.y, id.x)] =
        color.a;
}