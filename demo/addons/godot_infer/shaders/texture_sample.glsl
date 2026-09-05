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
  uint load_mode;
}
pc;

// Must match LoadTextureMode enum in inference_descriptor.hpp
#define MODE_RGB       0u
#define MODE_RGBA      1u
#define MODE_RED       2u
#define MODE_GREEN     3u
#define MODE_BLUE      4u
#define MODE_ALPHA     5u
#define MODE_GRAYSCALE 6u

void main() {
  uvec3 id = gl_GlobalInvocationID;

  if (id.x >= pc.width || id.y >= pc.height) {
    return;
  }

  vec2 uv = (vec2(id.xy) + vec2(0.5)) / vec2(pc.width, pc.height);
  vec4 color = texture(input_texture, uv);

  if (pc.load_mode == MODE_RGB) {
    output_tensor.data[chw_index(pc.height, pc.width, 0u, id.y, id.x)] = color.r;
    output_tensor.data[chw_index(pc.height, pc.width, 1u, id.y, id.x)] = color.g;
    output_tensor.data[chw_index(pc.height, pc.width, 2u, id.y, id.x)] = color.b;
  } else if (pc.load_mode == MODE_RGBA) {
    output_tensor.data[chw_index(pc.height, pc.width, 0u, id.y, id.x)] = color.r;
    output_tensor.data[chw_index(pc.height, pc.width, 1u, id.y, id.x)] = color.g;
    output_tensor.data[chw_index(pc.height, pc.width, 2u, id.y, id.x)] = color.b;
    output_tensor.data[chw_index(pc.height, pc.width, 3u, id.y, id.x)] = color.a;
  } else if (pc.load_mode == MODE_RED) {
    output_tensor.data[chw_index(pc.height, pc.width, 0u, id.y, id.x)] = color.r;
  } else if (pc.load_mode == MODE_GREEN) {
    output_tensor.data[chw_index(pc.height, pc.width, 0u, id.y, id.x)] = color.g;
  } else if (pc.load_mode == MODE_BLUE) {
    output_tensor.data[chw_index(pc.height, pc.width, 0u, id.y, id.x)] = color.b;
  } else if (pc.load_mode == MODE_ALPHA) {
    output_tensor.data[chw_index(pc.height, pc.width, 0u, id.y, id.x)] = color.a;
  } else {
    // GRAYSCALE: BT.601 luminance
    float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    output_tensor.data[chw_index(pc.height, pc.width, 0u, id.y, id.x)] = gray;
  }
}
