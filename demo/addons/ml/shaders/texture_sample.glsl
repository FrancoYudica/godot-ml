#[compute]
#version 450

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

  // Calculate base index in the 1D buffer: (y * width + x) * channels
  uint base_index = (id.y * pc.width + id.x) * pc.channel_count;

  // Fill the tensor buffer based on requested channels
  output_tensor.data[base_index + 0] = color.r;
  if (pc.channel_count > 1)
    output_tensor.data[base_index + 1] = color.g;
  if (pc.channel_count > 2)
    output_tensor.data[base_index + 2] = color.b;
  if (pc.channel_count > 3)
    output_tensor.data[base_index + 3] = color.a;
}