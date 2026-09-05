#[compute]
#version 450

layout(set = 0, binding = 0, std430) restrict readonly buffer InputBuffer {
  float in_data[];
};

layout(set = 0, binding = 1, std430) restrict writeonly buffer OutputBuffer {
  float out_data[];
};

layout(push_constant) uniform PushConstants { uint total; };

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
  uint gid = gl_GlobalInvocationID.x;
  out_data[gid] = 1.0 / (1.0 + exp(-in_data[gid]));
}