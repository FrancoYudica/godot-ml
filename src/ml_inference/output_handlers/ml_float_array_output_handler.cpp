#include "ml_float_array_output_handler.hpp"

void ml::FloatArrayOutputHandler::download(
    godot::RenderingDevice* rd, Ref<TensorResourceManager> activations_tm) {
    PackedByteArray bytes = activations_tm->get_buffer(_desc.tensor_name);
    PackedFloat32Array out;
    out.resize(bytes.size() / sizeof(float));
    memcpy(out.ptrw(), bytes.ptr(), bytes.size());
    _output = out;
}