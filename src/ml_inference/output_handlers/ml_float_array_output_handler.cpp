#include "ml_float_array_output_handler.hpp"

godot::Variant ml::FloatArrayOutputHandler::download(
    const std::unique_ptr<OutputDesc::BaseData>& desc,
    godot::RenderingDevice* rd,
    Ref<TensorResourceManager> activations_tm) {
    PackedByteArray bytes = activations_tm->get_buffer(desc->tensor_name);
    PackedFloat32Array out;
    out.resize(bytes.size() / sizeof(float));
    memcpy(out.ptrw(), bytes.ptr(), bytes.size());
    return out;
}