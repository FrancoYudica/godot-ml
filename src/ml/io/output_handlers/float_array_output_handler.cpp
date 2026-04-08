#include "float_array_output_handler.hpp"

using namespace godot;

namespace ml {

    Variant FloatArrayOutputHandler::download(
        const std::unique_ptr<OutputDesc::BaseData>& desc,
        RenderingDevice* rd,
        Ref<TensorResourceManager> activations_tm) {
        PackedByteArray bytes = activations_tm->get_buffer(desc->tensor_name);
        PackedFloat32Array out;
        out.resize(bytes.size() / sizeof(float));
        memcpy(out.ptrw(), bytes.ptr(), bytes.size());
        return out;
    }

}  // namespace ml