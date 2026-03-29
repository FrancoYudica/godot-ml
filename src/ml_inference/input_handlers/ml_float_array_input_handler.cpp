#include "ml_float_array_input_handler.hpp"

std::vector<int64_t> ml::FloatArrayInputHandler::upload(
    const std::string& tensor_name,
    godot::RenderingDevice* rd,
    Ref<TensorResourceManager> activations_tm) {
    activations_tm->get_or_create(tensor_name, _desc.shape,
                                  _desc.data.to_byte_array());

    return _desc.shape;
}