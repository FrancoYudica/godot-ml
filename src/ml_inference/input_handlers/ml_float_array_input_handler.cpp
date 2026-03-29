#include "ml_float_array_input_handler.hpp"

std::vector<int64_t> ml::FloatArrayInputHandler::upload(
    godot::RenderingDevice* rd, Ref<TensorResourceManager> activations_tm) {
    activations_tm->get_or_create(_desc.tensor_name, _desc.shape,
                                  _desc.data.to_byte_array());

    return _desc.shape;
}