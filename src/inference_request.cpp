#include "inference_request.hpp"

namespace godot {
    void InferenceRequest::_bind_methods() {
        ClassDB::bind_method(
            D_METHOD("add_float_array_input", "tensor_name", "data", "shape"),
            &InferenceRequest::add_float_array_input);
        ClassDB::bind_method(
            D_METHOD("add_texture_input", "tensor_name", "texture"),
            &InferenceRequest::add_texture_input);
        ClassDB::bind_method(
            D_METHOD("add_float_array_output", "tensor_name", "output_name"),
            &InferenceRequest::add_float_array_output);
    }

    bool InferenceRequest::has_inputs_defined(
        const std::vector<std::string>& input_names) const {
        bool has_all_defined = true;

        for (const std::string& name : input_names) {
            if (_defined_input_tensors.find(name) ==
                _defined_input_tensors.end()) {
                has_all_defined = false;

                ERR_PRINT("Input for tensor '" + String(name.c_str()) +
                          "' is missing.");
            }
        }
        return has_all_defined;
    }

    void InferenceRequest::add_float_array_input(
        const String& tensor_name,
        const PackedFloat32Array& data,
        const PackedFloat64Array& shape) {
        ERR_FAIL_COND_MSG(
            _defined_input_tensors.find(tensor_name.utf8().get_data()) !=
                _defined_input_tensors.end(),
            "Input for tensor '" + tensor_name + "' already exists.");

        std::vector<int64_t> vector_shape(shape.ptr(),
                                          shape.ptr() + shape.size());
        float_array_inputs.push_back(
            {tensor_name.utf8().get_data(), data, vector_shape});

        _defined_input_tensors.insert(tensor_name.utf8().get_data());
    }

    void InferenceRequest::add_texture_input(const String& tensor_name,
                                             Ref<Texture2D> texture) {
        ERR_FAIL_COND_MSG(
            _defined_input_tensors.find(tensor_name.utf8().get_data()) !=
                _defined_input_tensors.end(),
            "Input for tensor '" + tensor_name + "' already exists.");

        texture_inputs.push_back({tensor_name.utf8().get_data(), texture});
        _defined_input_tensors.insert(tensor_name.utf8().get_data());
    }

    void InferenceRequest::add_float_array_output(const String& tensor_name,
                                                  const String& output_name) {
        ERR_FAIL_COND_MSG(
            float_array_outputs.find(output_name.utf8().get_data()) !=
                float_array_outputs.end(),
            "User already defined an output named '" + output_name + "'.");

        float_array_outputs[output_name.utf8().get_data()] = {
            tensor_name.utf8().get_data()};
    }

}  // namespace godot