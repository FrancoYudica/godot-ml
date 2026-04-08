#include <godot_cpp/classes/rendering_server.hpp>
#include "inference_descriptor.hpp"

namespace godot {
    void InferenceDescriptor::_bind_methods() {
        ClassDB::bind_method(
            D_METHOD("add_float_array_input", "tensor_name", "data", "shape"),
            &InferenceDescriptor::add_float_array_input);
        ClassDB::bind_method(
            D_METHOD("add_texture_input", "tensor_name", "texture"),
            &InferenceDescriptor::add_texture_input);
        ClassDB::bind_method(
            D_METHOD("add_float_array_output", "tensor_name", "output_name"),
            &InferenceDescriptor::add_float_array_output);
        ClassDB::bind_method(
            D_METHOD("add_texture_output", "tensor_name", "texture"),
            &InferenceDescriptor::add_texture_output);
    }

    bool InferenceDescriptor::has_inputs_defined(
        const std::vector<std::string>& input_names) const {
        bool has_all_defined = true;

        for (const std::string& name : input_names) {
            if (inputs.find(name) == inputs.end()) {
                has_all_defined = false;

                ERR_PRINT("Input for tensor '" + String(name.c_str()) +
                          "' is missing.");
            }
        }
        return has_all_defined;
    }

    void InferenceDescriptor::add_float_array_input(
        const String& tensor_name,
        const PackedFloat32Array& data,
        const PackedFloat64Array& shape) {
        ERR_FAIL_COND_MSG(
            inputs.find(tensor_name.utf8().get_data()) != inputs.end(),
            "Input for tensor '" + tensor_name + "' already exists.");

        std::vector<int64_t> vector_shape(shape.ptr(),
                                          shape.ptr() + shape.size());

        auto desc = std::make_unique<ml::InputDesc::FloatArray>();
        desc->type = ml::InputType::FloatArray;
        desc->tensor_name = tensor_name.utf8().get_data();
        desc->data = data;
        desc->shape = vector_shape;
        inputs[tensor_name.utf8().get_data()] = std::move(desc);
    }

    void InferenceDescriptor::add_texture_input(const String& tensor_name,
                                                Ref<Texture2D> texture) {
        ERR_FAIL_COND_MSG(
            inputs.find(tensor_name.utf8().get_data()) != inputs.end(),
            "Input for tensor '" + tensor_name + "' already exists.");

        auto desc = std::make_unique<ml::InputDesc::Texture>();
        desc->type = ml::InputType::Texture2D;
        desc->tensor_name = tensor_name.utf8().get_data();
        desc->channels = 3;
        desc->texture = texture;
        inputs[tensor_name.utf8().get_data()] = std::move(desc);
    }

    void InferenceDescriptor::add_float_array_output(
        const String& tensor_name, const String& output_name) {
        ERR_FAIL_COND_MSG(
            outputs.find(output_name.utf8().get_data()) != outputs.end(),
            "User already defined an output named '" + output_name + "'.");

        auto desc = std::make_unique<ml::OutputDesc::FloatArray>();
        desc->type = ml::OutputType::FloatArray;
        desc->tensor_name = tensor_name.utf8().get_data();
        outputs[output_name.utf8().get_data()] = std::move(desc);
    }

    void InferenceDescriptor::add_texture_output(const String& tensor_name,
                                                 Ref<Texture2D> texture) {
        ERR_FAIL_COND_MSG(
            outputs.find(tensor_name.utf8().get_data()) != outputs.end(),
            "User already defined an output named '" + tensor_name + "'.");

        auto desc = std::make_unique<ml::OutputDesc::Texture>();
        desc->type = ml::OutputType::Texture2D;
        desc->tensor_name = tensor_name.utf8().get_data();
        desc->target_texture =
            RenderingServer::get_singleton()->texture_get_rd_texture(
                texture->get_rid());
        outputs[tensor_name.utf8().get_data()] = std::move(desc);
    }

}  // namespace godot