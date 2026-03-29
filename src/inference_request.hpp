#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <godot_cpp/classes/ref_counted.hpp>
#include "ml_inference/ml_io_descriptor.hpp"

namespace godot {

    class InferenceRequest : public RefCounted {
        GDCLASS(InferenceRequest, RefCounted);

    protected:
        static void _bind_methods();

    public:
        bool has_inputs_defined(
            const std::vector<std::string>& input_names) const;

        void add_float_array_input(const String& tensor_name,
                                   const PackedFloat32Array& data,
                                   const PackedFloat64Array& shape);

        void add_texture_input(const String& tensor_name,
                               Ref<Texture2D> texture);

        void add_float_array_output(const String& tensor_name,
                                    const String& output_name);

    public:
        std::vector<ml::InputDesc::FloatArray> float_array_inputs;
        std::vector<ml::InputDesc::Texture> texture_inputs;
        std::unordered_map<std::string, ml::OutputDesc::FloatArray>
            float_array_outputs;

    private:
        std::unordered_set<std::string> _defined_input_tensors;
    };

}  // namespace godot