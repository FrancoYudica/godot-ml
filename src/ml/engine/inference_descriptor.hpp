#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <godot_cpp/classes/ref_counted.hpp>
#include "io/io.hpp"

namespace godot {

    class InferenceDescriptor : public RefCounted {
        GDCLASS(InferenceDescriptor, RefCounted);

    protected:
        static void _bind_methods();

    public:
        bool has_inputs_defined(
            const std::vector<std::string>& input_names) const;

        void add_float_array_input(const String& tensor_name,
                                   const PackedFloat32Array& data,
                                   const PackedFloat64Array& shape);

        void add_texture_input(const String& tensor_name,
                               Ref<Texture2D> texture,
                               uint32_t process_width_override = 0,
                               uint32_t process_height_override = 0);

        void add_float_array_output(const String& tensor_name,
                                    const String& output_name);

        void add_texture_output(const String& tensor_name,
                                Ref<Texture2D> texture);

    public:
        /**
         * Maps the tensor name to the input descriptor. This is done
         * to easily make sure that the user isn't defining more than one
         * input handler per tensor.
         */
        std::unordered_map<std::string,
                           std::unique_ptr<ml::InputDesc::BaseData>>
            inputs;

        /**
         * Maps the output name to the output descriptor. Note that the output
         * name is user defined, since a single tensor could have many outputs.
         * For example, the case where the user wants to get the result as a
         * texture but also into a float array.
         */
        std::unordered_map<std::string,
                           std::unique_ptr<ml::OutputDesc::BaseData>>
            outputs;
    };

}  // namespace godot