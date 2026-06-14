#pragma once
#include "core/core.hpp"
#include "io/io.hpp"

namespace godot {

enum LoadTextureMode {
    TEXTURE_LOAD_RGB = 0,       // 3 channels → [R, G, B]
    TEXTURE_LOAD_RGBA = 1,      // 4 channels → [R, G, B, A]
    TEXTURE_LOAD_RED = 2,       // 1 channel  → [R]
    TEXTURE_LOAD_GREEN = 3,     // 1 channel  → [G]
    TEXTURE_LOAD_BLUE = 4,      // 1 channel  → [B]
    TEXTURE_LOAD_ALPHA = 5,     // 1 channel  → [A]
    TEXTURE_LOAD_GRAYSCALE = 6, // 1 channel  → [0.299R + 0.587G + 0.114B]
};

class InferenceDescriptor : public RefCounted {
    GDCLASS(InferenceDescriptor, RefCounted);

  protected:
    static void _bind_methods();

  public:
    bool has_inputs_defined(
        const std::vector<std::string>& input_names) const;

    void add_float_array_input(
        const String& tensor_name,
        const PackedFloat32Array& data,
        const PackedFloat64Array& shape);

    void add_texture_input(
        const String& tensor_name,
        Ref<Texture2D> texture,
        LoadTextureMode load_mode = LoadTextureMode::TEXTURE_LOAD_RGB,
        uint32_t process_width_override = 0,
        uint32_t process_height_override = 0);

    void add_float_array_output(
        const String& tensor_name,
        const String& output_name);

    void add_texture_output(const String& tensor_name, Ref<Texture2D> texture);

  public:
    /**
     * Maps the tensor name to the input descriptor. This is done
     * to easily make sure that the user isn't defining more than one
     * input handler per tensor.
     */
    std::unordered_map<std::string, std::unique_ptr<ml::InputDesc::BaseData>>
        inputs;

    /**
     * Maps the output name to the output descriptor. Note that the output
     * name is user defined, since a single tensor could have many outputs.
     * For example, the case where the user wants to get the result as a
     * texture but also into a float array.
     */
    std::unordered_map<std::string, std::unique_ptr<ml::OutputDesc::BaseData>>
        outputs;
};

} // namespace godot

VARIANT_ENUM_CAST(LoadTextureMode);