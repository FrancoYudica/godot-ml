/**
 * Defines all the types of I/O descriptors for ML inference.
 * Note that these don't hold the actual data of the output,
 * it's instead a
 */

#pragma once
#include <vector>
#include <string>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/rendering_device.hpp>

namespace ml {

    enum class InputType {
        Texture2D,      // GPU texture -> storage buffer
        FloatArray,     // CPU float array -> storage buffer
        StorageBuffer,  // Already on GPU, just bind it directly
    };

    enum class OutputType {
        FloatArray,     // Read back to CPU
        Texture2D,      // Write directly to a texture RID
        StorageBuffer,  // Leave on GPU, return RID
    };

    namespace InputDesc {

        struct Texture {
            godot::Ref<godot::Texture2D> texture;
            uint32_t channels = 3;  // how many channels to extract
        };

        struct FloatArray {
            godot::PackedFloat32Array data;
            std::vector<int64_t> shape;  // explicit shape, e.g. {4096, 3}
        };

        struct StorageBuffer {
            godot::RID buffer;
            std::vector<int64_t> shape;
        };

    }  // namespace InputDesc

    namespace OutputDesc {
        struct FloatArray {
            // which graph output to read back
            std::string tensor_name;
        };

        struct Texture {
            std::string tensor_name;

            // write result directly into this texture
            godot::RID target_texture;
            uint32_t width;
            uint32_t height;
        };

        struct StorageBuffer {
            std::string tensor_name;
        };
    }  // namespace OutputDesc

}  // namespace ml