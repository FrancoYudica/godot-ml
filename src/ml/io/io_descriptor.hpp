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

        class BaseData {
        public:
            virtual ~BaseData() = default;
            InputType type;
            std::string tensor_name;
        };

        class Texture : public BaseData {
        public:
            godot::Ref<godot::Texture2D> texture;
            uint32_t channels = 3;  // how many channels to extract
            uint32_t process_width_override = 0;
            uint32_t process_height_override = 0;
        };

        class FloatArray : public BaseData {
        public:
            godot::PackedFloat32Array data;
            std::vector<int64_t> shape;
        };

        class StorageBuffer : public BaseData {
        public:
            godot::RID buffer;
            std::vector<int64_t> shape;
        };

    }  // namespace InputDesc

    namespace OutputDesc {

        class BaseData {
        public:
            virtual ~BaseData() = default;
            OutputType type;
            std::string tensor_name;
        };
        class FloatArray : public BaseData {};

        class Texture : public BaseData {
        public:
            godot::RID target_texture;
        };

        class StorageBuffer : public BaseData {};
    }  // namespace OutputDesc

}  // namespace ml