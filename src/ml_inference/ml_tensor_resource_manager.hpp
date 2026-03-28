#pragma once
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <string>
#include <vector>
#include <unordered_map>

using namespace godot;

namespace ml {

    struct _TensorBuffer {
        std::vector<int64_t> shape;
        uint32_t buffer_size = 0;
        RID storage_buffer = RID();
    };

    class TensorResourceManager : public RefCounted {
    public:
        void init(RenderingDevice* rendering_device);
        void clear();
        /**
         * Gets an existing storage buffer or creates a new one if it doesn't
         * exist, based on it's shape
         */
        RID get_or_create(const std::string& name,
                          const std::vector<int64_t>& shape,
                          const std::vector<float>& data);

        RID get_or_create(const std::string& name,
                          const std::vector<int64_t>& shape = {},
                          const PackedByteArray& data = {});

        RID get_buffer_rid(const std::string& name);

        PackedByteArray get_buffer(const std::string& name);

        const std::vector<int64_t> get_tensor_shape(const std::string& name);

    private:
        void _update_gpu_buffer(const std::string& name,
                                const PackedByteArray& data,
                                const std::vector<int64_t>& shape);

    private:
        RenderingDevice* _rd;
        std::unordered_map<std::string, _TensorBuffer> _tensors_data;
    };
}  // namespace ml