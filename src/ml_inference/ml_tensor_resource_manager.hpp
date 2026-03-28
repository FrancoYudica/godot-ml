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

        /**
         * Owner set to 0 means that it's shared, usually for weights
         * defined by the model. For non 0, it's owned by a specific task
         */
        uint32_t owner = 0;
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
                          const std::vector<float>& data,
                          uint32_t owner);

        RID get_or_create(const std::string& name,
                          const std::vector<int64_t>& shape = {},
                          const PackedByteArray& data = {},
                          uint32_t owner = 0);

        PackedByteArray get_buffer(const std::string& name, uint32_t owner);
        RID get_buffer_rid(const std::string& name, uint32_t owner);

        const std::vector<int64_t> get_tensor_shape(const std::string& name,
                                                    uint32_t owner);
        void free_owned_by(uint32_t owner);

    private:
        void _update_gpu_buffer(const std::string& name,
                                const PackedByteArray& data,
                                const std::vector<int64_t>& shape);

        void _create_tensor_storage(const std::string& name,
                                    const std::vector<int64_t>& shape);

        std::string _make_key(const std::string& name, uint32_t owner);

        bool _exists(const std::string& name, uint32_t owner);

    private:
        RenderingDevice* _rd;
        std::unordered_map<std::string, _TensorBuffer> _tensors_data;
    };
}  // namespace ml