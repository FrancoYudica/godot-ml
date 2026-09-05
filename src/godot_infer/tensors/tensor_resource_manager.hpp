#pragma once
#include "core/core.hpp"
#include "storage_buffer_pool.hpp"

#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/rendering_device.hpp>

using namespace godot;

namespace ml {

struct _TensorBuffer {
    std::vector<int64_t> shape;
    uint32_t buffer_size = 0;
    StorageBufferHandle storage_buffer;
};

struct _TensorAlias {
    std::string source_name;    // points to _tensors_data entry
    std::vector<int64_t> shape; // may differ from source
};

class TensorResourceManager : public RefCounted {
  public:
    void init(RenderingDevice* rendering_device, StorageBufferPool* buffer_pool);
    void destroy();
    /**
     * Gets an existing storage buffer or creates a new one if it doesn't
     * exist, based on it's shape
     */
    RID get_or_create(
        const std::string& name,
        const std::vector<int64_t>& shape,
        const std::vector<float>& data);

    RID get_or_create(
        const std::string& name,
        const std::vector<int64_t>& shape = {},
        const PackedByteArray& data = {});

    RID get_buffer_rid(const std::string& name);

    PackedByteArray get_buffer(const std::string& name);

    OperationResult reshape(const std::string& name, const std::vector<int64_t>& new_shape);

    bool has(const std::string& name);

    const std::vector<int64_t> get_tensor_shape(const std::string& name);

    // Creates an alias: 'alias_name' points to the same GPU buffer as 'source_name'
    // but can have a different shape. No data is copied.
    OperationResult create_alias(
        const std::string& source_name,
        const std::string& alias_name,
        const std::vector<int64_t>& alias_shape);

    // Removes an alias without releasing the underlying buffer
    void remove_alias(const std::string& alias_name);

    bool is_alias(const std::string& name) const;

  private:
    void _update_gpu_buffer(
        const std::string& name,
        const PackedByteArray& data,
        const std::vector<int64_t>& shape);
    // Resolves a name through aliases to the real _TensorBuffer
    _TensorBuffer* _resolve(const std::string& name);
    const _TensorBuffer* _resolve(const std::string& name) const;

  private:
    RenderingDevice* _rd;
    StorageBufferPool* _pool;
    std::unordered_map<std::string, _TensorBuffer> _tensors_data;
    std::unordered_map<std::string, _TensorAlias> _aliases;
};
} // namespace ml