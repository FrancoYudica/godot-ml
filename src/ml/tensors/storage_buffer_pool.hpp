#pragma once
#include "core/core.hpp"

#include <godot_cpp/classes/rendering_device.hpp>
#include <map>

namespace ml {

typedef uint32_t StorageBufferHandle;

struct _StorageBufferData {
    godot::RID rid;
    uint32_t size;
    bool available;
};

class StorageBufferPool {
  public:
    constexpr static StorageBufferHandle INVALID_HANDLE = 0;

  public:
    void init(godot::RenderingDevice* p_rd);

    // Clears all RIDs from the GPU
    void destroy();

    // Returns a buffer to the pool for future reuse
    void release(StorageBufferHandle p_buffer);

    // Return buffer handle. This might will create a new storage buffer
    // if not available
    StorageBufferHandle get(uint32_t p_size, godot::PackedByteArray p_data = {});

    godot::RID get_rid(StorageBufferHandle p_buffer);

  private:
    godot::RenderingDevice* _rd;

    std::map<StorageBufferHandle, _StorageBufferData> _buffers;

    // Multi-map allows multiple buffers of the same size to be pooled
    std::multimap<uint32_t, StorageBufferHandle> _pool;

    StorageBufferHandle _next_buffer_handle = 1;
};

} // namespace ml