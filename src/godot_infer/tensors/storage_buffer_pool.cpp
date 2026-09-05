#include "storage_buffer_pool.hpp"

namespace ml {
void StorageBufferPool::init(godot::RenderingDevice* p_rd) {
    _rd = p_rd;
}

void StorageBufferPool::destroy() {
    for (auto const& [handle, buffer_data] : _buffers) {
        if (buffer_data.rid.is_valid())
            _rd->free_rid(buffer_data.rid);
    }
    _pool.clear();
    _buffers.clear();
}

void StorageBufferPool::release(StorageBufferHandle p_buffer) {

    auto it = _buffers.find(p_buffer);
    if (it == _buffers.end()) {
        return;
    }

    auto& buffer_data = it->second;

    if (buffer_data.available) {
        return;
    }

    buffer_data.available = true;
    _pool.insert({buffer_data.size, p_buffer});
}

StorageBufferHandle StorageBufferPool::get(uint32_t p_size, godot::PackedByteArray p_data) {
    auto it = _pool.lower_bound(p_size);

    // We found a reusable buffer
    if (it != _pool.end()) {
        StorageBufferHandle handle = it->second;
        _pool.erase(it);
        _buffers[handle].available = false;
        return handle;
    }

    // Create a new buffer if the pool is empty
    StorageBufferHandle new_handle = _next_buffer_handle++;
    _StorageBufferData data;
    data.size = p_size;
    data.available = false; // Immediately used
    data.rid = _rd->storage_buffer_create(p_size, p_data);
    _buffers[new_handle] = std::move(data);

    godot::UtilityFunctions::print(godot::String("Creating buffer of size: ") + std::to_string(p_size).c_str());

    return new_handle;
}

godot::RID StorageBufferPool::get_rid(StorageBufferHandle p_buffer) {
    auto it = _buffers.find(p_buffer);
    if (it == _buffers.end()) {
        return godot::RID();
    }
    return it->second.rid;
}

} // namespace ml