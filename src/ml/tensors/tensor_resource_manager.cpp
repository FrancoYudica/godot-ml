#include "tensor_resource_manager.hpp"

#include "core/core.hpp"

using namespace godot;

namespace ml {

void TensorResourceManager::init(
    RenderingDevice* rendering_device,
    StorageBufferPool* buffer_pool) {
    _rd = rendering_device;
    _pool = buffer_pool;
}

OperationResult TensorResourceManager::create_alias(
    const std::string& source_name,
    const std::string& alias_name,
    const std::vector<int64_t>& alias_shape) {

    // Source must exist and must not itself be an alias
    if (_tensors_data.find(source_name) == _tensors_data.end()) {
        return OperationResult{
            false,
            "create_alias: source '" + source_name + "' not found"};
    }
    if (_aliases.find(source_name) != _aliases.end()) {
        return OperationResult{
            false,
            "create_alias: source '" + source_name + "' is itself an alias - chaining not supported"};
    }

    // Validate size compatibility
    uint32_t source_size = Utils::get_tensor_floats(_tensors_data[source_name].shape);
    uint32_t alias_size = Utils::get_tensor_floats(alias_shape);

    if (source_size != alias_size) {
        return {
            false,
            "create_alias: size mismatch between source (" +
                std::to_string(source_size) + ") and alias (" +
                std::to_string(alias_size) + ")"};
    }

    _aliases[alias_name] = _TensorAlias{source_name, alias_shape};
    return {true, {}};
}

void TensorResourceManager::remove_alias(const std::string& alias_name) {
    _aliases.erase(alias_name);
}

bool TensorResourceManager::is_alias(const std::string& name) const {
    return _aliases.find(name) != _aliases.end();
}

void TensorResourceManager::destroy() {

    _aliases.clear();

    // Releases all used buffers
    for (auto& [name, tensor] : _tensors_data) {
        _pool->release(tensor.storage_buffer);
    }

    _tensors_data.clear();
}

RID TensorResourceManager::get_or_create(
    const std::string& name,
    const std::vector<int64_t>& shape,
    const std::vector<float>& data) {

    PackedByteArray bytes;
    bytes.resize(data.size() * sizeof(float));
    memcpy(bytes.ptrw(), data.data(), data.size() * sizeof(float));
    return get_or_create(name, shape, bytes);
}

RID TensorResourceManager::get_or_create(
    const std::string& name,
    const std::vector<int64_t>& shape,
    const PackedByteArray& data) {

    // If it doesn't exist, initialize the metadata
    if (!has(name)) {
        _TensorBuffer new_tensor;
        new_tensor.shape = shape; // This makes a DEEP COPY of the vector
        new_tensor.buffer_size = 0;
        new_tensor.storage_buffer = StorageBufferPool::INVALID_HANDLE;
        _tensors_data[name] = new_tensor;
    }

    // Now that we definitely have an entry, ensure the GPU buffer matches
    _update_gpu_buffer(name, data, shape);

    return _pool->get_rid(_tensors_data[name].storage_buffer);
}

RID TensorResourceManager::get_buffer_rid(const std::string& name) {

    if (!has(name)) return RID();

    _TensorBuffer* buffer = _resolve(name);

    if (buffer == nullptr) return RID();

    return _pool->get_rid(buffer->storage_buffer);
}

PackedByteArray TensorResourceManager::get_buffer(const std::string& name) {
    RID sb = get_buffer_rid(name);

    if (!sb.is_valid()) {
        UtilityFunctions::printerr(String("Unable to get buffer data for tensor: ") + name.c_str());
        return PackedByteArray();
    }

    return _rd->buffer_get_data(sb);
}

OperationResult TensorResourceManager::reshape(
    const std::string& name,
    const std::vector<int64_t>& new_shape) {

    if (!has(name)) {
        return {false, "Tensor named: " + name + " not found"};
    }

    // Reshape is only allowed when their sizes are compatible
    uint32_t actual_size = Utils::get_tensor_floats(_tensors_data[name].shape);
    uint32_t required_size = Utils::get_tensor_floats(new_shape);

    if (actual_size != required_size) {
        return {
            false,
            "Incompatible tensor sizes: " +
                std::to_string(actual_size) +
                " != " +
                std::to_string(required_size)};
    }

    _tensors_data[name].shape = new_shape;
    return {true, {}};
}

bool TensorResourceManager::has(const std::string& name) {
    return _tensors_data.find(name) != _tensors_data.end() ||
           _aliases.find(name) != _aliases.end();
}

const std::vector<int64_t> TensorResourceManager::get_tensor_shape(
    const std::string& name) {

    // Alias shape takes priority - it may differ from source
    auto alias_it = _aliases.find(name);
    if (alias_it != _aliases.end())
        return alias_it->second.shape;

    auto data_it = _tensors_data.find(name);
    if (data_it != _tensors_data.end())
        return data_it->second.shape;

    return {};
}

void TensorResourceManager::_update_gpu_buffer(
    const std::string& name,
    const PackedByteArray& data,
    const std::vector<int64_t>& shape) {
    _TensorBuffer& tensor = _tensors_data[name];

    // Calculate required size based on shape if data is empty
    uint32_t required_size = data.size();

    // Initialization
    if (required_size == 0) {
        required_size = Utils::get_tensor_floats(shape) * sizeof(float);
    }

    // Reallocate only if size grows or RID is null
    if (tensor.storage_buffer == StorageBufferPool::INVALID_HANDLE ||
        tensor.buffer_size < required_size) {

        // It no longer fits, makes releases buffer
        if (tensor.storage_buffer != StorageBufferPool::INVALID_HANDLE) {
            _pool->release(tensor.storage_buffer);
        }

        tensor.buffer_size = required_size;
        tensor.storage_buffer = _pool->get(tensor.buffer_size, data);
    }
    // Reuse existing buffer if it's big enough
    else if (data.size() > 0) {

        RID buffer_rid = _pool->get_rid(tensor.storage_buffer);
        _rd->buffer_update(buffer_rid, 0, data.size(), data);
    }
}

_TensorBuffer* TensorResourceManager::_resolve(
    const std::string& name) {

    // Check aliases first
    auto alias_it = _aliases.find(name);
    if (alias_it != _aliases.end()) {
        auto data_it = _tensors_data.find(alias_it->second.source_name);
        if (data_it != _tensors_data.end())
            return &data_it->second;
        return nullptr;
    }

    auto data_it = _tensors_data.find(name);
    if (data_it != _tensors_data.end())
        return &data_it->second;

    return nullptr;
}

} // namespace ml