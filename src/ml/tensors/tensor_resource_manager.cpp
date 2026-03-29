#include "tensor_resource_manager.hpp"

namespace ml {

    void TensorResourceManager::init(RenderingDevice* rendering_device) {
        _rd = rendering_device;
    }

    void TensorResourceManager::destroy() {
        for (auto& [name, tensor] : _tensors_data) {
            if (tensor.storage_buffer.is_valid()) {
                _rd->free_rid(tensor.storage_buffer);
            }
        }
        _tensors_data.clear();
    }

    RID TensorResourceManager::get_or_create(const std::string& name,
                                             const std::vector<int64_t>& shape,
                                             const std::vector<float>& data) {
        PackedByteArray bytes;
        bytes.resize(data.size() * sizeof(float));
        memcpy(bytes.ptrw(), data.data(), data.size() * sizeof(float));
        return get_or_create(name, shape, bytes);
    }

    RID TensorResourceManager::get_or_create(const std::string& name,
                                             const std::vector<int64_t>& shape,
                                             const PackedByteArray& data) {
        // If it doesn't exist, initialize the metadata
        if (_tensors_data.find(name) == _tensors_data.end()) {
            _TensorBuffer new_tensor;
            new_tensor.shape = shape;  // This makes a DEEP COPY of the vector
            new_tensor.buffer_size = 0;
            new_tensor.storage_buffer = RID();
            _tensors_data[name] = new_tensor;
        }

        // Now that we definitely have an entry, ensure the GPU buffer matches
        _update_gpu_buffer(name, data, shape);

        return _tensors_data[name].storage_buffer;
    }

    RID TensorResourceManager::get_buffer_rid(const std::string& name) {
        if (_tensors_data.find(name) == _tensors_data.end()) {
            return RID();
        }

        return _tensors_data[name].storage_buffer;
    }

    PackedByteArray TensorResourceManager::get_buffer(const std::string& name) {
        RID sb = get_buffer_rid(name);

        if (!sb.is_valid()) {
            return PackedByteArray();
        }

        return _rd->buffer_get_data(sb);
    }

    const std::vector<int64_t> TensorResourceManager::get_tensor_shape(
        const std::string& name) {
        if (_tensors_data.find(name) != _tensors_data.end()) {
            return _tensors_data[name].shape;
        }

        return std::vector<int64_t>();
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
            uint32_t elements = 1;
            for (int64_t d : shape) elements *= (uint32_t)d;
            required_size = elements * sizeof(float);
        }

        // Reallocate only if size grows or RID is null
        if (tensor.storage_buffer == RID() ||
            tensor.buffer_size < required_size) {
            if (tensor.storage_buffer.is_valid()) {
                _rd->free_rid(tensor.storage_buffer);
            }
            tensor.buffer_size = required_size;
            tensor.storage_buffer =
                _rd->storage_buffer_create(tensor.buffer_size, data);
        }
        // Reuse existing buffer if it's big enough
        else if (data.size() > 0) {
            _rd->buffer_update(tensor.storage_buffer, 0, data.size(), data);
        }
    }
}  // namespace ml