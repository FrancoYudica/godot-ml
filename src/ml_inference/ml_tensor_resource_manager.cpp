#include "ml_tensor_resource_manager.hpp"

namespace ml {

    void TensorResourceManager::init(RenderingDevice* rendering_device) {
        _rd = rendering_device;
    }

    void TensorResourceManager::clear() {
        for (auto& [name, tensor] : _tensors_data) {
            if (tensor.storage_buffer.is_valid()) {
                _rd->free_rid(tensor.storage_buffer);
            }
        }
        _tensors_data.clear();
    }

    RID TensorResourceManager::get_or_create(const std::string& name,
                                             const std::vector<int64_t>& shape,
                                             const std::vector<float>& data,
                                             uint32_t owner) {
        PackedByteArray bytes;
        bytes.resize(data.size() * sizeof(float));
        memcpy(bytes.ptrw(), data.data(), data.size() * sizeof(float));
        return get_or_create(name, shape, bytes, owner);
    }

    RID TensorResourceManager::get_or_create(const std::string& name,
                                             const std::vector<int64_t>& shape,
                                             const PackedByteArray& data,
                                             uint32_t owner) {
        const std::string& key = _make_key(name, owner);

        // The resource exists and is owned by the specified owner
        if (_exists(key, owner)) {
            _update_gpu_buffer(key, data, shape);
            return _tensors_data[key].storage_buffer;
        }

        // The specified owner doesn't own the resource, but it's shared
        if (_exists(name, 0)) {
            // If it's a shared resource, then it can't be changed by
            // the specified owner
            if (data.size() > 0) {
                ERR_PRINT(
                    "Attempting to update a shared tensor with new data.");
                return RID();
            }

            // Just gets, it's fine
            return _tensors_data[name].storage_buffer;
        }

        // If it doesn't exist, initialize the metadata
        if (_tensors_data.find(key) == _tensors_data.end()) {
            _TensorBuffer new_tensor;
            new_tensor.shape = shape;  // This makes a DEEP COPY of the vector
            new_tensor.buffer_size = 0;
            new_tensor.storage_buffer = RID();
            new_tensor.owner = owner;
            _tensors_data[key] = new_tensor;
        }

        // Now that we definitely have an entry, ensure the GPU buffer matches
        _update_gpu_buffer(key, data, shape);
        return _tensors_data[key].storage_buffer;
    }

    PackedByteArray TensorResourceManager::get_buffer(const std::string& name,
                                                      uint32_t owner) {
        RID sb = get_buffer_rid(name, owner);
        if (sb == RID()) return PackedByteArray();
        return _rd->buffer_get_data(sb);
    }

    RID TensorResourceManager::get_buffer_rid(const std::string& name,
                                              uint32_t owner) {
        const std::string& key = _make_key(name, owner);

        if (_tensors_data.find(key) !=
            _tensors_data.end()) {  // Use the key directly here
            return _tensors_data[key].storage_buffer;
        }

        if (_tensors_data.find(name) != _tensors_data.end()) {  // owner 0
            return _tensors_data[name].storage_buffer;
        }
        ERR_PRINT("Tensor named \"" + String(name.c_str()) + "\" not found.");
        return RID();
    }

    const std::vector<int64_t> TensorResourceManager::get_tensor_shape(
        const std::string& name, uint32_t owner) {
        const std::string& key = _make_key(name, owner);

        if (_exists(name, owner)) {
            return _tensors_data[key].shape;
        }

        if (_exists(name, 0)) {
            return _tensors_data[name].shape;
        }

        return std::vector<int64_t>();
    }

    void TensorResourceManager::free_owned_by(uint32_t owner) {
        for (auto it = _tensors_data.begin(); it != _tensors_data.end(); ++it) {
            if (it->second.owner == owner) {
                if (it->second.storage_buffer.is_valid()) {
                    _rd->free_rid(it->second.storage_buffer);
                }
                _tensors_data.erase(it);
            }
        }
    }

    void TensorResourceManager::_update_gpu_buffer(
        const std::string& key,
        const PackedByteArray& data,
        const std::vector<int64_t>& shape) {
        _TensorBuffer& tensor = _tensors_data[key];

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

    void TensorResourceManager::_create_tensor_storage(
        const std::string& name, const std::vector<int64_t>& shape) {
        if (_tensors_data.find(name) == _tensors_data.end()) {
            UtilityFunctions::print(
                "Tensor named \"" + String(name.c_str()) +
                "\" does not exist. Cannot create storage.");
            return;
        }

        _TensorBuffer& tensor = _tensors_data[name];

        if (tensor.storage_buffer == RID()) {
            // Calculate the total size of the tensor
            uint32_t total_size = 1;
            for (int64_t dim : shape) {
                total_size *= static_cast<uint32_t>(dim);
            }

            // Create the storage buffer
            tensor.buffer_size = total_size * sizeof(float);
            tensor.storage_buffer = _rd->storage_buffer_create(
                tensor.buffer_size, PackedByteArray());
        }

        else {
            ERR_PRINT("Tensor named \"" + String(name.c_str()) +
                      "\" already exists.");
        }
    }

    std::string TensorResourceManager::_make_key(const std::string& name,
                                                 uint32_t owner) {
        if (owner == 0) {
            return name;
        }

        return name + "_" + std::to_string(owner);
    }

    bool TensorResourceManager::_exists(const std::string& name,
                                        uint32_t owner) {
        return _tensors_data.find(_make_key(name, owner)) !=
               _tensors_data.end();
    }

}  // namespace ml