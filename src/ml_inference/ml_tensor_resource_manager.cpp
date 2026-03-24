#include "ml_tensor_resource_manager.hpp"

namespace ml {

    void TensorResourceManager::init(RenderingDevice* rendering_device) {
        _rd = rendering_device;
    }

    RID TensorResourceManager::get_or_create(const std::string& name,
                                             const std::vector<int64_t>& shape,
                                             const std::vector<float>& data) {
        // Copies the floats to a PackedFloat32Array
        PackedFloat32Array packed_floats;
        packed_floats.resize(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            packed_floats[i] = data[i];
        }

        PackedByteArray bytes = packed_floats.to_byte_array();

        return get_or_create(name, shape, bytes);
    }

    RID TensorResourceManager::get_or_create(const std::string& name,
                                             const std::vector<int64_t>& shape,
                                             const PackedByteArray& data) {
        // Already loaded
        if (_tensors_data.find(name) != _tensors_data.end()) {
            if (data.size() > 0) {
                _write_tensor_buffer(name, data);
            }

            return _tensors_data[name].storage_buffer;
        }

        if (shape.empty()) {
            ERR_PRINT(
                "Shape is empty. Cannot create storage buffer for tensor \"" +
                String(name.c_str()) + "\".");
            return RID();
        }

        // Creates a new entry
        _TensorBuffer tensor;
        tensor.shape = shape;
        _tensors_data[name] = tensor;

        _create_tensor_storage(name, shape);

        //  Write gpu data
        if (data.size() > 0) {
            _write_tensor_buffer(name, data);
        }

        return tensor.storage_buffer;
    }

    const std::vector<int64_t> TensorResourceManager::get_tensor_shape(
        const std::string& name) {
        if (_tensors_data.find(name) != _tensors_data.end()) {
            return _tensors_data[name].shape;
        }

        return std::vector<int64_t>();
    }

    void TensorResourceManager::_write_tensor_buffer(
        const std::string& name, const PackedByteArray& data) {
        if (_tensors_data.find(name) == _tensors_data.end()) {
            UtilityFunctions::print("Tensor named \"" + String(name.c_str()) +
                                    "\" does not exist. Cannot write data.");
            return;
        }

        if (data.size() == 0) {
            ERR_PRINT("Data is empty. Cannot write to tensor \"" +
                      String(name.c_str()) + "\".");
            return;
        }

        _TensorBuffer& tensor = _tensors_data[name];

        // Create the storage buffer if null or if the size is different
        if (tensor.storage_buffer == RID() ||
            tensor.buffer_size <= data.size()) {
            // Free previous storage buffer
            if (tensor.storage_buffer != RID()) {
                _rd->free_rid(tensor.storage_buffer);
            }

            tensor.buffer_size = data.size();
            tensor.storage_buffer =
                _rd->storage_buffer_create(tensor.buffer_size, data);
        } else {
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

}  // namespace ml