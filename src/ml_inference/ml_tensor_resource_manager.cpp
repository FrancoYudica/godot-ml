#include "ml_tensor_resource_manager.hpp"

namespace ml {

    void TensorResourceManager::init(RenderingDevice* rendering_device) {
        _rd = rendering_device;
    }

    RID TensorResourceManager::get_or_create(const std::string& name,
                                             const std::vector<int64_t>& shape,
                                             const std::vector<float>& data) {
        // Already loaded
        if (_tensors_data.find(name) != _tensors_data.end()) {
            if (!data.empty()) {
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

        //  Write gpu data
        if (!data.empty()) {
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
        const std::string& name, const std::vector<float>& data) {
        if (_tensors_data.find(name) == _tensors_data.end()) {
            UtilityFunctions::print("Tensor named \"" + String(name.c_str()) +
                                    "\" does not exist. Cannot write data.");
            return;
        }

        if (data.empty()) {
            ERR_PRINT("Data is empty. Cannot write to tensor \"" +
                      String(name.c_str()) + "\".");
            return;
        }

        // Copies the floats to a PackedFloat32Array
        PackedFloat32Array packed_floats;
        packed_floats.resize(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            packed_floats[i] = data[i];
        }

        PackedByteArray bytes = packed_floats.to_byte_array();

        _TensorBuffer& tensor = _tensors_data[name];

        // Create the storage buffer if null or if the size is different
        if (tensor.storage_buffer == RID() ||
            tensor.buffer_size <= bytes.size()) {
            tensor.buffer_size = bytes.size();
            tensor.storage_buffer =
                _rd->storage_buffer_create(tensor.buffer_size, bytes);
        } else {
            _rd->buffer_update(tensor.storage_buffer, 0, bytes.size(), bytes);
        }
    }

}  // namespace ml