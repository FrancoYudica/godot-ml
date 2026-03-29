#pragma once
#include <godot_cpp/classes/rendering_device.hpp>
#include "ml_tensor_resource_manager.hpp"
#include "ml_io_descriptor.hpp"

namespace ml {

    class IInputHandler {
    public:
        virtual ~IInputHandler() = default;

        /**
         * Converts input data to a GPU storage buffer and registers it
         * in the activation manager under the given tensor name.
         * Returns the shape of the uploaded tensor.
         */
        virtual std::vector<int64_t> upload(
            const std::string& tensor_name,
            godot::RenderingDevice* rd,
            Ref<TensorResourceManager> activations_tm) = 0;
    };

}  // namespace ml