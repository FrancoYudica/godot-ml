#pragma once
#include <memory>
#include <godot_cpp/classes/rendering_device.hpp>
#include "tensors/tensors.hpp"
#include "io_descriptor.hpp"

namespace ml {

    struct InputHandlerContext {
        godot::RenderingDevice* rd;
        godot::Ref<TensorResourceManager> activations_tm;
        int64_t compute_list;
    };

    class IInputHandler {
    public:
        virtual ~IInputHandler() = default;
        virtual bool init(godot::RenderingDevice* rd) {
            return true;
        }

        virtual void destroy(RenderingDevice* rd) {
        }
        /**
         * Converts input data to a GPU storage buffer and registers it
         * in the activation manager under the given tensor name.
         * Returns the shape of the uploaded tensor.
         */
        virtual std::vector<int64_t> upload(
            const std::unique_ptr<InputDesc::BaseData>& desc,
            const InputHandlerContext& ctx) = 0;

        virtual void dispatch(const InputHandlerContext& ctx) {
        }
    };

}  // namespace ml