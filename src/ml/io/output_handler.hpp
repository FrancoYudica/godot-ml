#pragma once
#include <memory>
#include <godot_cpp/classes/rendering_device.hpp>
#include "io_descriptor.hpp"
#include "tensors/tensors.hpp"
#include "core/core.hpp"

namespace ml {

    struct OutputHandlerContext {
        godot::RenderingDevice* rd;
        godot::Ref<TensorResourceManager> activations_tm;
        int64_t compute_list;
        DeletionStack* frame_deletion_stack;
    };

    class IOutputHandler {
    public:
        virtual ~IOutputHandler() = default;

        virtual bool init(godot::RenderingDevice* rd) {
            return true;
        }

        virtual void destroy(RenderingDevice* rd) {
        }

        /**
         * Called after inference completes. Reads or transfers the result
         * from the activation manager to whatever the user requested.
         */
        virtual godot::Variant download(
            const std::unique_ptr<OutputDesc::BaseData>& desc,
            godot::RenderingDevice* rd,
            Ref<TensorResourceManager> activations_tm) = 0;

        virtual void dispatch(const OutputHandlerContext& ctx) {
        }
    };

}  // namespace ml