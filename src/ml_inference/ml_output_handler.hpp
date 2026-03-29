#pragma once
#include "ml_tensor_resource_manager.hpp"
#include "ml_io_descriptor.hpp"

namespace ml {

    class IOutputHandler {
    public:
        virtual ~IOutputHandler() = default;

        /**
         * Called after inference completes. Reads or transfers the result
         * from the activation manager to whatever the user requested.
         */
        virtual void download(godot::RenderingDevice* rd,
                              Ref<TensorResourceManager> activations_tm) = 0;

        virtual void destroy(RenderingDevice* rd) {
        }

        /**
         * Returns the result of the output handler. This could be a texture,
         * cpu buffer, storage buffer, whatever
         */
        godot::Variant get() {
            return _output;
        }

    protected:
        godot::Variant _output;
    };

}  // namespace ml