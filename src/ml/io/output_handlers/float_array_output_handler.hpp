#pragma once
#include "io/output_handler.hpp"

namespace ml {

    class FloatArrayOutputHandler : public IOutputHandler {
    public:
        godot::Variant download(
            const std::unique_ptr<OutputDesc::BaseData>& desc,
            godot::RenderingDevice* rd,
            Ref<TensorResourceManager> activations_tm) override;

        void dispatch(const OutputHandlerContext& ctx) override;
    };

}  // namespace ml