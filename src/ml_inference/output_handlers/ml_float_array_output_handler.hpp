#pragma once
#include "../ml_output_handler.hpp"

namespace ml {

    class FloatArrayOutputHandler : public IOutputHandler {
    public:
        godot::Variant download(
            const std::unique_ptr<OutputDesc::BaseData>& desc,
            godot::RenderingDevice* rd,
            Ref<TensorResourceManager> activations_tm) override;
    };

}  // namespace ml