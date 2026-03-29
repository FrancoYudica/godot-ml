#pragma once
#include "io/input_handler.hpp"
#include "io/io_descriptor.hpp"

namespace ml {
    class FloatArrayInputHandler : public IInputHandler {
    public:
        std::vector<int64_t> upload(
            const std::unique_ptr<InputDesc::BaseData>& desc,
            godot::RenderingDevice* rd,
            Ref<TensorResourceManager> activations_tm) override;
    };

}  // namespace ml