#pragma once
#include "../ml_input_handler.hpp"
#include "../ml_io_descriptor.hpp"

namespace ml {
    class FloatArrayInputHandler : public IInputHandler {
    public:
        FloatArrayInputHandler(InputDesc::FloatArray desc)
            : _desc(std::move(desc)) {
        }

        std::vector<int64_t> upload(
            const std::string& tensor_name,
            godot::RenderingDevice* rd,
            Ref<TensorResourceManager> activations_tm) override;

    private:
        InputDesc::FloatArray _desc;
    };

}  // namespace ml