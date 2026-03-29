#pragma once
#include "../ml_input_handler.hpp"
#include "../ml_io_descriptor.hpp"

namespace ml {
    class TextureInputHandler : public IInputHandler {
    public:
        TextureInputHandler(InputDesc::Texture desc) : _desc(std::move(desc)) {
        }

        std::vector<int64_t> upload(
            godot::RenderingDevice* rd,
            Ref<TensorResourceManager> activations_tm) override;

    private:
        InputDesc::Texture _desc;
    };

}  // namespace ml