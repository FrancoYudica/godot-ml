#pragma once
#include "../ml_output_handler.hpp"

namespace ml {

    class FloatArrayOutputHandler : public IOutputHandler {
    public:
        FloatArrayOutputHandler(OutputDesc::FloatArray desc)
            : _desc(std::move(desc)) {
        }

        void download(godot::RenderingDevice* rd,
                      Ref<TensorResourceManager> activations_tm) override;

    private:
        OutputDesc::FloatArray _desc;
    };

}  // namespace ml