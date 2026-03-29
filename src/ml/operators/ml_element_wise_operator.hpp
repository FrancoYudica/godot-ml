// operators/ml_op_elementwise.hpp
#pragma once
#include "ml_operator.hpp"

namespace ml {

    // Base for any activation that operates element-wise with no extra params
    class ElementwiseOperator : public IOperator {
    public:
        bool init(godot::RenderingDevice* rd) override;
        void dispatch(const GraphNode& node,
                      const OperatorContext& ctx) override;
        void destroy(godot::RenderingDevice* rd) override;

    protected:
        // Subclasses provide the shader path
        virtual const char* shader_path() const = 0;

    private:
        struct PushConstants {
            uint32_t M;
            uint32_t K;
            float padding[2];  // Alignment to 16 bytes
        };

        godot::RID _shader;
        godot::RID _pipeline;
    };

    class ReLUOperator : public ElementwiseOperator {
        const char* shader_path() const override {
            return "shaders/relu.glsl";
        }
    };

    class SigmoidOperator : public ElementwiseOperator {
        const char* shader_path() const override {
            return "shaders/sigmoid.glsl";
        }
    };

}  // namespace ml