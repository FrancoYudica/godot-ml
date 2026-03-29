#include "operator.hpp"

namespace ml {
    class GemmOperator : public IOperator {
    public:
        virtual bool init(godot::RenderingDevice* rd) override;
        virtual void dispatch(const ml::GraphNode& node,
                              const OperatorContext& ctx) override;
        void destroy(godot::RenderingDevice* rd) override;

    private:
        struct PushConstants {
            uint32_t M;
            uint32_t N;
            uint32_t K;
            float alpha;
            float beta;
            float padding[3];  // Alignment to 32 bytes
        };

        godot::RID _shader;
        godot::RID _pipeline;
    };

}  // namespace ml