#include "operator.hpp"

namespace ml {
class Im2ColOperator : public IOperator {
  public:
    virtual bool init(godot::RenderingDevice* rd) override;
    virtual void dispatch(
        const ml::PhysicalNode& node,
        const OperatorContext& ctx) override;
    void destroy(godot::RenderingDevice* rd) override;

  private:
    struct PushConstants {
        uint32_t in_width;
        uint32_t in_height;
        uint32_t in_channels;
        uint32_t kernel_size;
        uint32_t pad;
        uint32_t stride_x;
        uint32_t stride_y;
        uint32_t out_width;
        uint32_t out_height;
        float padding[3];
    };

    godot::RID _shader;
    godot::RID _pipeline;
};

} // namespace ml