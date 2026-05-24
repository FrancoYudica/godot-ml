#include "operator.hpp"

namespace ml {
class MaxPool2DOperator : public IOperator {
  public:
    virtual bool init(godot::RenderingDevice* rd) override;
    virtual void dispatch(
        const ml::Physical::Node& node,
        const OperatorContext& ctx) override;
    void destroy(godot::RenderingDevice* rd) override;

  private:
    struct PushConstants {
        uint32_t batch_size;
        uint32_t channels;
        uint32_t in_width;
        uint32_t in_height;
        uint32_t out_width;
        uint32_t out_height;
        uint32_t kernel_size;
        uint32_t stride_x;
        uint32_t stride_y;
        uint32_t pad_left;
        uint32_t pad_top;
        uint32_t dilation_x;
        uint32_t dilation_y;
        uint32_t _padding[3];
    };

    godot::RID _shader;
    godot::RID _pipeline;
};

} // namespace ml