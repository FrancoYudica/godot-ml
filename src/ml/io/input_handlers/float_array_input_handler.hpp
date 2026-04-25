#pragma once
#include "io/input_handler.hpp"
#include "io/io_descriptor.hpp"

namespace ml {
class FloatArrayInputHandler : public IInputHandler {
  public:
    bool upload(
        const std::unique_ptr<InputDesc::BaseData>& desc,
        const InputHandlerContext& ctx) override;

    std::vector<int64_t> get_shape(const std::unique_ptr<InputDesc::BaseData>& desc) const override;
};

} // namespace ml