#pragma once
#include "io/input_handler.hpp"
#include "io/io_descriptor.hpp"

namespace ml {
    class TextureInputHandler : public IInputHandler {
    public:
        std::vector<int64_t> upload(
            const std::unique_ptr<InputDesc::BaseData>& desc,
            const InputHandlerContext& ctx) override;

        void dispatch(const InputHandlerContext& ctx) override;
    };

}  // namespace ml