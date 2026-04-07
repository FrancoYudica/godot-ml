#include "input_handler_registry.hpp"
#include "input_handlers/float_array_input_handler.hpp"
#include "input_handlers/texture_input_handler.hpp"
#include "input_handlers/texture_input_handler_compute.hpp"

namespace ml {

    bool InputHandlerRegistry::init(godot::RenderingDevice* rd) {
        // Register all supported operators
        ERR_FAIL_COND_V_MSG(
            !_register<FloatArrayInputHandler>(InputType::FloatArray, rd),
            false,
            "InputHandlerRegistry: failed to register FloatArray operator.");
        ERR_FAIL_COND_V_MSG(
            !_register<TextureInputHandlerCompute>(InputType::Texture2D, rd),
            false,
            "InputHandlerRegistry: failed to register Texture operator.");

        return true;
    }

    const std::unique_ptr<ml::IInputHandler>& InputHandlerRegistry::get(
        const ml::InputType& desc) const {
        return _handlers.at(desc);
    }

    void InputHandlerRegistry::destroy(godot::RenderingDevice* rd) {
        for (auto& pair : _handlers) {
            pair.second->destroy(rd);
        }
    }

}  // namespace ml