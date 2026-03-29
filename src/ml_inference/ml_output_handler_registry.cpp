#include "ml_output_handler_registry.hpp"
#include "output_handlers/ml_float_array_output_handler.hpp"

namespace ml {

    bool OutputHandlerRegistry::init(godot::RenderingDevice* rd) {
        // Register all supported operators
        ERR_FAIL_COND_V_MSG(
            !_register<FloatArrayOutputHandler>(OutputType::FloatArray, rd),
            false,
            "OutputHandlerRegistry: failed to register FloatArray operator.");

        return true;
    }

    const std::unique_ptr<ml::IOutputHandler>& OutputHandlerRegistry::get(
        const ml::OutputType& desc) const {
        return _handlers.at(desc);
    }

    void OutputHandlerRegistry::destroy(godot::RenderingDevice* rd) {
        for (auto& pair : _handlers) {
            pair.second->destroy(rd);
        }
    }

}  // namespace ml