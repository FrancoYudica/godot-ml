#pragma once
#include <unordered_map>
#include <memory>
#include "ml_io_descriptor.hpp"
#include "ml_output_handler.hpp"
#include "ml_types.hpp"
#include "ml_utils.hpp"

namespace ml {

    class OutputHandlerRegistry {
    public:
        bool init(godot::RenderingDevice* rd);

        const std::unique_ptr<ml::IOutputHandler>& get(
            const ml::OutputType& desc) const;
        void destroy(godot::RenderingDevice* rd);

    private:
        template <typename T>
        bool _register(ml::OutputType type, godot::RenderingDevice* rd) {
            auto impl = std::make_unique<T>();
            if (impl->init(rd)) {
                _handlers[type] = std::move(impl);
                return true;
            }
            return false;
        }
        std::unordered_map<ml::OutputType, std::unique_ptr<ml::IOutputHandler>>
            _handlers;
    };

}  // namespace ml