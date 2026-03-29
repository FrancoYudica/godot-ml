#pragma once
#include <unordered_map>
#include <memory>
#include "core/core.hpp"
#include "ml_input_handler.hpp"

namespace ml {

    class InputHandlerRegistry {
    public:
        bool init(godot::RenderingDevice* rd);

        const std::unique_ptr<ml::IInputHandler>& get(
            const ml::InputType& desc) const;
        void destroy(godot::RenderingDevice* rd);

    private:
        template <typename T>
        bool _register(ml::InputType type, godot::RenderingDevice* rd) {
            auto impl = std::make_unique<T>();
            if (impl->init(rd)) {
                _handlers[type] = std::move(impl);
                return true;
            }
            return false;
        }
        std::unordered_map<ml::InputType, std::unique_ptr<ml::IInputHandler>>
            _handlers;
    };

}  // namespace ml