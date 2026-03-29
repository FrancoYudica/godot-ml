#pragma once
#include <functional>
#include <stack>

namespace ml {
    class DeletionStack {
    public:
        void push(std::function<void()> func) {
            _stack.push(func);
        }

        void process() {
            while (!_stack.empty()) {
                _stack.top()();
                _stack.pop();
            }
        }

    private:
        std::stack<std::function<void()>> _stack;
    };
}  // namespace ml