#include "operator_registry.hpp"

#include "operators/element_wise_operator.hpp"
#include "operators/gemm_operator.hpp"

namespace ml {
bool OperatorRegistry::init(godot::RenderingDevice* rd) {
    // Register all supported operators
    ERR_FAIL_COND_V_MSG(
        !_register<GemmOperator>(NodeOperator::Gemm, rd),
        false,
        "OperatorRegistry: failed to register Gemm operator.");
    ERR_FAIL_COND_V_MSG(
        !_register<ReLUOperator>(NodeOperator::ReLU, rd),
        false,
        "OperatorRegistry: failed to register ReLU operator.");
    ERR_FAIL_COND_V_MSG(
        !_register<SigmoidOperator>(NodeOperator::Sigmoid, rd),
        false,
        "OperatorRegistry: failed to register Sigmoid operator.");

    return true;
}

IOperator* OperatorRegistry::get(NodeOperator op) const {
    auto it = _operators.find(op);
    ERR_FAIL_COND_V_MSG(
        it == _operators.end(),
        nullptr,
        "OperatorRegistry: no operator registered for this op type.");
    return it->second.get();
}

bool OperatorRegistry::supports(NodeOperator op) const {
    return _operators.count(op) > 0;
}

void OperatorRegistry::destroy(godot::RenderingDevice* rd) {
    for (auto& [op, impl] : _operators) {
        impl->destroy(rd);
    }
    _operators.clear();
}

} // namespace ml