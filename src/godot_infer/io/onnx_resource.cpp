#include "onnx_resource.hpp"
#include <godot_cpp/core/class_db.hpp>

namespace godot {

void ONNXResource::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_data"), &ONNXResource::get_data);
    ClassDB::bind_method(D_METHOD("set_data", "data"), &ONNXResource::set_data);
    ADD_PROPERTY(
        PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data"),
        "set_data",
        "get_data");
}

} // namespace godot
