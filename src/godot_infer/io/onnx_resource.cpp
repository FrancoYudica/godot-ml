#include "onnx_resource.hpp"
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/core/class_db.hpp>

namespace godot {

void ONNXResource::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_data"), &ONNXResource::get_data);
    ClassDB::bind_method(D_METHOD("set_data", "data"), &ONNXResource::set_data);
    ADD_PROPERTY(
        PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data"),
        "set_data",
        "get_data");
    ClassDB::bind_static_method(
        "ONNXResource",
        D_METHOD("load_from_file", "path"),
        &ONNXResource::load_from_file);
}

Ref<ONNXResource> ONNXResource::load_from_file(const String& path) {
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
    ERR_FAIL_COND_V_MSG(
        file.is_null(),
        Ref<ONNXResource>(),
        "ONNXResource: cannot open file: " + path);

    Ref<ONNXResource> resource;
    resource.instantiate();
    resource->set_data(file->get_buffer(file->get_length()));
    return resource;
}

} // namespace godot
