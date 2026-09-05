#include "onnx_resource_loader.hpp"
#include "onnx_resource.hpp"

namespace godot {

PackedStringArray ONNXResourceLoader::_get_recognized_extensions() const {
    PackedStringArray extensions;
    extensions.push_back("onnx");
    return extensions;
}

bool ONNXResourceLoader::_handles_type(const StringName& type) const {
    return type == StringName("ONNXResource");
}

String ONNXResourceLoader::_get_resource_type(const String& path) const {
    if (path.get_extension().to_lower() == "onnx")
        return "ONNXResource";
    return "";
}

Variant ONNXResourceLoader::_load(
    const String& path,
    const String& original_path,
    bool use_sub_threads,
    int32_t cache_mode) const {
    return ONNXResource::load_from_file(path);
}

} // namespace godot
