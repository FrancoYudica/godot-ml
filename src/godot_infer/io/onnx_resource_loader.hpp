#pragma once
#include <godot_cpp/classes/resource_format_loader.hpp>

namespace godot {

class ONNXResourceLoader : public ResourceFormatLoader {
    GDCLASS(ONNXResourceLoader, ResourceFormatLoader)

protected:
    static void _bind_methods() {}

public:
    PackedStringArray _get_recognized_extensions() const override;
    bool _handles_type(const StringName& type) const override;
    String _get_resource_type(const String& path) const override;
    Variant _load(
        const String& path,
        const String& original_path,
        bool use_sub_threads,
        int32_t cache_mode) const override;
};

} // namespace godot
