#pragma once
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>

namespace godot {

class ONNXResource : public Resource {
    GDCLASS(ONNXResource, Resource)

    PackedByteArray _data;

  protected:
    static void _bind_methods();

  public:
    PackedByteArray get_data() const { return _data; }
    void set_data(const PackedByteArray& data) { _data = data; }

    // Reads a raw .onnx file from disk into a new ONNXResource. Shared by
    // ONNXResourceLoader (runtime loads of files outside the res:// import
    // database) and the editor import plugin (onnx_import_plugin.gd), so both
    // paths parse the source file the same way.
    static Ref<ONNXResource> load_from_file(const String& path);
};

} // namespace godot
