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
};

} // namespace godot
