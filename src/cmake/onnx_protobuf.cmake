# Generates C++ sources for onnx's protobuf messages.
#
# We deliberately do NOT add_subdirectory(thirdparty/onnx) and consume its own
# CMakeLists.txt: that builds onnx's entire C++ library (defs, checker, shape
# inference, optimizer...), requires a Python interpreter, and pins C++11.
# We only use onnx's protobuf *messages* (godot_infer/passes/parser.cpp does
# our own graph parsing), so this mirrors thirdparty/SCsub instead: invoke
# protoc directly on the two .proto files we actually need.
function(onnx_generate_protobuf_sources OUT_SOURCES)
    set(_onnx_dir "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnx/onnx")

    if(TARGET protobuf::protoc)
        set(_protoc protobuf::protoc)
    else()
        set(_protoc "${Protobuf_PROTOC_EXECUTABLE}")
    endif()

    set(_generated_sources "")
    foreach(_proto_name onnx-ml onnx-data)
        set(_proto_file "${_onnx_dir}/${_proto_name}.proto")
        set(_generated_cc "${_onnx_dir}/${_proto_name}.pb.cc")
        set(_generated_h "${_onnx_dir}/${_proto_name}.pb.h")

        add_custom_command(
            OUTPUT "${_generated_cc}" "${_generated_h}"
            COMMAND
                ${_protoc}
                "--cpp_out=${_onnx_dir}"
                "--proto_path=${_onnx_dir}"
                "--proto_path=${_onnx_dir}/.."
                "${_proto_file}"
            DEPENDS "${_proto_file}" ${_protoc}
            COMMENT "Generating protobuf sources for ${_proto_name}.proto"
            VERBATIM
        )
        list(APPEND _generated_sources "${_generated_cc}")
    endforeach()

    set(${OUT_SOURCES} "${_generated_sources}" PARENT_SCOPE)
endfunction()
