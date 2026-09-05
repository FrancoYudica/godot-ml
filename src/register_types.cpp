#include "register_types.h"

#include "godot_infer/engine/engine.hpp"
#include "godot_infer/io/onnx_resource.hpp"
#include "godot_infer/io/onnx_resource_loader.hpp"

#include <gdextension_interface.h>
#include <godot_cpp/classes/resource_loader.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

static Ref<ONNXResourceLoader> _onnx_loader;

void initialize_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    GDREGISTER_RUNTIME_CLASS(MLInferenceEngine);
    GDREGISTER_RUNTIME_CLASS(InferenceTask);
    GDREGISTER_RUNTIME_CLASS(InferenceDescriptor);
    GDREGISTER_CLASS(ONNXResource);
    GDREGISTER_CLASS(ONNXResourceLoader);

    _onnx_loader.instantiate();
    ResourceLoader::get_singleton()->add_resource_format_loader(_onnx_loader);
}

void uninitialize_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    ResourceLoader::get_singleton()->remove_resource_format_loader(_onnx_loader);
    _onnx_loader.unref();
}

extern "C" {
// Initialization.
GDExtensionBool GDE_EXPORT library_init(
    GDExtensionInterfaceGetProcAddress p_get_proc_address,
    const GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization* r_initialization) {

    godot::GDExtensionBinding::InitObject init_obj(
        p_get_proc_address,
        p_library,
        r_initialization);

    init_obj.register_initializer(initialize_module);
    init_obj.register_terminator(uninitialize_module);
    init_obj.set_minimum_library_initialization_level(
        MODULE_INITIALIZATION_LEVEL_SCENE);

    return init_obj.init();
}
}