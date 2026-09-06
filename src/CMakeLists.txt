cmake_minimum_required(VERSION 3.24) # LINK_GROUP:RESCAN (see the Linux link_libraries call below)

# vcpkg.json lives at the project root, one level up from this file (src/).
# These must be set before project() so vcpkg's toolchain file (passed via
# -DCMAKE_TOOLCHAIN_FILE=.../vcpkg.cmake) picks up the manifest correctly.
get_filename_component(GD_INFER_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
if(NOT DEFINED VCPKG_MANIFEST_DIR)
    set(VCPKG_MANIFEST_DIR "${GD_INFER_PROJECT_ROOT}" CACHE PATH "Directory containing vcpkg.json")
endif()
if(NOT DEFINED VCPKG_INSTALLED_DIR)
    set(VCPKG_INSTALLED_DIR "${GD_INFER_PROJECT_ROOT}/vcpkg_installed" CACHE PATH "vcpkg installed tree")
endif()

project(godot_infer LANGUAGES CXX)

# Phase 1 of the SCons -> CMake migration: Linux and Windows only (the
# platforms covered by CI today). macOS/iOS/Android output shaping
# (framework bundle, device+simulator static libs) is not implemented yet.
if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    message(FATAL_ERROR
        "This CMakeLists.txt currently only supports Windows and Linux. "
        "macOS/iOS/Android still build via src/SConstruct."
    )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED GODOTCPP_TARGET)
    set(GODOTCPP_TARGET "template_debug" CACHE STRING "template_debug, template_release, or editor")
endif()

if(WIN32 AND NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    # godot-cpp/cmake/linux.cmake and windows.cmake both declare an option
    # named GODOTCPP_USE_STATIC_CPP with different defaults (OFF vs ON), and
    # godotcpp_options() unconditionally registers every platform's options --
    # so linux.cmake's OFF silently wins the option() race even when
    # targeting Windows, leaving MSVC on the dynamic CRT (/MD) by default.
    # Set CMAKE_MSVC_RUNTIME_LIBRARY directly (godot-cpp's own "set(...CACHE
    # STRING...)" is then a no-op, since the entry already exists) so both
    # godot-cpp and our own target consistently use the static CRT, matching
    # vcpkg's x64-windows-static triplet: /MT for Release, /MTd for Debug.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "MSVC runtime library")
endif()

add_subdirectory(thirdparty/godot-cpp)

find_package(Protobuf CONFIG REQUIRED)

include(cmake/onnx_protobuf.cmake)
onnx_generate_protobuf_sources(GD_INFER_ONNX_SOURCES)

file(GLOB_RECURSE GD_INFER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/godot_infer/*.cpp")

add_library(godot_infer SHARED
    register_types.cpp
    ${GD_INFER_SOURCES}
    ${GD_INFER_ONNX_SOURCES}
)

target_compile_definitions(godot_infer PRIVATE ONNX_ML=0)

target_include_directories(godot_infer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/godot_infer"
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnx"
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnx/onnx"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # vcpkg's static protobuf/abseil libs have circular inter-dependencies
    # (abseil ships as dozens of small .a files). LINK_GROUP:RESCAN wraps the
    # group in the linker's equivalent of -Wl,--start-group/--end-group,
    # re-scanning until all symbols resolve regardless of link order --
    # mirrors the equivalent SConstruct workaround.
    target_link_libraries(godot_infer PRIVATE godot::cpp "$<LINK_GROUP:RESCAN,protobuf::libprotobuf>")
    target_link_options(godot_infer PRIVATE "LINKER:--no-undefined")
else()
    target_link_libraries(godot_infer PRIVATE godot::cpp protobuf::libprotobuf)
endif()

# Match godot-cpp's own suffix scheme so the .gdextension file's library
# paths (e.g. lib_godot_infer.windows.template_debug.x86_64.dll) keep working
# unchanged.
get_target_property(GD_INFER_SUFFIX godot-cpp GODOTCPP_SUFFIX)
set_target_properties(godot_infer PROPERTIES
    PREFIX "lib_"
    OUTPUT_NAME "godot_infer${GD_INFER_SUFFIX}"
    RUNTIME_OUTPUT_DIRECTORY "${GD_INFER_PROJECT_ROOT}/demo/addons/godot_infer/bin"
    LIBRARY_OUTPUT_DIRECTORY "${GD_INFER_PROJECT_ROOT}/demo/addons/godot_infer/bin"
    # Must match godot-cpp's (and vcpkg's static-triplet libs') MSVC runtime
    # choice exactly, or the linker rejects mixing /MT(d) and /MD(d) objects.
    MSVC_RUNTIME_LIBRARY "$<$<PLATFORM_ID:Windows>:MultiThreaded$<$<CONFIG:Debug>:Debug>>"
)

# Multi-config generators (Visual Studio) append a per-config subdirectory
# (e.g. bin/Release/) to *_OUTPUT_DIRECTORY unless the per-config property
# variants are set too. The .gdextension file expects the library directly
# under bin/, with no config subfolder.
foreach(GD_INFER_CONFIG Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${GD_INFER_CONFIG}" GD_INFER_CONFIG_UPPER)
    set_target_properties(godot_infer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${GD_INFER_CONFIG_UPPER} "${GD_INFER_PROJECT_ROOT}/demo/addons/godot_infer/bin"
        LIBRARY_OUTPUT_DIRECTORY_${GD_INFER_CONFIG_UPPER} "${GD_INFER_PROJECT_ROOT}/demo/addons/godot_infer/bin"
    )
endforeach()
