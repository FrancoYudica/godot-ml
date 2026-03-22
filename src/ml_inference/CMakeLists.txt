cmake_minimum_required(VERSION 3.50)

project(onnx_parser VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ─────────────────────────────────────────────────────────────

include(FetchContent)

# Protobuf (required by ONNX)
find_package(Protobuf REQUIRED)

# ONNX
FetchContent_Declare(
    onnx
    GIT_REPOSITORY https://github.com/onnx/onnx.git
    GIT_TAG        v1.16.0
)
set(ONNX_ML OFF)                  # disable ML extensions, not needed
set(ONNX_BUILD_TESTS OFF)
set(ONNX_BUILD_BENCHMARKS OFF)
set(ONNX_VERIFY_PROTO3 OFF)
FetchContent_MakeAvailable(onnx)

# ── Library ──────────────────────────────────────────────────────────────────

add_library(onnx_parser STATIC
    ml_parser.cpp ml_utils.cpp
)

target_include_directories(onnx_parser PUBLIC
    include/
    ${PROTOBUF_INCLUDE_DIRS}
)

target_link_libraries(onnx_parser PUBLIC
    onnx
    ${PROTOBUF_LIBRARIES}
)

# ── Test executable ──────────────────────────────────────────────────────────
add_executable(parser_test tests/parser_test.cpp)

find_package(absl REQUIRED)

target_link_libraries(parser_test PRIVATE
    onnx_parser
    onnx
    ${PROTOBUF_LIBRARIES}
    absl::base
    absl::log_internal_check_op
    absl::log_internal_message
    absl::status
    absl::strings
)