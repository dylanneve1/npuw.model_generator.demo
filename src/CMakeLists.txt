# Standalone build for npuw_model_generator_demo
#
# Requires:
#   OPENVINO_SOURCE_DIR  - path to OpenVINO source tree (has model_builder.cpp/hpp)
#   OpenVINO installed   - found via find_package (set OpenVINO_DIR or source setupvars.sh)
#
# Usage:
#   source <openvino_install>/setupvars.sh
#   cmake -S src -B build -DOPENVINO_SOURCE_DIR=~/openvino
#   cmake --build build

cmake_minimum_required(VERSION 3.16)
project(npuw_model_generator_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Accept OPENVINO_SOURCE_DIR from cmake var or env
if(NOT OPENVINO_SOURCE_DIR)
    if(DEFINED ENV{OPENVINO_SOURCE_DIR})
        set(OPENVINO_SOURCE_DIR "$ENV{OPENVINO_SOURCE_DIR}")
    else()
        message(FATAL_ERROR
            "OPENVINO_SOURCE_DIR not set. Pass -DOPENVINO_SOURCE_DIR=<path> or export it.\n"
            "It should point to the OpenVINO source tree root.")
    endif()
endif()

set(MODEL_BUILDER_DIR "${OPENVINO_SOURCE_DIR}/src/plugins/intel_npu/tests/unit/npuw/model_builder")

if(NOT EXISTS "${MODEL_BUILDER_DIR}/model_builder.cpp")
    message(FATAL_ERROR
        "model_builder.cpp not found at ${MODEL_BUILDER_DIR}\n"
        "Check that OPENVINO_SOURCE_DIR points to the correct OpenVINO source tree.")
endif()

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_executable(npuw_model_generator_demo
    demo_model_generator.cpp
    ${MODEL_BUILDER_DIR}/model_builder.cpp
)

target_include_directories(npuw_model_generator_demo PRIVATE
    ${MODEL_BUILDER_DIR}
)

target_link_libraries(npuw_model_generator_demo PRIVATE
    openvino::runtime
    ${CMAKE_DL_LIBS}
)
