cmake_minimum_required(VERSION 3.28)

project(MiniInfer VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "MiniInfer v0.1 targets Linux. On Windows, configure and build through WSL.")
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
    message(FATAL_ERROR "MiniInfer v0.1 requires GCC or Clang.")
endif()
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Use an out-of-source build directory, such as ~/builds/MiniInfer/debug.")
endif()

option(MINIINFER_ENABLE_ASAN_UBSAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(MINIINFER_ENABLE_TSAN "Enable ThreadSanitizer in a separate build" OFF)
if(MINIINFER_ENABLE_ASAN_UBSAN AND MINIINFER_ENABLE_TSAN)
    message(FATAL_ERROR "MINIINFER_ENABLE_ASAN_UBSAN and MINIINFER_ENABLE_TSAN are mutually exclusive. Use separate build directories.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

function(miniinfer_configure_target target)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(MINIINFER_ENABLE_ASAN_UBSAN)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    elseif(MINIINFER_ENABLE_TSAN)
        target_compile_options(${target} PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=thread)
    endif()
endfunction()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(miniinfer_core STATIC
    src/version.cpp src/checked_size.cpp src/buffer.cpp src/mlp_model.cpp
    src/data_io.cpp src/cpu_ops.cpp src/mlp_input.cpp src/mlp_result.cpp
    src/workspace.cpp src/cpu_executor.cpp src/thread_pool.cpp src/runtime.cpp)
add_library(miniinfer::core ALIAS miniinfer_core)
target_include_directories(miniinfer_core PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(miniinfer_core PUBLIC cxx_std_17)
target_compile_definitions(miniinfer_core PRIVATE MINIINFER_VERSION_STRING="${PROJECT_VERSION}")
target_link_libraries(miniinfer_core PUBLIC Threads::Threads)
miniinfer_configure_target(miniinfer_core)

add_executable(infer_cli apps/infer_cli.cpp)
target_link_libraries(infer_cli PRIVATE miniinfer::core)
miniinfer_configure_target(infer_cli)

add_executable(benchmark_cli apps/benchmark_cli.cpp)
target_link_libraries(benchmark_cli PRIVATE miniinfer::core)
miniinfer_configure_target(benchmark_cli)

add_executable(runtime_demo apps/runtime_demo.cpp)
target_link_libraries(runtime_demo PRIVATE miniinfer::core)
miniinfer_configure_target(runtime_demo)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_project tests/test_project.cpp)
    target_link_libraries(test_project PRIVATE miniinfer::core)
    miniinfer_configure_target(test_project)
    add_test(NAME project_contract COMMAND test_project)
    add_test(NAME cli_contract
        COMMAND "${CMAKE_COMMAND}"
            "-DCLI=$<TARGET_FILE:infer_cli>"
            "-DEXPECTED_VERSION=${PROJECT_VERSION}"
            "-DREFERENCE=${PROJECT_SOURCE_DIR}/data/reference"
            -P "${PROJECT_SOURCE_DIR}/tests/test_cli.cmake")
    add_executable(test_buffer tests/test_buffer.cpp)
    target_link_libraries(test_buffer PRIVATE miniinfer::core)
    miniinfer_configure_target(test_buffer)
    add_test(NAME buffer_contract COMMAND test_buffer)
    add_executable(test_tensor_view tests/test_tensor_view.cpp)
    target_link_libraries(test_tensor_view PRIVATE miniinfer::core)
    miniinfer_configure_target(test_tensor_view)
    add_test(NAME tensor_view_contract COMMAND test_tensor_view)
    add_executable(test_model tests/test_model.cpp)
    target_link_libraries(test_model PRIVATE miniinfer::core)
    miniinfer_configure_target(test_model)
    add_test(NAME model_contract COMMAND test_model "${PROJECT_SOURCE_DIR}/data/reference")
    add_executable(test_cpu_ops tests/test_cpu_ops.cpp)
    target_link_libraries(test_cpu_ops PRIVATE miniinfer::core)
    miniinfer_configure_target(test_cpu_ops)
    add_test(NAME cpu_ops_contract COMMAND test_cpu_ops)
    add_executable(test_inference tests/test_inference.cpp)
    target_link_libraries(test_inference PRIVATE miniinfer::core)
    miniinfer_configure_target(test_inference)
    add_test(NAME inference_contract COMMAND test_inference "${PROJECT_SOURCE_DIR}/data/reference")
    add_executable(test_runtime tests/test_runtime.cpp)
    target_link_libraries(test_runtime PRIVATE miniinfer::core)
    miniinfer_configure_target(test_runtime)
    add_test(NAME runtime_contract COMMAND test_runtime "${PROJECT_SOURCE_DIR}/data/reference")
    add_executable(test_thread_pool tests/test_thread_pool.cpp)
    target_link_libraries(test_thread_pool PRIVATE miniinfer::core)
    miniinfer_configure_target(test_thread_pool)
    add_test(NAME thread_pool_contract COMMAND test_thread_pool)
    foreach(test_name IN ITEMS concurrency shutdown timing)
        add_executable(test_${test_name} tests/test_${test_name}.cpp)
        target_link_libraries(test_${test_name} PRIVATE miniinfer::core)
        miniinfer_configure_target(test_${test_name})
    endforeach()
    add_test(NAME concurrency_contract COMMAND test_concurrency "${PROJECT_SOURCE_DIR}/data/reference")
    add_test(NAME shutdown_contract COMMAND test_shutdown "${PROJECT_SOURCE_DIR}/data/reference")
    add_test(NAME timing_contract COMMAND test_timing)
    add_test(NAME runtime_demo_contract COMMAND runtime_demo "${PROJECT_SOURCE_DIR}/data/reference")
    set_tests_properties(concurrency_contract shutdown_contract PROPERTIES TIMEOUT 90)
    set_tests_properties(timing_contract runtime_demo_contract PROPERTIES TIMEOUT 20)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME runtime_baseline_driver_contract
            COMMAND "${Python3_EXECUTABLE}" -m unittest discover
                -s "${PROJECT_SOURCE_DIR}/tests" -p test_runtime_baseline.py -v)
        set_tests_properties(runtime_baseline_driver_contract PROPERTIES TIMEOUT 20)
    endif()
    add_test(NAME benchmark_cli_contract
        COMMAND "${CMAKE_COMMAND}" "-DCLI=$<TARGET_FILE:benchmark_cli>"
            "-DREFERENCE=${PROJECT_SOURCE_DIR}/data/reference"
            -P "${PROJECT_SOURCE_DIR}/tests/test_benchmark_cli.cmake")
    set_tests_properties(project_contract cli_contract buffer_contract
        tensor_view_contract model_contract cpu_ops_contract inference_contract runtime_contract
        thread_pool_contract benchmark_cli_contract
        PROPERTIES TIMEOUT 20)
endif()
