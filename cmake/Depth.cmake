# Include after miniinfer_core and miniinfer_configure_target are defined.
option(MINIINFER_ENABLE_ORT "Build CPU ONNX Runtime depth backend" OFF)
set(MINIINFER_ORT_ROOT "" CACHE PATH "ONNX Runtime distribution root with include/ and lib/")
set(MINIINFER_DEPTH_MODEL "" CACHE FILEPATH "Optional static Depth Anything V2 ONNX test model")
set(MINIINFER_DEPTH_REFERENCE_DIR "" CACHE PATH "Optional one-sample golden directory containing original.ppm and *.f32")
set(MINIINFER_DEPTH_ATOL "" CACHE STRING "Explicit frozen FP32 absolute tolerance for golden ORT test")
set(MINIINFER_DEPTH_RTOL "" CACHE STRING "Explicit frozen FP32 relative tolerance for golden ORT test")
set(MINIINFER_DEPTH_ALLOWED_RATIO "" CACHE STRING "Explicit allowed out-of-tolerance fraction for golden ORT test")

add_library(miniinfer_depth_image STATIC
    "${PROJECT_SOURCE_DIR}/src/depth/image.cpp"
    "${PROJECT_SOURCE_DIR}/src/depth/preprocess.cpp")
add_library(miniinfer::depth_image ALIAS miniinfer_depth_image)
target_include_directories(miniinfer_depth_image PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(miniinfer_depth_image PUBLIC cxx_std_17)
miniinfer_configure_target(miniinfer_depth_image)
target_compile_options(miniinfer_depth_image PRIVATE -ffp-contract=off -fno-fast-math)

if(BUILD_TESTING)
    add_executable(test_depth_image "${PROJECT_SOURCE_DIR}/tests/test_depth_image.cpp")
    target_link_libraries(test_depth_image PRIVATE miniinfer::depth_image)
    miniinfer_configure_target(test_depth_image)
    target_compile_options(test_depth_image PRIVATE -ffp-contract=off -fno-fast-math)
    add_test(NAME depth_image_contract COMMAND test_depth_image)
    set_tests_properties(depth_image_contract PROPERTIES TIMEOUT 30)
endif()

if(MINIINFER_ENABLE_ORT)
    if(NOT MINIINFER_ORT_ROOT)
        message(FATAL_ERROR "MINIINFER_ENABLE_ORT requires MINIINFER_ORT_ROOT")
    endif()
    find_path(MINIINFER_ORT_INCLUDE onnxruntime_cxx_api.h
        PATHS "${MINIINFER_ORT_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
    find_library(MINIINFER_ORT_LIBRARY onnxruntime
        PATHS "${MINIINFER_ORT_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
    add_library(miniinfer_ort_external SHARED IMPORTED)
    set_target_properties(miniinfer_ort_external PROPERTIES
        IMPORTED_LOCATION "${MINIINFER_ORT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MINIINFER_ORT_INCLUDE}")
    add_library(miniinfer_depth_ort STATIC "${PROJECT_SOURCE_DIR}/src/depth/ort_runner.cpp")
    add_library(miniinfer::depth_ort ALIAS miniinfer_depth_ort)
    target_link_libraries(miniinfer_depth_ort PUBLIC miniinfer::depth_image PRIVATE miniinfer_ort_external)
    miniinfer_configure_target(miniinfer_depth_ort)
    add_executable(depth_cli "${PROJECT_SOURCE_DIR}/apps/depth_cli.cpp")
    target_link_libraries(depth_cli PRIVATE miniinfer::depth_ort)
    miniinfer_configure_target(depth_cli)
    set_target_properties(depth_cli PROPERTIES BUILD_RPATH "${MINIINFER_ORT_ROOT}/lib")
    if(BUILD_TESTING)
        add_executable(test_depth_ort "${PROJECT_SOURCE_DIR}/tests/test_depth_ort.cpp")
        target_link_libraries(test_depth_ort PRIVATE miniinfer::depth_ort)
        miniinfer_configure_target(test_depth_ort)
        set_target_properties(test_depth_ort PROPERTIES BUILD_RPATH "${MINIINFER_ORT_ROOT}/lib")
        add_test(NAME depth_ort_contract COMMAND test_depth_ort)
        set_tests_properties(depth_ort_contract PROPERTIES TIMEOUT 30)
        if(MINIINFER_DEPTH_MODEL AND MINIINFER_DEPTH_REFERENCE_DIR
           AND NOT MINIINFER_DEPTH_ATOL STREQUAL "" AND NOT MINIINFER_DEPTH_RTOL STREQUAL ""
           AND NOT MINIINFER_DEPTH_ALLOWED_RATIO STREQUAL "")
            add_test(NAME depth_ort_reference COMMAND test_depth_ort
                "${MINIINFER_DEPTH_MODEL}" "${MINIINFER_DEPTH_REFERENCE_DIR}"
                "${MINIINFER_DEPTH_ATOL}" "${MINIINFER_DEPTH_RTOL}" "${MINIINFER_DEPTH_ALLOWED_RATIO}")
            set_tests_properties(depth_ort_reference PROPERTIES TIMEOUT 180)
        endif()
    endif()
endif()
