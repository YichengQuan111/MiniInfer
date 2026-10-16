# Included after miniinfer_depth_trt exists. CPU-only and legacy TRT builds
# remain possible without a CUDA compiler; requesting a GPU path then fails.
if(MINIINFER_ENABLE_CUDA_PREPROCESS)
    if(NOT MINIINFER_ENABLE_TRT)
        message(FATAL_ERROR "CUDA preprocessing requires MINIINFER_ENABLE_TRT")
    endif()
    enable_language(CUDA)
    set_target_properties(miniinfer_depth_trt PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY None)
    target_sources(miniinfer_depth_trt PRIVATE "${PROJECT_SOURCE_DIR}/src/depth/preprocess_cuda.cu")
    target_compile_definitions(miniinfer_depth_trt PRIVATE MINIINFER_HAS_CUDA_PREPROCESS=1)
    target_compile_options(miniinfer_depth_trt PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-Xcompiler=-ffp-contract=off,-fno-fast-math>")
    add_executable(depth_preprocess_benchmark "${PROJECT_SOURCE_DIR}/apps/depth_preprocess_benchmark.cpp")
    target_link_libraries(depth_preprocess_benchmark PRIVATE miniinfer::depth_trt)
    target_include_directories(depth_preprocess_benchmark PRIVATE "${PROJECT_SOURCE_DIR}/src/depth")
    target_include_directories(depth_preprocess_benchmark SYSTEM PRIVATE
        "${MINIINFER_CUDA_RUNTIME}/include" "${MINIINFER_CUDA_NVCC_INCLUDE}")
    target_compile_options(depth_preprocess_benchmark PRIVATE -ffp-contract=off -fno-fast-math)
    miniinfer_configure_target(depth_preprocess_benchmark)
    if(BUILD_TESTING)
        add_executable(test_depth_preprocess_cuda "${PROJECT_SOURCE_DIR}/tests/test_depth_preprocess_cuda.cpp")
        target_link_libraries(test_depth_preprocess_cuda PRIVATE miniinfer::depth_trt)
        target_include_directories(test_depth_preprocess_cuda PRIVATE "${PROJECT_SOURCE_DIR}/src/depth")
        target_include_directories(test_depth_preprocess_cuda SYSTEM PRIVATE
            "${MINIINFER_CUDA_RUNTIME}/include" "${MINIINFER_CUDA_NVCC_INCLUDE}")
        miniinfer_configure_target(test_depth_preprocess_cuda)
        add_test(NAME depth_preprocess_cuda_contract COMMAND test_depth_preprocess_cuda)
        set_tests_properties(depth_preprocess_cuda_contract PROPERTIES
            SKIP_RETURN_CODE 77 RESOURCE_LOCK depth_gpu TIMEOUT 120)
        add_executable(test_depth_gpu_pipeline "${PROJECT_SOURCE_DIR}/tests/test_depth_gpu_pipeline.cpp")
        target_link_libraries(test_depth_gpu_pipeline PRIVATE miniinfer::depth_trt)
        miniinfer_configure_target(test_depth_gpu_pipeline)
    endif()
endif()
