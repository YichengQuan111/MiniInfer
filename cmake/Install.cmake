include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(depth_install_targets miniinfer_core miniinfer_net miniinfer_depth_image miniinfer_depth_artifact miniinfer_depth_sdk)
foreach(pair IN ITEMS "miniinfer_core;core" "miniinfer_net;net" "miniinfer_depth_image;depth_image"
        "miniinfer_depth_artifact;depth_artifact" "miniinfer_depth_sdk;depth_sdk"
        "miniinfer_depth_ort;depth_ort" "miniinfer_depth_trt;depth_trt" "miniinfer_depth_net;depth_net")
    list(GET pair 0 target)
    list(GET pair 1 exported)
    if(TARGET ${target})
        set_target_properties(${target} PROPERTIES EXPORT_NAME ${exported})
    endif()
endforeach()
foreach(optional miniinfer_depth_ort miniinfer_depth_trt miniinfer_depth_net)
    if(TARGET ${optional})
        list(APPEND depth_install_targets ${optional})
    endif()
endforeach()
install(TARGETS ${depth_install_targets} EXPORT MiniInferTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/miniinfer" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
foreach(app depth_cli depth_trt_cli depth_engine_builder depth_sdk_cli depth_server depth_client depth_benchmark depth_replay depth_preprocess_benchmark)
    if(TARGET ${app})
        install(TARGETS ${app} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endforeach()
configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/MiniInferConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MiniInferConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniInfer)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/MiniInferConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(EXPORT MiniInferTargets NAMESPACE miniinfer:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniInfer)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/MiniInferConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/MiniInferConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MiniInfer)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/examples/depth_sdk"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/MiniInfer/examples)
install(FILES "${PROJECT_SOURCE_DIR}/data/depth/contract.json"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/MiniInfer)
install(FILES "${PROJECT_SOURCE_DIR}/docs/depth_deployment.md"
    "${PROJECT_SOURCE_DIR}/docs/depth_tcp.md" "${PROJECT_SOURCE_DIR}/docs/depth_benchmark_method.md"
    "${PROJECT_SOURCE_DIR}/docs/depth_accuracy.md" "${PROJECT_SOURCE_DIR}/docs/depth_validation.md"
    "${PROJECT_SOURCE_DIR}/docs/depth_performance.md" "${PROJECT_SOURCE_DIR}/docs/depth_troubleshooting.md"
    "${PROJECT_SOURCE_DIR}/docs/depth_v2_upgrade.md"
    "${PROJECT_SOURCE_DIR}/docs/MiniInfer_V1_V2_求职改进策略.md"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/MiniInfer/docs)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/data/depth/"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/MiniInfer/data/depth)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/docs/validation/"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/MiniInfer/docs/validation)
