# The V1 synchronous adapter also builds without any inference backend enabled.
add_executable(depth_replay "${PROJECT_SOURCE_DIR}/apps/depth_replay.cpp")
target_link_libraries(depth_replay PRIVATE miniinfer::depth_sdk Threads::Threads)
miniinfer_configure_target(depth_replay)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME depth_replay_statistics
            COMMAND "${Python3_EXECUTABLE}" -m unittest discover
                -s "${PROJECT_SOURCE_DIR}/tests" -p test_depth_replay.py -v)
        set_tests_properties(depth_replay_statistics PROPERTIES TIMEOUT 30)
    endif()
endif()
