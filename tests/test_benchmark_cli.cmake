if(NOT DEFINED CLI OR CLI STREQUAL "" OR NOT DEFINED REFERENCE OR REFERENCE STREQUAL "")
    message(FATAL_ERROR "CLI and REFERENCE are required")
endif()
if(NOT EXISTS "${REFERENCE}/inputs.f32")
    message(FATAL_ERROR "REFERENCE must contain inputs.f32: ${REFERENCE}")
endif()

set(CLI_TIMEOUT_SECONDS 10)

function(check_error name expected_pattern)
    execute_process(
        COMMAND "${CLI}" ${ARGN}
        RESULT_VARIABLE exit_code
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
        TIMEOUT ${CLI_TIMEOUT_SECONDS})

    if("${exit_code}" STREQUAL "Process terminated due to timeout")
        message(FATAL_ERROR "${name}: CLI timed out after ${CLI_TIMEOUT_SECONDS} seconds\nstdout: ${stdout}\nstderr: ${stderr}")
    elseif(NOT "${exit_code}" MATCHES "^[0-9]+$")
        message(FATAL_ERROR "${name}: could not run CLI: ${exit_code}\nstdout: ${stdout}\nstderr: ${stderr}")
    endif()
    if(NOT exit_code EQUAL 2 OR NOT stdout STREQUAL "" OR NOT stderr MATCHES "${expected_pattern}")
        message(FATAL_ERROR "${name}: expected exit 2 and matching stderr\nstdout: ${stdout}\nstderr: ${stderr}")
    endif()
endfunction()

execute_process(
    COMMAND "${CLI}" --help
    RESULT_VARIABLE help_exit_code
    OUTPUT_VARIABLE help_stdout
    ERROR_VARIABLE help_stderr
    TIMEOUT ${CLI_TIMEOUT_SECONDS})
if(NOT help_exit_code EQUAL 0 OR NOT help_stderr STREQUAL "" OR
   NOT help_stdout MATCHES "Usage: benchmark_cli")
    message(FATAL_ERROR "help: unexpected result (exit ${help_exit_code})\nstdout: ${help_stdout}\nstderr: ${help_stderr}")
endif()

check_error(missing_model "missing required parameter.*--model" --input "${REFERENCE}/inputs.f32")
check_error(unknown_argument "unknown argument.*--unknown" --model "${REFERENCE}" --input "${REFERENCE}/inputs.f32" --unknown)
check_error(missing_value "missing value.*--iterations" --model "${REFERENCE}" --input "${REFERENCE}/inputs.f32" --iterations)
check_error(duplicate_iterations "duplicate option.*--iterations" --model "${REFERENCE}" --input "${REFERENCE}/inputs.f32" --iterations 3 --iterations 4)
check_error(invalid_batch "invalid value.*--batch" --model "${REFERENCE}" --input "${REFERENCE}/inputs.f32" --batch 0)
check_error(negative_iterations "invalid value.*--iterations" --model "${REFERENCE}" --input "${REFERENCE}/inputs.f32" --iterations -1)
check_error(invalid_warmup "invalid value.*--warmup" --model "${REFERENCE}" --input "${REFERENCE}/inputs.f32" --warmup -1)
check_error(overflow_iterations "invalid value.*--iterations" --model "${REFERENCE}" --input "${REFERENCE}/inputs.f32" --iterations 18446744073709551616)

execute_process(
    COMMAND "${CLI}" --model "${REFERENCE}" --input "${REFERENCE}/inputs.f32"
            --batch 64 --iterations 3 --warmup 1
    RESULT_VARIABLE benchmark_exit_code
    OUTPUT_VARIABLE benchmark_stdout
    ERROR_VARIABLE benchmark_stderr
    TIMEOUT ${CLI_TIMEOUT_SECONDS})
if("${benchmark_exit_code}" STREQUAL "Process terminated due to timeout")
    message(FATAL_ERROR "benchmark: CLI timed out after ${CLI_TIMEOUT_SECONDS} seconds\nstdout: ${benchmark_stdout}\nstderr: ${benchmark_stderr}")
elseif(NOT "${benchmark_exit_code}" MATCHES "^[0-9]+$")
    message(FATAL_ERROR "benchmark: could not run CLI: ${benchmark_exit_code}\nstdout: ${benchmark_stdout}\nstderr: ${benchmark_stderr}")
endif()
if(NOT benchmark_exit_code EQUAL 0 OR NOT benchmark_stderr STREQUAL "")
    message(FATAL_ERROR "benchmark: expected exit 0 and empty stderr, got ${benchmark_exit_code}\nstdout: ${benchmark_stdout}\nstderr: ${benchmark_stderr}")
endif()

function(assert_json_string key expected)
    string(JSON actual ERROR_VARIABLE json_error GET "${benchmark_stdout}" "${key}")
    if(NOT "${json_error}" STREQUAL "NOTFOUND" OR NOT actual STREQUAL "${expected}")
        message(FATAL_ERROR "benchmark: JSON field '${key}' was '${actual}', expected '${expected}' (${json_error})\nstdout: ${benchmark_stdout}")
    endif()
endfunction()

function(assert_json_number key expected)
    string(JSON value_type ERROR_VARIABLE json_error TYPE "${benchmark_stdout}" "${key}")
    if(NOT "${json_error}" STREQUAL "NOTFOUND" OR NOT value_type STREQUAL "NUMBER")
        message(FATAL_ERROR "benchmark: JSON field '${key}' is not a number (${json_error})\nstdout: ${benchmark_stdout}")
    endif()
    string(JSON actual GET "${benchmark_stdout}" "${key}")
    if(NOT actual STREQUAL "${expected}")
        message(FATAL_ERROR "benchmark: JSON field '${key}' was '${actual}', expected '${expected}'\nstdout: ${benchmark_stdout}")
    endif()
endfunction()

function(assert_json_finite_number key)
    string(JSON value_type ERROR_VARIABLE json_error TYPE "${benchmark_stdout}" "${key}")
    if(NOT "${json_error}" STREQUAL "NOTFOUND" OR NOT value_type STREQUAL "NUMBER")
        message(FATAL_ERROR "benchmark: JSON field '${key}' is not a number (${json_error})\nstdout: ${benchmark_stdout}")
    endif()
    string(JSON actual GET "${benchmark_stdout}" "${key}")
    if(NOT actual MATCHES "^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$")
        message(FATAL_ERROR "benchmark: JSON field '${key}' is not a finite number: '${actual}'\nstdout: ${benchmark_stdout}")
    endif()
endfunction()

function(assert_json_nonnegative_number key)
    assert_json_finite_number("${key}")
    string(JSON actual GET "${benchmark_stdout}" "${key}")
    if(actual MATCHES "^-")
        message(FATAL_ERROR "benchmark: JSON field '${key}' is negative: '${actual}'\nstdout: ${benchmark_stdout}")
    endif()
endfunction()

assert_json_string(mode "direct_cpu")
assert_json_number(batch 64)
assert_json_number(iterations 3)
assert_json_number(warmup 1)
assert_json_number(successes 3)
assert_json_number(failures 0)
assert_json_number(rejected 0)
assert_json_number(allocation_probe_iterations 3)
assert_json_number(allocation_probe_new_events 6)
assert_json_finite_number(checksum)

foreach(metric IN ITEMS wall_ms req_per_sec samples_per_sec mean_ms p50_ms p95_ms p99_ms
                        rss_after_warmup_bytes rss_after_timing_bytes peak_rss_bytes hardware_threads)
    assert_json_nonnegative_number("${metric}")
endforeach()

string(JSON allocation_scope GET "${benchmark_stdout}" allocation_probe_scope)
string(FIND "${allocation_scope}" "replaceable global C++ operator new/new[]" allocation_scope_match)
string(FIND "${allocation_scope}" "excludes malloc/mmap" allocation_exclusion_match)
if(allocation_scope_match EQUAL -1 OR allocation_exclusion_match EQUAL -1)
    message(FATAL_ERROR "benchmark: allocation scope does not describe its counting limits: '${allocation_scope}'")
endif()
