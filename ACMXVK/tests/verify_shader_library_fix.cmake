if(NOT DEFINED ACMXVK_EXECUTABLE OR
   NOT DEFINED GLSLC_EXECUTABLE OR
   NOT DEFINED SOURCE_DIRECTORY OR
   NOT DEFINED OUTPUT_DIRECTORY)
    message(FATAL_ERROR "shader-library fix test is missing an input")
endif()

file(REMOVE_RECURSE "${OUTPUT_DIRECTORY}")
file(MAKE_DIRECTORY "${OUTPUT_DIRECTORY}")
file(WRITE "${OUTPUT_DIRECTORY}/invalid.frag.spv" "stale module")

execute_process(
    COMMAND "${ACMXVK_EXECUTABLE}"
        --build "${SOURCE_DIRECTORY}/library.json"
        --fix "${OUTPUT_DIRECTORY}"
        --glslc "${GLSLC_EXECUTABLE}"
    RESULT_VARIABLE build_result
    OUTPUT_VARIABLE build_output
    ERROR_VARIABLE build_errors
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR
        "--fix build returned ${build_result}\n${build_output}${build_errors}"
    )
endif()

foreach(percentage RANGE 5 100 5)
    string(FIND "${build_output}"
        "acmxvk: build progress: ${percentage}%" progress_position)
    if(progress_position EQUAL -1)
        message(FATAL_ERROR
            "--fix output omitted ${percentage}% progress:\n${build_output}"
        )
    endif()
endforeach()
string(FIND "${build_errors}" "acmxvk: fix omitted 'invalid.frag'"
    failure_position)
if(failure_position EQUAL -1)
    message(FATAL_ERROR
        "--fix did not report the invalid shader:\n${build_errors}"
    )
endif()

if(NOT EXISTS "${OUTPUT_DIRECTORY}/valid.frag.spv")
    message(FATAL_ERROR "valid shader was not compiled")
endif()
if(EXISTS "${OUTPUT_DIRECTORY}/invalid.frag.spv")
    message(FATAL_ERROR "failed shader output was not removed")
endif()

file(READ "${OUTPUT_DIRECTORY}/library.json" output_manifest)
string(FIND "${output_manifest}" "valid.frag.spv" valid_position)
string(FIND "${output_manifest}" "invalid.frag.spv" invalid_position)
if(valid_position EQUAL -1)
    message(FATAL_ERROR "valid shader is absent from output library.json")
endif()
if(NOT invalid_position EQUAL -1)
    message(FATAL_ERROR "failed shader remains in output library.json")
endif()
