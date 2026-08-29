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

# Pruning is destructive, so exercise it only on a temporary copy of the
# fixture and confirm that a real compiler failure removes just that source.
set(prune_source "${OUTPUT_DIRECTORY}-prune-source")
set(prune_output "${OUTPUT_DIRECTORY}-prune-output")
file(REMOVE_RECURSE "${prune_source}" "${prune_output}")
file(MAKE_DIRECTORY "${prune_source}")
file(COPY
    "${SOURCE_DIRECTORY}/library.json"
    "${SOURCE_DIRECTORY}/valid.frag"
    "${SOURCE_DIRECTORY}/invalid.frag"
    DESTINATION "${prune_source}"
)

# A prune request without explicit confirmation must stop before touching the
# copied source tree or creating the requested output directory.
execute_process(
    COMMAND "${ACMXVK_EXECUTABLE}"
        --build "${prune_source}/library.json"
        --fix "${prune_output}"
        --prune
        --glslc "${GLSLC_EXECUTABLE}"
    RESULT_VARIABLE unconfirmed_result
    OUTPUT_VARIABLE unconfirmed_output
    ERROR_VARIABLE unconfirmed_errors
)
if(unconfirmed_result EQUAL 0)
    message(FATAL_ERROR "unconfirmed --prune unexpectedly succeeded")
endif()
if(NOT EXISTS "${prune_source}/valid.frag" OR
   NOT EXISTS "${prune_source}/invalid.frag")
    message(FATAL_ERROR "unconfirmed --prune modified a source file")
endif()
if(EXISTS "${prune_output}")
    message(FATAL_ERROR "unconfirmed --prune created its output directory")
endif()
string(FIND "${unconfirmed_errors}" "WARNING: --prune permanently deletes"
    warning_position)
string(FIND "${unconfirmed_errors}" "--force" force_position)
if(warning_position EQUAL -1 OR force_position EQUAL -1)
    message(FATAL_ERROR
        "unconfirmed --prune omitted its warning:\n${unconfirmed_errors}"
    )
endif()

execute_process(
    COMMAND "${ACMXVK_EXECUTABLE}"
        --build "${prune_source}/library.json"
        --fix "${prune_output}"
        --prune
        --force
        --glslc "${GLSLC_EXECUTABLE}"
    RESULT_VARIABLE prune_result
    OUTPUT_VARIABLE prune_output_text
    ERROR_VARIABLE prune_errors
)
if(NOT prune_result EQUAL 0)
    message(FATAL_ERROR
        "--prune build returned ${prune_result}\n"
        "${prune_output_text}${prune_errors}"
    )
endif()
if(NOT EXISTS "${prune_source}/valid.frag")
    message(FATAL_ERROR "--prune removed the valid shader source")
endif()
if(EXISTS "${prune_source}/invalid.frag")
    message(FATAL_ERROR "--prune retained the failed shader source")
endif()
if(EXISTS "${prune_output}/invalid.frag.spv")
    message(FATAL_ERROR "--prune retained the failed shader output")
endif()
string(FIND "${prune_errors}" "acmxvk: pruned failed source '"
    prune_position)
if(prune_position EQUAL -1)
    message(FATAL_ERROR
        "--prune did not report the deleted source:\n${prune_errors}"
    )
endif()
