#!/bin/bash

if [ -f "$HOME/.zshrcenv" ]; then
    source "$HOME/.zshrcenv"
fi

TARGET_DIR="${1:-.}"

if ! command -v glslc &> /dev/null || ! command -v spirv-val &> /dev/null; then
    echo -e "\e[31mError: glslc or spirv-val not found in PATH.\e[0m"
    echo "Make sure your Vulkan SDK environment is sourced correctly."
    exit 1
fi

echo -e "\e[34mScanning for shaders in: $TARGET_DIR\e[0m"

find "$TARGET_DIR" -type f \( -name "*.frag" -o -name "*.comp" \) | while read -r SHADER_FILE; do
    OUTPUT_FILE="${SHADER_FILE}.spv"
    echo -n "Building ${SHADER_FILE} ... "
    if glslc "$SHADER_FILE" -o "$OUTPUT_FILE" 2> build_errors.log; then
        if spirv-val "$OUTPUT_FILE" 2> val_errors.log; then
            echo -e "\e[32mSUCCESS (Compiled & Validated)\e[0m"
        else
            echo -e "\e[31mVALIDATION FAILED\e[0m"
            cat val_errors.log
        fi
    else
        echo -e "\e[31mCOMPILATION FAILED\e[0m"
        cat build_errors.log
    fi
done

rm -f build_errors.log val_errors.log
