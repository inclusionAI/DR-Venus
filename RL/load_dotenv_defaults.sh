#!/usr/bin/env bash

# Source dotenv assignments as defaults: every variable that already exists in
# the caller keeps both its exact value (including an empty value) and export
# state. Values are never printed.
load_dotenv_defaults() {
    local dotenv_path=$1
    local line name declaration
    local allexport_was_set=false
    local -A previous_values=()
    local -A previous_exported=()

    while IFS= read -r line || [ -n "$line" ]; do
        if [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*= ]]; then
            name=${BASH_REMATCH[2]}
            if [[ -v "$name" && ! -v "previous_values[$name]" ]]; then
                previous_values["$name"]=${!name}
                declaration=$(declare -p "$name" 2>/dev/null || true)
                if [[ "$declaration" =~ ^declare[[:space:]]+-[^[:space:]]*x ]]; then
                    previous_exported["$name"]=true
                else
                    previous_exported["$name"]=false
                fi
            fi
        fi
    done <"$dotenv_path"

    if [[ "$-" == *a* ]]; then
        allexport_was_set=true
    else
        set -a
    fi
    # shellcheck disable=SC1090
    source "$dotenv_path"
    if [ "$allexport_was_set" != "true" ]; then
        set +a
    fi

    for name in "${!previous_values[@]}"; do
        printf -v "$name" '%s' "${previous_values[$name]}"
        if [ "${previous_exported[$name]}" = "true" ]; then
            export "$name"
        else
            export -n "$name"
        fi
    done
}
