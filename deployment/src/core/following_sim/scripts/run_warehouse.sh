#!/usr/bin/env bash
# Backwards-compat wrapper. Use `run_scenario.sh warehouse` for new code.
exec "$(dirname "${BASH_SOURCE[0]}")/run_scenario.sh" warehouse "$@"
