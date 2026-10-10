#!/usr/bin/env bash
set -euo pipefail
GATHER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${GATHER_ROOT}"
GATHER_PYTHON=${PYTHON_BIN:-python3}
export TEST_VISIBLE_DEVICES=${TEST_VISIBLE_DEVICES:-${A5_TEST_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}}
# Help and validation work on development hosts without CANN or Torch.
GATHER_PREVIEW=${DRY_RUN:-0}
for GATHER_ARG in "$@"; do
  case "${GATHER_ARG}" in --dry-run|--help|-h) GATHER_PREVIEW=1 ;; esac
done
if [[ "${GATHER_PREVIEW}" != 1 ]]; then
  export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
  export CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-${ASCEND_HOME_PATH}}
  if [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
    set +u
    source "${ASCEND_HOME_PATH}/set_env.sh"
    set -u
  fi
  export SOC_VERSION=ascend950
  export ASCEND_LAUNCH_BLOCKING=0
  export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
fi
export ASCEND_RT_VISIBLE_DEVICES=${TEST_VISIBLE_DEVICES}
export PYTHONUNBUFFERED=1
exec "${GATHER_PYTHON}" "${GATHER_ROOT}/tests/run_gather_pa_kv_cache_sweep.py" "$@"
