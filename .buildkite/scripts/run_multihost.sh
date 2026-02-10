#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Exit on error, exit on unset variable, fail on pipe errors.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "ERROR: Usage: $0 <command_and_args_to_run_in_docker...>"
  echo "Example: $0 python3 examples/offline_inference.py --model meta-llama/Llama-2-7b-hf"
  exit 1
fi

# Required environment variables for multi-host deployment
if [[ -z "${HEAD_SSH_IP:-}" ]] || [[ -z "${HEAD_INTERNAL_IP:-}" ]] || [[ -z "${WORKER_SSH_IPS:-}" ]]; then
  echo "ERROR: HEAD_SSH_IP, HEAD_INTERNAL_IP, and WORKER_SSH_IPS must be set in the environment."
  exit 1
fi

# We assume SSH_USER is provided or default to 'root' or current user
export SSH_USER="${SSH_USER:-$(whoami)}"

# Cleanup function that runs on exit to tear down the Ray cluster
cleanup() {
  echo "🧹 Cleaning up containers on head and workers..."
  IFS=',' read -r -a WORKER_IPS_ARRAY <<< "${WORKER_SSH_IPS}"
  ALL_IPS=("${HEAD_SSH_IP}" "${WORKER_IPS_ARRAY[@]}")
  
  for host in "${ALL_IPS[@]}"; do
    echo "   -> Cleaning ${host}"
    ssh -o StrictHostKeyChecking=no -o BatchMode=yes "${SSH_USER}@${host}" "docker stop node >/dev/null 2>&1 || true; docker rm -f node >/dev/null 2>&1 || true" || true
  done

  echo "   -> Cleaning vllm serve log on Head Node"
  ssh -o StrictHostKeyChecking=no -o BatchMode=yes "${SSH_USER}@${HEAD_SSH_IP}" "rm -f /root/vllm_serve.log" || true

  echo "✅ Cleanup complete."
}
trap cleanup EXIT

IMAGE_NAME='vllm-tpu'
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TOP_DIR=$(dirname $(dirname "$SCRIPT_DIR"))

# Source the environment setup script
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_docker_env.sh"
setup_environment $IMAGE_NAME

DOCKER_IMAGE="${IMAGE_NAME}:${BUILDKITE_COMMIT:-latest}"

# We should use runai_streamer instead of disks

# persist_cache_dir="/mnt/disks/persist/models"
# if ( mkdir -p "$persist_cache_dir" ); then
#   LOCAL_HF_HOME="$persist_cache_dir"
# else
#   echo "Error: Failed to create $persist_cache_dir"
#   exit 1
# fi

# Environment variables for docker run
ENV_VARS=(
  -e TEST_MODEL="${TEST_MODEL:-}"
  -e MINIMUM_ACCURACY_THRESHOLD="${MINIMUM_ACCURACY_THRESHOLD:-}"
  -e MINIMUM_THROUGHPUT_THRESHOLD="${MINIMUM_THROUGHPUT_THRESHOLD:-}"
  -e TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
  -e INPUT_LEN="${INPUT_LEN:-}"
  -e OUTPUT_LEN="${OUTPUT_LEN:-}"
  -e PREFIX_LEN="${PREFIX_LEN:-}"
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
  -e HF_TOKEN="${HF_TOKEN:-}"
)

# 1. Deploy the Ray cluster across multi-host TPU VMs
echo "--- Deploying Ray Cluster"
# We pipe 'y' to bypass the interactive cleanup prompt in deploy_cluster.sh
echo "y" | bash "${TOP_DIR}/scripts/multihost/deploy_cluster.sh" \
  -s "${TOP_DIR}/scripts/multihost/run_cluster.sh" \
  -d "${DOCKER_IMAGE}" \
  -c "${LOCAL_HF_HOME}" \
  -t "${HF_TOKEN:-}" \
  -H "${HEAD_SSH_IP}" \
  -i "${HEAD_INTERNAL_IP}" \
  -W "${WORKER_SSH_IPS}"

# Wait a few seconds for the Ray head node to come up
sleep 15

# 2. Start vLLM server on the head node
echo "--- Starting vLLM server on head node"
MODEL="${TEST_MODEL:-meta-llama/Llama-2-7b-hf}"
VLLM_PORT="8000"

# Launch vllm serve in the background inside the 'node' container
ssh -o StrictHostKeyChecking=no -o BatchMode=yes "${SSH_USER}@${HEAD_SSH_IP}" "docker exec \
  -d \
  -e HF_HOME=/root/.cache/huggingface \
  node bash -c \"vllm serve ${MODEL} \
    --port ${VLLM_PORT} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE:-8} \
    --trust-remote-code \
    --max-model-len 1024 \
    > /root/vllm_serve.log 2>&1\""

# 3. Wait for the server to be healthy
echo "--- Waiting for vLLM server to be healthy"
ssh -o StrictHostKeyChecking=no -o BatchMode=yes "${SSH_USER}@${HEAD_SSH_IP}" "docker exec node bash -c '
timeout=3600
for ((i=1; i<=timeout; i++)); do
  if curl -fs \"localhost:8000/health\" > /dev/null; then
    echo \"===== vLLM is healthy on port: 8000 ===\"
    exit 0
  fi
  sleep 1
done
echo \"Error: vLLM failed to become healthy within the timeout.\"
cat /root/vllm_serve.log
exit 1
'"

# 4. Run the curl test to verify the endpoint
echo "--- Running curl test"
ssh -o StrictHostKeyChecking=no -o BatchMode=yes "${SSH_USER}@${HEAD_SSH_IP}" "docker exec node bash -c '
curl http://localhost:8000/v1/completions \
  -X POST \
  -H \"Content-Type: application/json\" \
  -d \"{\\\"model\\\": \\\"${MODEL}\\\", \\\"prompt\\\": \\\"San Francisco is a\\\", \\\"max_tokens\\\": 50}\"
'"

echo "" # Newline for clean output

# 5. Run additional commanded tests if arg is provided
if [ "$#" -gt 0 ]; then
  # If the first argument is specifically a command, run it. 
  # Otherwise, we might just be passing a single command. 
  echo "--- Running additional Tests on Head Node"
  COMMAND_ARGS=("$@")
  
  ssh -o StrictHostKeyChecking=no -o BatchMode=yes "${SSH_USER}@${HEAD_SSH_IP}" "docker exec \
    -e HF_HOME=/root/.cache/huggingface \
    node ${COMMAND_ARGS[*]}"
fi

# Note: The Ray cluster containers remain running until explicitly cleaned up.
# If cleanup is required at the end of the CI step, we can either trigger it here
# or rely on the host's cleanup logic. For safety, we keep them running or 
# clean them manually in the next steps.

echo "--- Tests completed successfully"
