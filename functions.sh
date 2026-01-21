#!/bin/bash

submit_job() {
  local partition="$1"
  local nodelist="$2"
  local script="$3"
  local log_file="$4"
  shift 4
  local job_output
  job_output=$(sbatch --partition="$partition" \
                      --nodelist="$nodelist" \
                      --output="$log_file" \
                      "$script" "$@")
  if [[ "$job_output" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
      echo "${BASH_REMATCH[1]}"
  else
      echo "Submission failed: $job_output" >&2
      echo "Command: sbatch --partition=$partition --nodelist=$nodelist --output=$log_file $script $*" >&2
      exit 1
  fi
}

wait_for_job() {
  local job_id="$1"
  local description="$2"
  echo "Waiting for $description job ($job_id) to complete..."
  while true; do
    state=$(squeue -j "$job_id" -h -o %T 2>/dev/null)
    if [[ -z "$state" ]]; then
      echo "$description job completed"
      break
    else
      sleep 60
    fi
  done
}

wait_for_jobs() {
  local description="$1"
  shift
  local job_ids=("$@")

  if [[ ${#job_ids[@]} -eq 0 ]]; then
    echo "No $description jobs to wait for."
    return 0
  fi

  echo "Waiting for all $description jobs to complete (Total: ${#job_ids[@]})..."

  declare -A running_map

  while true; do
    local unfinished=0
    
    running_map=()

    local squeue_output
    if ! squeue_output=$(squeue -u "$USER" -h -o "%i" 2>/dev/null); then
      echo "Warning: squeue command failed (scheduler might be busy), retrying in 60 seconds..."
      sleep 60
      continue # Skip this loop iteration to avoid false completion detection
    fi

    # 2. Read output into array
    local running_jobs
    mapfile -t running_jobs <<< "$squeue_output"

    # 3. Build Hash Map
    for jid in "${running_jobs[@]}"; do
      # Process non-empty lines only
      if [[ -n "$jid" ]]; then
        running_map["$jid"]=1
      fi
    done

    # 4. Check status of target jobs
    for job_id in "${job_ids[@]}"; do
      if [[ -n "${running_map[$job_id]}" ]]; then
        unfinished=$((unfinished + 1))
      fi
    done

    # 5. Determine result
    if [[ "$unfinished" -eq 0 ]]; then
      echo "All $description jobs completed"
      break
    else
      # Optional: Print current time for troubleshooting
      echo "[$(date '+%H:%M:%S')] $unfinished $description jobs still pending/running..."
      sleep 60
    fi
  done
}

log_step() {
    echo "========================================================================================"
    echo "========== [Step $1] $2"
    echo "========================================================================================"
}

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S')  $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S')  $1" >&2
    exit 1
}