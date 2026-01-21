#!/bin/bash

# ==================
SCRIPT="/lustre/grp/cmclab/share/liuy/1_AF3Score/AF3score_main/AF3score_pipeline.sh"
INPUT_DIRS=(
  "/lustre/grp/cmclab/share/liuy/1_AF3Score/AF3score_test"
)

OUTPUT_PARENT_DIR="/lustre/grp/cmclab/share/liuy/1_AF3Score/AF3score_test"
NUM_JOBS=3
LOG_DIR="${OUTPUT_PARENT_DIR}/logs"
# ==================

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_PARENT_DIR"

i=1
for input in "${INPUT_DIRS[@]}"; do
  input_name=$(basename "$input")
  
  output="${OUTPUT_PARENT_DIR}/${input_name}_af3score"
  mkdir -p "$output"
  
  log_file="${LOG_DIR}/${input_name}.log"
  
  echo "Running task $i:"
  echo "  input : $input"
  echo "  output: $output"
  echo "  log   : $log_file"
  
  nohup bash "$SCRIPT" "$input" "$output" "$NUM_JOBS" > "$log_file" 2>&1 &
  
  i=$((i+1))
done

echo "全部任务已使用 nohup 启动。"
