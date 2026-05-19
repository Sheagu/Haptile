#!/usr/bin/env bash
#
# 用法：
#   chmod +x check_gpu_partition_resources.sh
#   bash check_gpu_partition_resources.sh
#   bash check_gpu_partition_resources.sh --partition gpu
#   bash check_gpu_partition_resources.sh --all
#   bash check_gpu_partition_resources.sh --partition gpu --debug-node erc-hpc-comp032
#
# 如果你想用 ./check_gpu_partition_resources.sh 直接运行，需要先加执行权限：
#   chmod +x check_gpu_partition_resources.sh
#   ./check_gpu_partition_resources.sh
#
# 说明：
#   默认查询所有 Slurm partition，只输出同时有空闲 GPU 和可用 CPU 的节点。
#   --partition NAME 可以只查询某个 partition。
#   --all 会输出所有 GPU 节点，包括当前没有空闲 GPU 的节点。
#   --debug-node NODE 会把某个节点解析到的 CfgTRES / AllocTRES 打印到 stderr，
#   方便检查脚本是否读到了 Slurm 的真实已分配资源。
#   输出列包括节点名、节点状态、空闲 GPU、可用 CPU、可用内存、
#   总内存、GPU 类型/数量，以及根据 GPU 类型或节点 feature 推断的单卡显存大小。
#   资源计算优先使用 scontrol 里的 CfgTRES / AllocTRES，避免把已分配 GPU
#   误判为空闲 GPU。

set -euo pipefail

PARTITION=""
SHOW_ALL=0
DEBUG_NODE=""

usage() {
  cat <<'EOF'
Usage: ./check_gpu_partition_resources.sh [--partition PARTITION] [--all] [--debug-node NODE]

List Slurm GPU nodes across all partitions that have free GPUs and free CPUs,
together with available CPU cores, memory, and GPU memory inferred from the GRES
GPU type.

Examples:
  ./check_gpu_partition_resources.sh
  ./check_gpu_partition_resources.sh --partition gpu
  ./check_gpu_partition_resources.sh --all
  ./check_gpu_partition_resources.sh --partition gpu --debug-node erc-hpc-comp032

Options:
  --partition PARTITION
           Query only one partition. By default, query all partitions.
  --all    Show all GPU nodes, including nodes with no free GPU.
  --debug-node NODE
           Print parsed Slurm fields for one node.
  -h,--help
           Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  arg="$1"
  case "$arg" in
    --all)
      SHOW_ALL=1
      shift
      ;;
    --partition|-p)
      PARTITION="${2:-}"
      if [[ -z "$PARTITION" ]]; then
        echo "Error: --partition requires a partition name." >&2
        exit 1
      fi
      shift 2
      ;;
    --debug-node)
      DEBUG_NODE="${2:-}"
      if [[ -z "$DEBUG_NODE" ]]; then
        echo "Error: --debug-node requires a node name." >&2
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PARTITION="$arg"
      shift
      ;;
  esac
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' is required but was not found. Run this script on the HPC login node." >&2
    exit 1
  fi
}

require_cmd sinfo
require_cmd scontrol

mb_to_gib() {
  local mb="${1:-0}"
  awk -v mb="$mb" 'BEGIN { printf "%.1fG", mb / 1024 }'
}

mem_to_mb() {
  local mem="${1:-0}"
  awk -v mem="$mem" '
    BEGIN {
      unit = substr(mem, length(mem), 1)
      value = mem + 0
      if (unit == "K") value = value / 1024
      else if (unit == "G") value = value * 1024
      else if (unit == "T") value = value * 1024 * 1024
      print int(value)
    }'
}

human_state() {
  local state="${1:-unknown}"
  echo "$state"
}

is_schedulable_state() {
  local state
  state="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"

  case "$state" in
    DOWN*|DRAIN*|DRAINED*|FAIL*|FAILING*|FUTURE*|MAINT*|NO_RESP*|POWER_DOWN*|*PLANNED*)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

gpu_mem_from_type() {
  local gpu_type="$1"
  local lower
  lower="$(printf '%s' "$gpu_type" | tr '[:upper:]' '[:lower:]')"

  if [[ "$lower" =~ ([0-9]+)[[:space:]_-]*g(b)? ]]; then
    echo "${BASH_REMATCH[1]}G"
  elif [[ "$lower" == *b200* ]]; then
    echo "180G"
  elif [[ "$lower" == *h200* ]]; then
    echo "141G"
  elif [[ "$lower" == *h100* ]]; then
    echo "80G"
  elif [[ "$lower" == *a100* ]]; then
    echo "40G/80G"
  elif [[ "$lower" == *a40* ]]; then
    echo "48G"
  elif [[ "$lower" == *a6000* ]]; then
    echo "48G"
  elif [[ "$lower" == *v100* ]]; then
    echo "16G/32G"
  elif [[ "$lower" == *t4* ]]; then
    echo "16G"
  else
    echo "unknown"
  fi
}

parse_field() {
  local line="$1"
  local key="$2"
  tr ' ' '\n' <<<"$line" | awk -v key="$key" '
    index($0, key "=") == 1 {
      print substr($0, length(key) + 2)
      exit
    }'
}

parse_tres_value() {
  local tres="$1"
  local key="$2"
  local item

  [[ -z "$tres" || "$tres" == "(null)" || "$tres" == "N/A" ]] && return

  IFS=',' read -ra tres_items <<<"$tres"
  for item in "${tres_items[@]}"; do
    if [[ "$item" == "$key="* ]]; then
      echo "${item#*=}"
      return
    fi
  done
}

parse_tres_gpu_count() {
  local tres="$1"
  local item value
  local total=0
  local found=0

  [[ -z "$tres" || "$tres" == "(null)" || "$tres" == "N/A" ]] && return

  IFS=',' read -ra tres_items <<<"$tres"
  for item in "${tres_items[@]}"; do
    if [[ "$item" == gres/gpu=* || "$item" == gres/gpu:* ]]; then
      value="${item##*=}"
      if [[ "$value" =~ ^[0-9]+$ ]]; then
        total=$(( total + value ))
        found=1
      fi
    fi
  done

  if [[ "$found" -eq 1 ]]; then
    echo "$total"
  fi
}

gpu_label_from_features() {
  local features="$1"
  local lower
  lower="$(printf '%s' "$features" | tr '[:upper:]' '[:lower:]')"

  if [[ "$lower" == *h200* ]]; then
    echo "h200"
  elif [[ "$lower" == *b200* ]]; then
    echo "b200"
  elif [[ "$lower" == *h100* ]]; then
    echo "h100"
  elif [[ "$lower" == *a100_80g* || "$lower" == *a100-80g* ]]; then
    echo "a100_80g"
  elif [[ "$lower" == *a100_40g* || "$lower" == *a100-40g* ]]; then
    echo "a100_40g"
  elif [[ "$lower" == *a100* ]]; then
    echo "a100"
  elif [[ "$lower" == *a40* ]]; then
    echo "a40"
  elif [[ "$lower" == *a6000* ]]; then
    echo "a6000"
  elif [[ "$lower" == *v100* ]]; then
    echo "v100"
  elif [[ "$lower" == *t4* ]]; then
    echo "t4"
  else
    echo "gpu"
  fi
}

gpu_counts_from_gres() {
  local gres="$1"
  local -n out_ref="$2"
  local item gpu_type count

  [[ -z "$gres" || "$gres" == "(null)" || "$gres" == "N/A" ]] && return

  IFS=',' read -ra gres_items <<<"$gres"
  for item in "${gres_items[@]}"; do
    item="${item%%(*}"
    [[ "$item" != gpu:* ]] && continue

    IFS=':' read -ra parts <<<"$item"
    if [[ "${#parts[@]}" -ge 3 ]]; then
      gpu_type="${parts[1]}"
      count="${parts[2]}"
    elif [[ "${#parts[@]}" -eq 2 ]]; then
      gpu_type="gpu"
      count="${parts[1]}"
    else
      continue
    fi

    [[ "$count" =~ ^[0-9]+$ ]] || continue
    out_ref["$gpu_type"]=$(( ${out_ref["$gpu_type"]:-0} + count ))
  done
}

print_header() {
  printf '%-18s %-24s %-12s %8s %10s %12s %14s %-18s %-12s\n' \
    "PARTITION" "NODE" "STATE" "FREE_GPU" "FREE_CPU" "FREE_MEM" "TOTAL_MEM" "GPU_TYPE" "GPU_MEM"
}

print_separator() {
  printf '%-18s %-24s %-12s %8s %10s %12s %14s %-18s %-12s\n' \
    "------------------" "------------------------" "------------" "--------" "----------" "------------" "--------------" "------------------" "------------"
}

if [[ -n "$PARTITION" ]]; then
  mapfile -t NODE_ROWS < <(sinfo -h -p "$PARTITION" -N -o "%P|%N" | sed 's/\*|/|/' | sort -u)
else
  mapfile -t NODE_ROWS < <(sinfo -h -N -o "%P|%N" | sed 's/\*|/|/' | sort -u)
fi

if [[ "${#NODE_ROWS[@]}" -eq 0 ]]; then
  if [[ -n "$PARTITION" ]]; then
    echo "No nodes found in partition '$PARTITION'." >&2
  else
    echo "No nodes found." >&2
  fi
  exit 1
fi

print_header
print_separator

found=0

for row in "${NODE_ROWS[@]}"; do
  partition="${row%%|*}"
  node="${row#*|}"
  node_info="$(scontrol show node "$node")"

  raw_state="$(parse_field "$node_info" "State")"
  state="$(human_state "$raw_state")"
  cpu_total="$(parse_field "$node_info" "CPUTot")"
  cpu_alloc="$(parse_field "$node_info" "CPUAlloc")"
  real_mem="$(parse_field "$node_info" "RealMemory")"
  gres="$(parse_field "$node_info" "Gres")"
  gres_used="$(parse_field "$node_info" "GresUsed")"
  cfg_tres="$(parse_field "$node_info" "CfgTRES")"
  alloc_tres="$(parse_field "$node_info" "AllocTRES")"
  active_features="$(parse_field "$node_info" "ActiveFeatures")"
  available_features="$(parse_field "$node_info" "AvailableFeatures")"

  cpu_total="$(parse_tres_value "$cfg_tres" "cpu" || true)"
  cpu_alloc="$(parse_tres_value "$alloc_tres" "cpu" || true)"
  mem_total="$(parse_tres_value "$cfg_tres" "mem" || true)"
  mem_alloc="$(parse_tres_value "$alloc_tres" "mem" || true)"
  gpu_total_tres="$(parse_tres_gpu_count "$cfg_tres" || true)"
  gpu_alloc_tres="$(parse_tres_gpu_count "$alloc_tres" || true)"

  cpu_total="${cpu_total:-$(parse_field "$node_info" "CPUTot")}"
  cpu_alloc="${cpu_alloc:-$(parse_field "$node_info" "CPUAlloc")}"
  cpu_total="${cpu_total:-0}"
  cpu_alloc="${cpu_alloc:-0}"
  real_mem="${real_mem:-0}"
  mem_total_mb="$(mem_to_mb "${mem_total:-${real_mem}M}")"
  mem_alloc_mb="$(mem_to_mb "${mem_alloc:-0}")"

  free_cpu=$(( cpu_total - cpu_alloc ))
  [[ "$free_cpu" -lt 0 ]] && free_cpu=0
  free_mem_mb=$(( mem_total_mb - mem_alloc_mb ))
  [[ "$free_mem_mb" -lt 0 ]] && free_mem_mb=0

  declare -A gpu_total=()
  declare -A gpu_used=()
  if [[ -n "${gpu_total_tres:-}" ]]; then
    feature_label="$(gpu_label_from_features "${active_features:-${available_features:-}}")"
    gpu_total["$feature_label"]="$gpu_total_tres"
    gpu_used["$feature_label"]="${gpu_alloc_tres:-0}"
  else
    gpu_counts_from_gres "$gres" gpu_total
    gpu_counts_from_gres "$gres_used" gpu_used
  fi

  node_free_gpu=0
  gpu_summary=()
  gpu_mem_summary=()

  for gpu_type in "${!gpu_total[@]}"; do
    total="${gpu_total[$gpu_type]}"
    used="${gpu_used[$gpu_type]:-0}"
    free=$(( total - used ))
    [[ "$free" -lt 0 ]] && free=0
    node_free_gpu=$(( node_free_gpu + free ))
    gpu_summary+=("${gpu_type}:${free}/${total}")
    gpu_mem_summary+=("${gpu_type}:$(gpu_mem_from_type "$gpu_type")")
  done

  gpu_type_text="$(IFS=,; echo "${gpu_summary[*]:-none}")"
  gpu_mem_text="$(IFS=,; echo "${gpu_mem_summary[*]:-unknown}")"
  has_gpu=1
  if [[ "$gpu_type_text" == "none" ]]; then
    has_gpu=0
  fi

  skip_reasons=()
  if [[ "$has_gpu" -eq 0 ]]; then
    skip_reasons+=("no_gpu")
  fi
  if [[ "$node_free_gpu" -le 0 ]]; then
    skip_reasons+=("no_free_gpu")
  fi
  if [[ "$free_cpu" -le 0 ]]; then
    skip_reasons+=("no_free_cpu")
  fi
  if ! is_schedulable_state "$state"; then
    skip_reasons+=("not_schedulable_state")
  fi
  skip_reason_text="$(IFS=,; echo "${skip_reasons[*]:-none}")"

  if [[ "$node" == "$DEBUG_NODE" ]]; then
    {
      echo "DEBUG partition=$partition node=$node"
      echo "  state=$state"
      echo "  CPUTot(raw)=$(parse_field "$node_info" "CPUTot")"
      echo "  CPUAlloc(raw)=$(parse_field "$node_info" "CPUAlloc")"
      echo "  RealMemory(raw)=$real_mem"
      echo "  Gres(raw)=$gres"
      echo "  GresUsed(raw)=$gres_used"
      echo "  ActiveFeatures(raw)=$active_features"
      echo "  AvailableFeatures(raw)=$available_features"
      echo "  CfgTRES(raw)=$cfg_tres"
      echo "  AllocTRES(raw)=$alloc_tres"
      echo "  parsed cpu_total=$cpu_total"
      echo "  parsed cpu_alloc=$cpu_alloc"
      echo "  parsed free_cpu=$free_cpu"
      echo "  parsed mem_total=${mem_total:-${real_mem}M}"
      echo "  parsed mem_alloc=${mem_alloc:-0}"
      echo "  parsed free_mem=$(mb_to_gib "$free_mem_mb")"
      echo "  parsed gpu_total=${gpu_total_tres:-from_gres}"
      echo "  parsed gpu_alloc=${gpu_alloc_tres:-from_gres}"
      echo "  computed gpu_summary=$gpu_type_text"
      echo "  computed free_gpu=$node_free_gpu"
      echo "  default_skip_reasons=$skip_reason_text"
    } >&2
  fi

  if [[ "$SHOW_ALL" -eq 0 ]]; then
    if [[ "${#skip_reasons[@]}" -gt 0 ]]; then
      continue
    fi
  elif [[ "$has_gpu" -eq 0 ]]; then
    continue
  fi

  found=1

  printf '%-18s %-24s %-12s %8s %10s %12s %14s %-18s %-12s\n' \
    "$partition" \
    "$node" \
    "$state" \
    "$node_free_gpu" \
    "$free_cpu/$cpu_total" \
    "$(mb_to_gib "$free_mem_mb")" \
    "$(mb_to_gib "$mem_total_mb")" \
    "$gpu_type_text" \
    "$gpu_mem_text"

  unset gpu_total
  unset gpu_used
  unset skip_reasons
done

if [[ "$found" -eq 0 ]]; then
  if [[ -n "$PARTITION" ]]; then
    echo "No nodes with free GPUs found in partition '$PARTITION'. Use --all to show busy nodes too."
  else
    echo "No nodes with free GPUs found in any partition. Use --all to show busy GPU nodes too."
  fi
fi
