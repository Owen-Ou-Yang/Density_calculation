#!/usr/bin/env bash
set -euo pipefail

# Dedicated 3/4-GPU density comparison launcher. The established 2-GPU
# launcher remains unchanged; the caller must choose the requested count.
case "${MACE_GPU_COUNT:-}" in
  3|4) gpu_count="${MACE_GPU_COUNT}" ;;
  *) printf '%s\n' 'MACE_GPU_COUNT must explicitly be 3 or 4' >&2; exit 64 ;;
esac
if [[ "${MACE_THREADS_PER_RANK:-4}" != 4 ]]; then
  printf '%s\n' 'This density comparison requires exactly 4 threads per rank' >&2
  exit 64
fi
threads=4

export CUDA_HOME="${CUDA_HOME:?set CUDA_HOME}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export IMPI_MPIRUN="${IMPI_MPIRUN:?set absolute Intel MPI mpirun path}"
export LAMMPS_RTX6K_IMPI="${LAMMPS_RTX6K_IMPI:?set absolute LAMMPS build directory}"
export I_MPI_OFFLOAD=1
export I_MPI_OFFLOAD_MODE=cuda
export I_MPI_OFFLOAD_CUDA_LIBRARY="${I_MPI_OFFLOAD_CUDA_LIBRARY:-/usr/lib64/libcuda.so.1}"
export I_MPI_OFFLOAD_IPC=0

task_gpu_allocation="${SGE_HGR_TASK_gpu_card:-}"
host_gpu_allocation="${SGE_HGR_gpu_card:-}"
if [[ -n "${task_gpu_allocation}" && -n "${host_gpu_allocation}" \
   && "${task_gpu_allocation}" != "${host_gpu_allocation}" ]]; then
  printf 'CRC SGE gpu_card allocation channels disagree: task=%q host=%q\n' \
    "${task_gpu_allocation}" "${host_gpu_allocation}" >&2
  exit 66
fi
scheduler_gpu_allocation="${task_gpu_allocation:-${host_gpu_allocation}}"
if [[ -z "${scheduler_gpu_allocation}" ]]; then
  printf '%s\n' 'CRC SGE gpu_card allocation is missing' >&2
  exit 66
fi
read -r -a scheduler_gpu_tokens <<< "${scheduler_gpu_allocation//,/ }"
if [[ "${#scheduler_gpu_tokens[@]}" -ne "${gpu_count}" ]]; then
  printf 'CRC SGE allocation count must equal MACE_GPU_COUNT=%s; got: %q\n' \
    "${gpu_count}" "${scheduler_gpu_allocation}" >&2
  exit 66
fi
allocated_minors=()
seen_gpu_devices=" "
for token in "${scheduler_gpu_tokens[@]}"; do
  # Canonical gpuN syntax avoids ambiguous aliases such as gpu01/gpu1.
  if [[ ! "${token}" =~ ^gpu(0|[1-9][0-9]*)$ ]]; then
    printf 'CRC SGE allocation token is not canonical gpuN syntax: %q\n' "${token}" >&2
    exit 66
  fi
  device="${token#gpu}"
  if [[ "${seen_gpu_devices}" == *" ${device} "* ]]; then
    printf 'CRC SGE gpu_card allocation contains duplicate device: %s\n' "${token}" >&2
    exit 66
  fi
  seen_gpu_devices+="${device} "
  allocated_minors+=("${device}")
done

# SGE gpuN identifies the host device minor, not necessarily its job-visible
# CUDA/NVML ordinal. In a device-restricted job, host 0/2/3 can be enumerated
# as 0/1/2. Resolve each allocated minor to its stable UUID before launching.
nvidia_smi="${MACE_NVIDIA_SMI:-/usr/bin/nvidia-smi}"
if [[ "${nvidia_smi}" != /* || ! -x "${nvidia_smi}" ]]; then
  printf '%s\n' 'MACE_NVIDIA_SMI must be an absolute executable path' >&2
  exit 66
fi
inventory_python="${THERMAL_PYTHON:?set absolute Python executable}"
if [[ "${inventory_python}" != /* || ! -x "${inventory_python}" ]]; then
  printf '%s\n' 'THERMAL_PYTHON must be an absolute executable path' >&2
  exit 66
fi
# CRC's installed nvidia-smi reports minor_number in -q -x XML, but rejects
# minor_number as a selective CSV query field. Do not retry that rejected API.
if gpu_inventory_xml="$("${nvidia_smi}" -q -x)"; then
  inventory_returncode=0
else
  inventory_returncode=$?
fi
printf 'MACE_THERMAL_GPU_INVENTORY_XML executable=%s argv=-q,-x returncode=%s\n%s\nMACE_THERMAL_GPU_INVENTORY_XML_END\n' \
  "${nvidia_smi}" "${inventory_returncode}" "${gpu_inventory_xml}" >&2
if [[ "${inventory_returncode}" -ne 0 ]]; then
  printf '%s\n' 'CRC GPU inventory query failed; no MPI/model process started' >&2
  exit 66
fi
# stdlib-only XML parsing. Entry ordinal is diagnostic XML ordering only:
# it is not a physical GPU index or a CUDA device selector.
if ! gpu_inventory="$("${inventory_python}" -I -B -c '
import csv
import sys
import xml.etree.ElementTree as ET
try:
    root = ET.fromstring(sys.stdin.read())
    if root.tag != "nvidia_smi_log":
        raise ValueError("unexpected XML root")
    records = []
    for ordinal, gpu in enumerate(root.findall("gpu")):
        values = []
        for field in ("uuid", "minor_number", "pci/pci_bus_id"):
            nodes = gpu.findall(field)
            if len(nodes) != 1 or not (nodes[0].text or "").strip():
                raise ValueError("missing or duplicate XML field: " + field)
            values.append(nodes[0].text.strip())
        records.append((ordinal, *values))
    csv.writer(sys.stdout, lineterminator="\n").writerows(records)
except (ET.ParseError, ValueError) as exc:
    print("CRC GPU inventory XML parse failed: " + str(exc), file=sys.stderr)
    raise SystemExit(66)
' <<< "${gpu_inventory_xml}")"; then
  printf '%s\n' 'CRC GPU inventory XML validation failed; no MPI/model process started' >&2
  exit 66
fi
inventory_entries=()
inventory_uuids=()
inventory_minors=()
inventory_pci=()
seen_entries=' '
seen_uuids=' '
seen_minors=' '
seen_pci=' '
while IFS= read -r line; do
  [[ -n "${line//[[:space:]]/}" ]] || continue
  without_commas="${line//,/}"
  if [[ $((${#line} - ${#without_commas})) -ne 3 ]]; then
    printf 'CRC GPU inventory has malformed CSV row: %q\n' "${line}" >&2
    exit 66
  fi
  IFS=, read -r xml_entry_ordinal gpu_uuid minor pci_bus <<< "${line}"
  fields=("${xml_entry_ordinal}" "${gpu_uuid}" "${minor}" "${pci_bus}")
  for field_index in "${!fields[@]}"; do
    value="${fields[field_index]}"
    value="${value#"${value%%[![:space:]]*}"}"
    fields[field_index]="${value%"${value##*[![:space:]]}"}"
  done
  xml_entry_ordinal="${fields[0]}"
  gpu_uuid="${fields[1]}"
  minor="${fields[2]}"
  pci_bus="${fields[3]}"
  if [[ ! "${xml_entry_ordinal}" =~ ^(0|[1-9][0-9]*)$ \
     || ! "${minor}" =~ ^(0|[1-9][0-9]*)$ \
     || ! "${gpu_uuid}" =~ ^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ \
     || ! "${pci_bus}" =~ ^[0-9a-fA-F]{4}([0-9a-fA-F]{4})?:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$ ]]; then
    printf 'CRC GPU inventory has invalid identity fields: %q\n' "${line}" >&2
    exit 66
  fi
  if [[ "${seen_entries}" == *" ${xml_entry_ordinal} "* \
     || "${seen_uuids}" == *" ${gpu_uuid} "* \
     || "${seen_minors}" == *" ${minor} "* \
     || "${seen_pci}" == *" ${pci_bus} "* ]]; then
    printf 'CRC GPU inventory has duplicate/ambiguous identity: %q\n' "${line}" >&2
    exit 66
  fi
  seen_entries+="${xml_entry_ordinal} "
  seen_uuids+="${gpu_uuid} "
  seen_minors+="${minor} "
  seen_pci+="${pci_bus} "
  inventory_entries+=("${xml_entry_ordinal}")
  inventory_uuids+=("${gpu_uuid}")
  inventory_minors+=("${minor}")
  inventory_pci+=("${pci_bus}")
done <<< "${gpu_inventory}"
allocated_gpus=()
for index in "${!allocated_minors[@]}"; do
  match=''
  for inventory_index in "${!inventory_minors[@]}"; do
    if [[ "${inventory_minors[inventory_index]}" == "${allocated_minors[index]}" ]]; then
      match="${inventory_index}"
    fi
  done
  if [[ -z "${match}" ]]; then
    printf 'CRC allocated device has no unique inventory mapping: %s\n' \
      "${scheduler_gpu_tokens[index]}" >&2
    exit 66
  fi
  allocated_gpus+=("${inventory_uuids[match]}")
  printf '%s\n' \
    "MACE_THERMAL_GPU_MAPPING scheduler_token=${scheduler_gpu_tokens[index]} minor=${inventory_minors[match]} xml_entry_ordinal=${inventory_entries[match]} uuid=${inventory_uuids[match]} pci_bus_id=${inventory_pci[match]} logical_cuda_device=0" >&2
done

# Do not let a login-shell GPU mask or a higher-priority allocator alias
# override the allocation and allocator used by this comparison.
unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
unset PYTORCH_ALLOC_CONF PYTORCH_NO_CUDA_MEMORY_CACHING
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${threads}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp
export I_MPI_PIN_ORDER=spread

mace_conda_prefix="${MACE_CONDA_PREFIX:?set runtime environment prefix}"
conda_lib="${mace_conda_prefix}/lib"
export LD_LIBRARY_PATH="${LAMMPS_RTX6K_IMPI}:${CUDA_HOME}/lib64:${conda_lib}:${IMPI_LIBRARY_PATH:?set colon-separated Intel MPI library directories}:${LD_LIBRARY_PATH:-}"

# Preserve the actual 2-GPU thermal default and the same validated override.
skin="${MACE_NEIGH_SKIN-2.0}"
if [[ ! "${skin}" =~ ^[+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] ||
   ! LC_ALL=C awk -v value="${skin}" 'BEGIN {
     number = value + 0
     exit !(number >= 0 && tolower(sprintf("%.17g", number)) !~ /inf|nan/)
   }'; then
  printf '%s\n' 'MACE_NEIGH_SKIN must be a finite nonnegative number in Angstrom' >&2
  exit 64
fi
lmp=(
  "${LAMMPS_RTX6K_IMPI}/lmp"
  -var mace_neigh_skin "${skin}"
  -var thermal_mace_neigh_skin "${skin}"
  -k on t "${threads}" g 1
  -sf kk
  -pk kokkos newton on neigh half comm device gpu/aware on
)
mpi_args=(
  -genv I_MPI_OFFLOAD "${I_MPI_OFFLOAD}"
  -genv I_MPI_OFFLOAD_MODE "${I_MPI_OFFLOAD_MODE}"
  -genv I_MPI_OFFLOAD_CUDA_LIBRARY "${I_MPI_OFFLOAD_CUDA_LIBRARY}"
  -genv I_MPI_OFFLOAD_IPC "${I_MPI_OFFLOAD_IPC}"
  -genv OMP_NUM_THREADS "${OMP_NUM_THREADS}"
  -genv OMP_PROC_BIND "${OMP_PROC_BIND}"
  -genv OMP_PLACES "${OMP_PLACES}"
  -genv I_MPI_PIN_DOMAIN "${I_MPI_PIN_DOMAIN}"
  -genv I_MPI_PIN_ORDER "${I_MPI_PIN_ORDER}"
  -genv PYTORCH_CUDA_ALLOC_CONF "${PYTORCH_CUDA_ALLOC_CONF}"
)
for index in "${!scheduler_gpu_tokens[@]}"; do
  if [[ "${index}" -ne 0 ]]; then
    mpi_args+=(:)
  fi
  mpi_args+=(
    -np 1
    -env CUDA_VISIBLE_DEVICES "${allocated_gpus[index]}"
    -env THERMAL_SCHEDULER_GPU_TOKEN "${scheduler_gpu_tokens[index]}"
    -env THERMAL_LOGICAL_GPU_INDEX 0
    -env OMPI_COMM_WORLD_LOCAL_RANK 0
    -env MPI_LOCALRANKID 0
    "${lmp[@]}" "$@"
  )
done
printf '%s\n' \
  "MACE_THERMAL_3OR4GPU_LAUNCHER gpu_count=${gpu_count} mpi_ranks=${gpu_count} ipc=${I_MPI_OFFLOAD_IPC} threads_per_rank=${threads} scheduler_tokens=${scheduler_gpu_tokens[*]} cuda_devices=${allocated_gpus[*]} logical_cuda_device_per_rank=0 allocator=${PYTORCH_CUDA_ALLOC_CONF} neighbor_skin_A=${skin} pin_domain=${I_MPI_PIN_DOMAIN} pin_order=${I_MPI_PIN_ORDER} proc_bind=${OMP_PROC_BIND} places=${OMP_PLACES}" \
  >&2
exec "${IMPI_MPIRUN}" "${mpi_args[@]}"
