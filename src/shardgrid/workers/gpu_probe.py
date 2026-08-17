"""GPU probe for Windows GPU Workers (T041).

The probe runs a Torch + nvidia-smi evidence script INSIDE the WSL2 selected
Conda training environment through the T040 runtime wrapper, parses the real
results, and writes them into the existing ``WorkerResource`` / ``WorkerRuntime``
models.  GPU identity always comes from the real probe output, never from
configuration.  Partial failures keep their real reasons and are never filled
with fake values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from shardgrid.common.config import WorkerConfig
from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_hostname
from shardgrid.resources.models import WorkerResource
from shardgrid.transport.runtime import WSLRuntimeWrapper
from shardgrid.workers.models import WorkerRuntime
from shardgrid.workers.probe import ProbeFailure

_PROBE_SCRIPT = (
    "import json,subprocess\n"
    "out={'torch':None,'nvidia_smi':None}\n"
    "try:\n"
    " import torch\n"
    " t={'version':torch.__version__,'cuda_version':str(torch.version.cuda),"
    "'available':torch.cuda.is_available(),'device_count':torch.cuda.device_count()}\n"
    " if t['available'] and t['device_count']>0:\n"
    "  d=torch.cuda.current_device();p=torch.cuda.get_device_properties(d)\n"
    "  t.update({'name':p.name,'total_memory_mb':p.total_memory//1048576,"
    "'compute_capability':'.'.join(str(x) for x in torch.cuda.get_device_capability(d))})\n"
    "  try:t['free_memory_mb']=torch.cuda.mem_get_info(d)[0]//1048576\n"
    "  except Exception:t['free_memory_mb']=None\n"
    "  try:t['utilization_percent']=torch.cuda.utilization() "
    "if hasattr(torch.cuda,'utilization') else None\n"
    "  except Exception:t['utilization_percent']=None\n"
    "  try:t['nccl']='.'.join(str(x) for x in torch.cuda.nccl.version())\n"
    "  except Exception:t['nccl']=None\n"
    "  try:\n"
    "   import torch.distributed as dist;t['gloo']=bool(dist.is_gloo_available())\n"
    "  except Exception:t['gloo']=None\n"
    " out['torch']=t\n"
    "except Exception as exc:out['torch']={'error':str(exc)}\n"
    "try:\n"
    " r=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free,"
    "utilization.gpu,driver_version,compute_cap','--format=csv,noheader,nounits'],"
    "capture_output=True,text=True,timeout=30)\n"
    " if r.returncode==0:\n"
    "  q=[x.strip() for x in r.stdout.strip().split(',')]\n"
    "  if len(q)>=6:out['nvidia_smi']={'name':q[0],'memory_total_mb':int(q[1]),"
    "'memory_free_mb':int(q[2]),'utilization_percent':int(q[3]),"
    "'driver_version':q[4],'compute_capability':q[5]}\n"
    "  else:out['nvidia_smi']={'error':'unexpected nvidia-smi output'}\n"
    " else:out['nvidia_smi']={'error':(r.stderr or 'nvidia-smi failed').strip()}\n"
    "except Exception as exc:out['nvidia_smi']={'error':str(exc)}\n"
    "print(json.dumps(out))\n"
)


@dataclass(frozen=True)
class GPUProbeResult:
    worker_resource: WorkerResource
    worker_runtime: WorkerRuntime
    failures: tuple[ProbeFailure, ...]
    health: Health
    probe_status: str
    raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_resource": self.worker_resource.to_dict(),
            "worker_runtime": self.worker_runtime.to_dict(),
            "failures": [failure.to_dict() for failure in self.failures],
            "health": self.health.value,
            "probe_status": self.probe_status,
        }


def _parse_probe_payload(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except ValueError:
                continue
            if isinstance(payload, dict) and "torch" in payload:
                return payload
    return None


def _select_gpu_name(torch_gpu: dict[str, Any] | None, smi: dict[str, Any] | None) -> str | None:
    if torch_gpu and torch_gpu.get("name"):
        return str(torch_gpu["name"])
    if smi and smi.get("name"):
        return str(smi["name"])
    return None


def probe_gpu(
    wrapper: WSLRuntimeWrapper,
    worker: WorkerConfig,
    *,
    probe_status: str = "live",
    timeout: float = 120.0,
) -> GPUProbeResult:
    failures: list[ProbeFailure] = []
    health = Health.HEALTHY

    result = wrapper.run_script(_PROBE_SCRIPT, timeout=timeout)
    payload = _parse_probe_payload(result.stdout) if result.ok else None
    if payload is None:
        failures.append(
            ProbeFailure(
                layer="wsl_runtime",
                check="gpu_probe_script",
                message="GPU probe script did not produce a parseable result",
                exit_code=result.exit_code,
                output=(result.stderr or result.stdout)[:500] or None,
            )
        )
        health = Health.FAILED

    torch_data = payload.get("torch") if payload else None
    smi = payload.get("nvidia_smi") if payload else None

    torch_error = None
    if isinstance(torch_data, dict) and torch_data.get("error"):
        torch_error = str(torch_data["error"])
        failures.append(
            ProbeFailure(
                layer="wsl_runtime",
                check="torch",
                message="PyTorch is not usable in the selected Conda environment",
                output=torch_error,
            )
        )
        health = Health.FAILED

    if isinstance(torch_data, dict) and not torch_data.get("error"):
        if torch_data.get("available") is not True:
            failures.append(
                ProbeFailure(
                    layer="wsl_runtime",
                    check="cuda",
                    message=(
                        "CUDA is not available from the selected Conda environment "
                        "(torch.cuda.is_available() is False)"
                    ),
                )
            )
            health = Health.FAILED
        if torch_data.get("nccl") is None:
            failures.append(
                ProbeFailure(
                    layer="wsl_runtime",
                    check="nccl",
                    message="NCCL is not available in the selected Conda environment",
                )
            )
            health = Health.FAILED

    smi_error = None
    if isinstance(smi, dict) and smi.get("error"):
        smi_error = str(smi["error"])
        failures.append(
            ProbeFailure(
                layer="wsl_runtime",
                check="nvidia_smi",
                message="nvidia-smi is not usable inside the WSL2 runtime",
                output=smi_error,
            )
        )
        health = Health.FAILED

    gpu_name = _select_gpu_name(torch_data, smi)
    if gpu_name is None:
        failures.append(
            ProbeFailure(
                layer="wsl_runtime",
                check="gpu",
                message="No GPU is visible from the selected Conda environment",
            )
        )
        health = Health.FAILED

    total_memory = None
    free_memory = None
    utilization = None
    compute_capability = None
    driver_version = None
    if isinstance(smi, dict) and not smi.get("error"):
        total_memory = smi.get("memory_total_mb")
        free_memory = smi.get("memory_free_mb")
        utilization = smi.get("utilization_percent")
        compute_capability = smi.get("compute_capability")
        driver_version = smi.get("driver_version")
    if isinstance(torch_data, dict) and not torch_data.get("error"):
        if total_memory is None:
            total_memory = torch_data.get("total_memory_mb")
        if free_memory is None:
            free_memory = torch_data.get("free_memory_mb")
        if utilization is None:
            utilization = torch_data.get("utilization_percent")
        if compute_capability is None:
            compute_capability = torch_data.get("compute_capability")

    torch_version = None
    torch_cuda_version = None
    cuda_available = False
    nccl_available = False
    gloo_available = False
    if isinstance(torch_data, dict) and not torch_data.get("error"):
        torch_version = torch_data.get("version")
        torch_cuda_version = torch_data.get("cuda_version")
        cuda_available = torch_data.get("available") is True
        nccl_available = torch_data.get("nccl") is not None
        gloo_available = torch_data.get("gloo") is True

    python_executable = None
    if wrapper.config.conda_prefix:
        python_executable = f"{wrapper.config.conda_prefix}/bin/python"

    worker_runtime = WorkerRuntime(
        worker_id=worker.worker_id,
        runtime_os=RuntimeOS.WSL2_LINUX,
        runtime_version=wrapper.config.distro,
        environment_manager="conda",
        conda_environment=wrapper.config.conda_environment,
        conda_prefix=wrapper.config.conda_prefix,
        conda_active=True,
        python_executable=python_executable,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        nccl_available=nccl_available,
        gloo_available=gloo_available,
        path_style="posix",
        health=health,
    )
    worker_resource = WorkerResource(
        worker_id=worker.worker_id,
        hostname=as_hostname(worker.host),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        conda_environment=wrapper.config.conda_environment,
        conda_prefix=wrapper.config.conda_prefix,
        python_executable=python_executable,
        gpu_name=gpu_name,
        gpu_total_memory=total_memory,
        gpu_free_memory=free_memory,
        gpu_utilization=utilization,
        compute_capability=compute_capability,
        driver_version=driver_version,
        cuda_version=torch_cuda_version,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        nccl_available=nccl_available,
        gloo_available=gloo_available,
        health=health,
    )

    return GPUProbeResult(
        worker_resource=worker_resource,
        worker_runtime=worker_runtime,
        failures=tuple(failures),
        health=health,
        probe_status=probe_status,
        raw_output=result.stdout if result.ok else "",
    )