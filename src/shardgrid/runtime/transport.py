"""torch.distributed tensor transport for generic DAG edges."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TensorTransferEvidence:
    step: int
    value_id: str
    source_rank: int
    destination_rank: int
    direction: str
    shape: tuple[int, ...]
    dtype: str
    bytes: int
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass(frozen=True)
class PendingTensorSend:
    tensor: torch.Tensor
    work: Any
    evidence: TensorTransferEvidence

    def wait(self) -> TensorTransferEvidence:
        self.work.wait()
        return self.evidence


def tensor_tag(*, step: int, value_id: str, direction: str) -> int:
    payload = f"{step}:{value_id}:{direction}".encode("utf-8")
    return 1 + int.from_bytes(hashlib.sha1(payload).digest()[:3], "big")


def send_tensor(
    tensor: torch.Tensor,
    *,
    dst: int,
    step: int,
    value_id: str,
    direction: str,
) -> TensorTransferEvidence:
    tag = tensor_tag(step=step, value_id=value_id, direction=direction)
    dist.send(tensor.contiguous(), dst=dst, tag=tag)
    return _evidence(
        tensor,
        step=step,
        value_id=value_id,
        peer=dst,
        direction=direction,
        sending=True,
    )


def send_tensor_async(
    tensor: torch.Tensor,
    *,
    dst: int,
    step: int,
    value_id: str,
    direction: str,
) -> PendingTensorSend:
    payload = tensor.contiguous()
    tag = tensor_tag(step=step, value_id=value_id, direction=direction)
    work = dist.isend(payload, dst=dst, tag=tag)
    return PendingTensorSend(
        tensor=payload,
        work=work,
        evidence=_evidence(
            payload,
            step=step,
            value_id=value_id,
            peer=dst,
            direction=direction,
            sending=True,
        ),
    )


def recv_tensor(
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    src: int,
    device: torch.device,
    step: int,
    value_id: str,
    direction: str,
) -> tuple[torch.Tensor, TensorTransferEvidence]:
    tensor = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(tensor, src=src, tag=tensor_tag(step=step, value_id=value_id, direction=direction))
    return tensor, _evidence(
        tensor,
        step=step,
        value_id=value_id,
        peer=src,
        direction=direction,
        sending=False,
    )


def _evidence(
    tensor: torch.Tensor,
    *,
    step: int,
    value_id: str,
    peer: int,
    direction: str,
    sending: bool,
) -> TensorTransferEvidence:
    rank = dist.get_rank()
    return TensorTransferEvidence(
        step=step,
        value_id=value_id,
        source_rank=rank if sending else peer,
        destination_rank=peer if sending else rank,
        direction=direction,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype).replace("torch.", ""),
        bytes=tensor.numel() * tensor.element_size(),
        complete=True,
    )
