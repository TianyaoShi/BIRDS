from __future__ import annotations

from .models import GPULease, PortReservation


class ResourceUnavailableError(RuntimeError):
    pass


class GPULeaseManager:
    def __init__(self, *, allowed_gpu_ids: tuple[int, ...], max_active_gpus: int) -> None:
        if max_active_gpus > len(allowed_gpu_ids):
            raise ValueError("max_active_gpus cannot exceed available GPU ids")
        self._usable_gpu_ids = tuple(allowed_gpu_ids[:max_active_gpus])
        self._free_gpu_ids = list(self._usable_gpu_ids)
        self._leased_gpu_ids: set[int] = set()

    def acquire(self, gpu_count: int = 1) -> GPULease:
        if gpu_count <= 0:
            raise ValueError("gpu_count must be positive")
        if len(self._free_gpu_ids) < gpu_count:
            raise ResourceUnavailableError(
                f"not enough free GPU leases available: requested={gpu_count}, free={len(self._free_gpu_ids)}"
            )
        gpu_ids = tuple(self._free_gpu_ids[:gpu_count])
        del self._free_gpu_ids[:gpu_count]
        self._leased_gpu_ids.update(gpu_ids)
        return GPULease(gpu_ids=gpu_ids)

    def release(self, lease: GPULease) -> None:
        for gpu_id in lease.gpu_ids:
            if gpu_id not in self._leased_gpu_ids:
                raise ValueError(f"GPU id is not currently leased: {gpu_id}")
        for gpu_id in lease.gpu_ids:
            self._leased_gpu_ids.remove(gpu_id)
            self._free_gpu_ids.append(gpu_id)
        self._free_gpu_ids.sort()

    def snapshot(self) -> dict[str, object]:
        return {
            "usable_gpu_ids": list(self._usable_gpu_ids),
            "free_gpu_ids": list(self._free_gpu_ids),
            "leased_gpu_ids": sorted(self._leased_gpu_ids),
        }


class PortAllocator:
    def __init__(self, *, base_port_start: int, base_port_end: int, metrics_port_offset: int) -> None:
        if base_port_start <= 0 or base_port_end <= 0:
            raise ValueError("base ports must be positive")
        if base_port_end < base_port_start:
            raise ValueError("base_port_end must be >= base_port_start")
        if metrics_port_offset <= 0:
            raise ValueError("metrics_port_offset must be positive")
        self._base_port_start = base_port_start
        self._base_port_end = base_port_end
        self._metrics_port_offset = metrics_port_offset
        self._free_base_ports = list(range(base_port_start, base_port_end + 1))
        self._used_base_ports: set[int] = set()

    def reserve(self) -> PortReservation:
        if not self._free_base_ports:
            raise ResourceUnavailableError("no free base ports available")
        base_port = self._free_base_ports.pop(0)
        self._used_base_ports.add(base_port)
        return PortReservation(
            base_port=base_port,
            metrics_port=base_port + self._metrics_port_offset,
        )

    def release(self, reservation: PortReservation) -> None:
        if reservation.base_port not in self._used_base_ports:
            raise ValueError(f"base port is not currently reserved: {reservation.base_port}")
        self._used_base_ports.remove(reservation.base_port)
        self._free_base_ports.append(reservation.base_port)
        self._free_base_ports.sort()

    def snapshot(self) -> dict[str, object]:
        return {
            "base_port_start": self._base_port_start,
            "base_port_end": self._base_port_end,
            "metrics_port_offset": self._metrics_port_offset,
            "free_base_ports": list(self._free_base_ports),
            "used_base_ports": sorted(self._used_base_ports),
        }
