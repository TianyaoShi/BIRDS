import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlDeviceGetClockInfo, NVMLError


def _default_power_stats() -> Dict[str, float]:
    return {
        "min_power": 0,
        "power_5p": 0,
        "power_25p": 0,
        "median_power": 0,
        "power_75p": 0,
        "power_95p": 0,
        "max_power": 0,
        "power_std": 0,
        "dropped_power_samples": 0,
    }


@dataclass
class GPUMonitorSnapshot:
    avg_power_mw: float = 0.0
    avg_gpu_util: float = 0.0
    power_stats: Dict[str, float] = field(default_factory=_default_power_stats)
    power_trace_mw: List[int] = field(default_factory=list)
    clock_trace_mhz: Optional[List[Tuple[int, int, int]]] = None


class GPUMonitor:
    def __init__(self, gpu_id=0, interval=0.025, truncate=0, monitor_clock=False, max_samples=1000000):
        '''
        truncates the last `truncate` seconds of the monitoring data
        '''
        self.gpu_id = gpu_id
        self.done = False
        self.interval = interval
        self.thread = None
        self.truncate = truncate
        self.monitor_clock = monitor_clock
        self.max_samples = max_samples if max_samples and max_samples > 0 else None
        self._snapshot_lock = threading.Lock()
        self._snapshot_ready = threading.Event()
        self.latest_snapshot = GPUMonitorSnapshot()
        
        try:
            nvmlInit()
            self.gpu_handle = nvmlDeviceGetHandleByIndex(gpu_id)
        except NVMLError as e:
            print(f"NVML Error: {e}")
            self.gpu_handle = None

    def _reset_snapshot(self):
        with self._snapshot_lock:
            self.latest_snapshot = GPUMonitorSnapshot()
        self._snapshot_ready.clear()

    def _set_snapshot(self, snapshot: GPUMonitorSnapshot):
        with self._snapshot_lock:
            self.latest_snapshot = snapshot
        self._snapshot_ready.set()

    def get_snapshot(self, wait=False, timeout=None) -> GPUMonitorSnapshot:
        if wait:
            self._snapshot_ready.wait(timeout=timeout)

        with self._snapshot_lock:
            snapshot = self.latest_snapshot
            return GPUMonitorSnapshot(
                avg_power_mw=snapshot.avg_power_mw,
                avg_gpu_util=snapshot.avg_gpu_util,
                power_stats=dict(snapshot.power_stats),
                power_trace_mw=list(snapshot.power_trace_mw),
                clock_trace_mhz=(list(snapshot.clock_trace_mhz)
                                 if snapshot.clock_trace_mhz is not None else None),
            )

    def start(self):
        if self.gpu_handle is None:
            print("GPU handle not initialized. Monitoring cannot start.")
            return
        self.done = False
        self._reset_snapshot()
        self.thread = threading.Thread(target=self._monitor_gpu)
        self.thread.start()

    def stop(self):
        if self.thread:
            self.done = True
            self.thread.join(timeout=2.0)  # Add timeout to prevent hanging
            if self.thread.is_alive():
                print(f"Warning: GPU monitor thread for GPU {self.gpu_id} did not stop cleanly")
            self.thread = None

    def _monitor_gpu(self):
        gpu_power_readings = deque(maxlen=self.max_samples)
        dropped_power_samples = 0
        # gpu_utilization_readings = []
        # memory_utilization_readings = []
        if self.monitor_clock:
            gpu_clock_readings = deque(maxlen=self.max_samples)
            dropped_clock_samples = 0
        
        while not self.done:
            try:
                power = nvmlDeviceGetPowerUsage(self.gpu_handle)
            except NVMLError as e:
                print(f"NVML Error while reading power on GPU {self.gpu_id}: {e}")
                break
            # utilization = nvmlDeviceGetUtilizationRates(self.gpu_handle)
            # memory = nvmlDeviceGetMemoryInfo(self.gpu_handle)

            if gpu_power_readings.maxlen is not None and len(gpu_power_readings) == gpu_power_readings.maxlen:
                dropped_power_samples += 1
            gpu_power_readings.append(power)
            # gpu_utilization_readings.append(utilization.gpu)
            # DO NOT USE utilization.memory, it is not what we want
            # utilization.memory is "Percent of time over the past second in which any framebuffer memory has been read or stored."
            # memory_utilization_readings.append(memory.used / memory.total * 100)

            if self.monitor_clock:
                graphics_clock = nvmlDeviceGetClockInfo(self.gpu_handle, 0)
                sm_clock = nvmlDeviceGetClockInfo(self.gpu_handle, 1)
                memory_clock = nvmlDeviceGetClockInfo(self.gpu_handle, 2)
                if gpu_clock_readings.maxlen is not None and len(gpu_clock_readings) == gpu_clock_readings.maxlen:
                    dropped_clock_samples += 1
                gpu_clock_readings.append((graphics_clock, sm_clock, memory_clock))

            time.sleep(self.interval)

        gpu_power_readings = list(gpu_power_readings)
        if self.monitor_clock:
            gpu_clock_readings = list(gpu_clock_readings)

        if self.truncate > 0:
            seconds_to_truncate = int(self.truncate/self.interval)
            if seconds_to_truncate * 2 > len(gpu_power_readings):
                print(f"[Warning] Truncate value too high. This will lead to empty readings.")
            gpu_power_readings = gpu_power_readings[seconds_to_truncate:-seconds_to_truncate]
            # gpu_utilization_readings = gpu_utilization_readings[seconds_to_truncate:-seconds_to_truncate]
            # memory_utilization_readings = memory_utilization_readings[seconds_to_truncate:-seconds_to_truncate]

        avg_power = sum(gpu_power_readings) / len(gpu_power_readings) if gpu_power_readings else 0
        # avg_gpu_util = sum(gpu_utilization_readings) / len(gpu_utilization_readings) if gpu_utilization_readings else 0
        # avg_mem_util = sum(memory_utilization_readings) / len(memory_utilization_readings) if memory_utilization_readings else 0

        min_power = min(gpu_power_readings) if gpu_power_readings else 0
        power_5p = np.percentile(gpu_power_readings, 5) if gpu_power_readings else 0
        power_25p = np.percentile(gpu_power_readings, 25) if gpu_power_readings else 0
        median_power = np.median(gpu_power_readings) if gpu_power_readings else 0
        power_75p = np.percentile(gpu_power_readings, 75) if gpu_power_readings else 0
        power_95p = np.percentile(gpu_power_readings, 95) if gpu_power_readings else 0
        max_power = max(gpu_power_readings) if gpu_power_readings else 0
        power_std = np.std(np.array(gpu_power_readings)/1000) if gpu_power_readings else 0

        stats_payload = {
            "min_power": min_power,
            "power_5p": power_5p,
            "power_25p": power_25p,
            "median_power": median_power,
            "power_75p": power_75p,
            "power_95p": power_95p,
            "max_power": max_power,
            "power_std": power_std,
            "dropped_power_samples": dropped_power_samples,
        }
        if self.monitor_clock:
            stats_payload["dropped_clock_samples"] = dropped_clock_samples

        if dropped_power_samples > 0:
            print(
                f"[Warning] GPUMonitor for GPU {self.gpu_id} dropped {dropped_power_samples} "
                f"old power samples after reaching max_samples={self.max_samples}."
            )

        self._set_snapshot(
            GPUMonitorSnapshot(
                avg_power_mw=avg_power,
                avg_gpu_util=0.0,
                power_stats=stats_payload,
                power_trace_mw=gpu_power_readings,
                clock_trace_mhz=(gpu_clock_readings if self.monitor_clock else None),
            ))

    def __del__(self):
        try:
            if self.gpu_handle:
                self.stop()
                nvmlShutdown()
        except:
            pass  # Ignore errors during cleanup
