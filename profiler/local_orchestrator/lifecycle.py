from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .models import ActiveServer, ExpandedExperimentJob, LaunchConfig, PortReservation


ReadinessProbe = Callable[[str, str, float], bool]


class VLLMLifecycleError(RuntimeError):
    pass


class VLLMLifecycleManager:
    def __init__(
        self,
        *,
        process_factory: Callable[..., object] | None = None,
        readiness_probe: ReadinessProbe | None = None,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._process_factory = process_factory or _default_process_factory
        self._readiness_probe = readiness_probe or _default_readiness_probe
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._active_server: ActiveServer | None = None

    @property
    def active_server(self) -> ActiveServer | None:
        return self._active_server

    def ensure_server(
        self,
        *,
        job: ExpandedExperimentJob,
        gpu_ids: tuple[int, ...],
        ports: PortReservation,
        runtime_signature: str,
        logs_dir: Path,
        force_restart: bool = False,
    ) -> ActiveServer:
        if self._active_server is not None:
            if (
                not force_restart
                and self._active_server.runtime_signature == runtime_signature
                and self._process_is_running(self._active_server.process)
                and self.is_ready(self._active_server)
            ):
                return self._active_server
            self.stop_active_server(reason="signature_change_or_unhealthy")

        logs_dir.mkdir(parents=True, exist_ok=True)
        gpu_label = "-".join(str(gpu_id) for gpu_id in gpu_ids)
        log_suffix = f"gpu{gpu_label}_{runtime_signature[:12]}"
        stdout_log = logs_dir / f"vllm_{log_suffix}.stdout.log"
        stderr_log = logs_dir / f"vllm_{log_suffix}.stderr.log"
        stdout_handle = stdout_log.open("a", encoding="utf-8")
        stderr_handle = stderr_log.open("a", encoding="utf-8")

        command = render_launch_command(
            launch=job.launch,
            model=job.model,
            gpu_ids=gpu_ids,
            base_port=ports.base_port,
            metrics_port=ports.metrics_port,
        )
        env = os.environ.copy()
        env.update(job.launch.env)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        process = self._process_factory(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env=env,
        )

        base_url = f"http://{job.launch.host}:{ports.base_port}"
        server = ActiveServer(
            reuse_key=job.server_signature_key,
            runtime_signature=runtime_signature,
            model=job.model,
            endpoint=job.endpoint,
            gpu_id=gpu_ids[0],
            gpu_ids=gpu_ids,
            base_port=ports.base_port,
            metrics_port=ports.metrics_port,
            command=command,
            base_url=base_url,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            process=process,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
        )

        try:
            self._wait_until_ready(server, launch=job.launch)
        except Exception:
            self._terminate_server(server)
            raise

        self._active_server = server
        return server

    def is_ready(self, server: ActiveServer, *, timeout_s: float = 2.0) -> bool:
        if not self._process_is_running(server.process):
            return False
        return self._readiness_probe(server.base_url, "/v1/models", timeout_s)

    def stop_active_server(self, *, reason: str) -> None:
        del reason
        if self._active_server is None:
            return
        self._terminate_server(self._active_server)
        self._active_server = None

    def shutdown(self) -> None:
        self.stop_active_server(reason="shutdown")

    def _wait_until_ready(self, server: ActiveServer, *, launch: LaunchConfig) -> None:
        deadline = self._time_fn() + launch.readiness_timeout_s
        while self._time_fn() < deadline:
            if not self._process_is_running(server.process):
                raise VLLMLifecycleError(
                    "vLLM process exited before readiness: "
                    f"command={' '.join(server.command)}"
                )
            if self._readiness_probe(server.base_url, launch.readiness_path, 2.0):
                return
            remaining_s = deadline - self._time_fn()
            if remaining_s <= 0:
                break
            self._sleep_fn(min(launch.readiness_interval_s, remaining_s))
        raise VLLMLifecycleError(
            "timed out waiting for vLLM readiness at "
            f"{server.base_url}{launch.readiness_path}"
        )

    @staticmethod
    def _process_is_running(process: object) -> bool:
        poll_fn = getattr(process, "poll", None)
        if not callable(poll_fn):
            return False
        return poll_fn() is None

    def _terminate_server(self, server: ActiveServer) -> None:
        process = server.process
        try:
            poll_fn = getattr(process, "poll", None)
            terminate_fn = getattr(process, "terminate", None)
            kill_fn = getattr(process, "kill", None)
            wait_fn = getattr(process, "wait", None)
            if callable(poll_fn) and poll_fn() is None:
                pid = getattr(process, "pid", None)
                used_process_group = False
                if isinstance(pid, int) and pid > 0:
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                        used_process_group = True
                    except Exception:
                        used_process_group = False
                if not used_process_group and callable(terminate_fn):
                    terminate_fn()
                if callable(wait_fn):
                    try:
                        wait_fn(timeout=10)
                    except Exception:
                        if used_process_group and isinstance(pid, int) and pid > 0:
                            try:
                                os.killpg(os.getpgid(pid), signal.SIGKILL)
                            except Exception:
                                if callable(kill_fn):
                                    kill_fn()
                        elif callable(kill_fn):
                            kill_fn()
                        if callable(wait_fn):
                            wait_fn(timeout=5)
        finally:
            stdout_close = getattr(server.stdout_handle, "close", None)
            stderr_close = getattr(server.stderr_handle, "close", None)
            if callable(stdout_close):
                stdout_close()
            if callable(stderr_close):
                stderr_close()


def render_launch_command(
    *,
    launch: LaunchConfig,
    model: str,
    gpu_ids: tuple[int, ...],
    base_port: int,
    metrics_port: int,
) -> tuple[str, ...]:
    context = {
        "model": model,
        "gpu_id": str(gpu_ids[0]),
        "gpu_ids": ",".join(str(gpu_id) for gpu_id in gpu_ids),
        "base_port": str(base_port),
        "port": str(base_port),
        "metrics_port": str(metrics_port),
        "host": launch.host,
    }
    if launch.template is not None:
        rendered = tuple(_format_token(token, context) for token in launch.template)
        if not rendered:
            raise ValueError("launch template rendered to an empty command")
        return rendered

    command: list[str] = [
        launch.executable,
        "serve",
        model,
        "--host",
        launch.host,
        "--port",
        str(base_port),
        "--tensor-parallel-size",
        str(launch.tensor_parallel_size),
    ]
    if launch.dtype is not None:
        command.extend(["--dtype", launch.dtype])
    if launch.quantization is not None:
        command.extend(["--quantization", launch.quantization])
    if launch.tokenizer_mode is not None:
        command.extend(["--tokenizer-mode", launch.tokenizer_mode])
    if launch.gpu_memory_utilization is not None:
        command.extend(["--gpu-memory-utilization", _number_to_text(launch.gpu_memory_utilization)])
    if launch.max_model_len is not None:
        command.extend(["--max-model-len", str(launch.max_model_len)])
    if launch.max_num_seqs is not None:
        command.extend(["--max-num-seqs", _number_to_text(launch.max_num_seqs)])
    if launch.max_num_batched_tokens is not None:
        command.extend(["--max-num-batched-tokens", _number_to_text(launch.max_num_batched_tokens)])
    for token in launch.extra_args:
        command.append(_format_token(token, context))
    return tuple(command)


def _format_token(token: str, context: dict[str, str]) -> str:
    try:
        return token.format_map(context)
    except KeyError as exc:
        raise ValueError(f"unknown template placeholder in launch token {token!r}: {exc}") from exc


def _number_to_text(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value}"


def _default_readiness_probe(base_url: str, readiness_path: str, timeout_s: float) -> bool:
    url = f"{base_url.rstrip('/')}{readiness_path}"
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = getattr(response, "status", 0)
            return 200 <= int(status) < 500
    except urllib.error.URLError:
        return False
    except Exception:
        return False


def _default_process_factory(command: tuple[str, ...], **kwargs) -> object:
    kwargs.setdefault("start_new_session", True)
    return subprocess.Popen(command, **kwargs)
