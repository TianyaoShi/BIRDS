from __future__ import annotations

import pytest

from local_orchestrator.lifecycle import render_launch_command
from local_orchestrator.models import LaunchConfig
from local_orchestrator.resources import GPULeaseManager, ResourceUnavailableError


def test_gpu_lease_manager_acquires_and_releases_multi_gpu_lease() -> None:
    manager = GPULeaseManager(allowed_gpu_ids=(2, 3, 4), max_active_gpus=3)

    lease = manager.acquire(2)
    assert lease.gpu_ids == (2, 3)
    assert manager.snapshot()["free_gpu_ids"] == [4]

    with pytest.raises(ResourceUnavailableError):
        manager.acquire(2)

    manager.release(lease)
    assert manager.snapshot()["free_gpu_ids"] == [2, 3, 4]


def test_render_launch_command_expands_multi_gpu_template_placeholders() -> None:
    command = render_launch_command(
        launch=LaunchConfig(
            template=("serve", "{model}", "--gpus", "{gpu_ids}", "--primary", "{gpu_id}"),
            gpu_count=2,
            tensor_parallel_size=2,
        ),
        model="model-8b",
        gpu_ids=(2, 3),
        base_port=8000,
        metrics_port=9000,
    )

    assert command == ("serve", "model-8b", "--gpus", "2,3", "--primary", "2")
