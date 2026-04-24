# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path


def test_turboquant_decode_scratch_is_not_registered_per_layer():
    attention_source = Path(
        "vllm/model_executor/layers/attention/attention.py"
    ).read_text()

    tq_init_source = attention_source.split("def _init_turboquant_buffers", 1)[1]
    tq_init_source = tq_init_source.split("def forward", 1)[0]

    assert "_tq_centroids" in tq_init_source
    assert 'register_buffer("_tq_mid_o_buf"' not in tq_init_source
    assert 'register_buffer("_tq_output_buf"' not in tq_init_source
    assert 'register_buffer("_tq_lse_buf"' not in tq_init_source


def test_turboquant_decode_uses_workspace_manager():
    decode_source = Path(
        "vllm/v1/attention/ops/triton_turboquant_decode.py"
    ).read_text()

    assert "current_workspace_manager" in decode_source
    assert "get_simultaneous" in decode_source
    assert "is_workspace_manager_initialized" in decode_source
    assert "buf_holder._tq_mid_o_buf" not in decode_source
    assert "buf_holder._tq_output_buf" not in decode_source
    assert "buf_holder._tq_lse_buf" not in decode_source


def test_turboquant_workspace_is_reserved_before_capture_lock():
    runner_source = Path("vllm/v1/worker/gpu_model_runner.py").read_text()
    capture_source = runner_source.split("def capture_model", 1)[1]
    capture_source = capture_source.split("def get_cuda_graph_size", 1)[0]

    assert "def _reserve_turboquant_decode_workspace" in runner_source
    assert "self._reserve_turboquant_decode_workspace()" in capture_source
    assert capture_source.index("self._reserve_turboquant_decode_workspace()") < (
        capture_source.index("lock_workspace()")
    )
