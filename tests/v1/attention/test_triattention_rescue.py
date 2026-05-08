# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for the TriAttention V3 + longctx rescue stack.

Pure-Python tests covering the three new modules wired in 2026-05-07:

  - Tier 1 query-aware eviction in `engine._capture_query_q` and the
    score blend in `accumulate_layer_score`.
  - Tier 2 evict-to-vector callback (`backend_helpers._evict_to_longctx_callback`)
    + `maybe_finalize_evict` self-gating.
  - Tier 3 prefill rehydrate (`prefill_rehydrate.maybe_rehydrate_messages`)
    + `stash_prompt_token_ids` round-trip.

No live model needed — engine is constructed directly and longctx-svc is
mocked via monkeypatched `_LONGCTX_HTTP`.
"""
from __future__ import annotations

import os
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

# Stub vllm.config + vllm.model_executor.models.utils when the heavy
# import chain isn't available locally (CI / cluster has the real
# packages; this only kicks in for dev boxes without the full vLLM dep
# tree). Must run BEFORE importing vllm.v1.attention.triattention.*
# which transitively pulls them in via hooks.py + backend_helpers.py.
try:
    import vllm.config  # noqa: F401
except ModuleNotFoundError:
    _stub_cfg = types.ModuleType("vllm.config")
    _stub_cfg.get_current_vllm_config = lambda: None
    sys.modules["vllm.config"] = _stub_cfg

try:
    from vllm.model_executor.models.utils import extract_layer_index  # noqa
except (ModuleNotFoundError, ImportError):
    import re as _re

    def _extract_layer_index(layer_name: str) -> int:
        m = _re.search(r"\.(\d+)\.", layer_name)
        return int(m.group(1)) if m else 0

    _stub_utils = types.ModuleType("vllm.model_executor.models.utils")
    _stub_utils.extract_layer_index = _extract_layer_index
    # Build the parent packages too if missing.
    for pkg in ("vllm.model_executor", "vllm.model_executor.models"):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)
    sys.modules["vllm.model_executor.models.utils"] = _stub_utils

from vllm.v1.attention.triattention import backend_helpers, hooks
from vllm.v1.attention.triattention import prefill_rehydrate
from vllm.v1.attention.triattention.engine import (
    TriAttentionV3Config,
    TriAttentionV3Engine,
)


N_LAYERS = 4
N_HEADS = 8
N_KV_HEADS = 4
HEAD_DIM = 64


def _make_engine(
    lambda_query_attn: float = 0.0,
    budget: int = 64,
    warmup_tokens: int = 64,
    boundary_skip: int = 0,
    expected_layers: int | None = None,
    divide_length: int = 8,
) -> TriAttentionV3Engine:
    cfg = TriAttentionV3Config(
        budget=budget,
        prefix_protect=8,
        window_size=8,
        n_segments=4,
        warmup_tokens=warmup_tokens,
        hybrid_mode=2,
        lambda_query_attn=lambda_query_attn,
        query_tokens=16,
        query_min_window=32,
        boundary_skip=boundary_skip,
        expected_layers=expected_layers,
        divide_length=divide_length,
    )
    return TriAttentionV3Engine(
        cfg=cfg,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        rope_theta=10000.0,
    )


def _calibrated_engine(**kwargs) -> TriAttentionV3Engine:
    eng = _make_engine(**kwargs)
    # Push enough Q to trip calibration.
    q = torch.randn(eng.cfg.warmup_tokens, eng.n_heads, eng.head_dim)
    eng.accumulate_q(q, layer_idx=0)
    assert eng.calibrated
    return eng


@pytest.fixture(autouse=True)
def _isolate_module_state(monkeypatch):
    """Keep every test's module-level singletons / env from bleeding."""
    # Engine singleton (set/get via hooks.set_engine).
    hooks.set_engine(None)
    # backend_helpers module-level state.
    monkeypatch.setattr(backend_helpers, "_PROMPT_TOKEN_IDS", {})
    monkeypatch.setattr(backend_helpers, "_TOKENIZER", None)
    monkeypatch.setattr(backend_helpers, "_LONGCTX_BASE_URL", None)
    monkeypatch.setattr(backend_helpers, "_LONGCTX_HTTP", None)
    monkeypatch.setattr(
        backend_helpers, "_DEFAULT_LONGCTX_SESSION_ID", "v3-single-session"
    )
    monkeypatch.setattr(
        backend_helpers, "_LONGCTX_SESSION_ID", "v3-single-session"
    )
    # Env: clear LONGCTX_ENDPOINT for default no-op behaviour. Tests that
    # need it set use monkeypatch.setenv inside the test body.
    monkeypatch.delenv("LONGCTX_ENDPOINT", raising=False)
    yield
    hooks.set_engine(None)


# ---------------------------------------------------------------------------
# Tier 1: query Q capture + score blend
# ---------------------------------------------------------------------------


class TestQueryCapture:

    def test_capture_query_q_populates_buffers_per_layer(self):
        eng = _calibrated_engine(lambda_query_attn=1.0)
        # Pre-condition: query buffers should be either None OR not yet
        # populated for layer 1 (warmup pass only touched layer 0).
        # accumulate_q runs query capture as a side effect — verify the
        # layer index gets recorded.
        assert eng.q_query_real is not None  # layer 0 populated by warmup
        assert 0 in eng.q_query_layers_seen
        assert 1 not in eng.q_query_layers_seen

        # Now feed layer 1.
        q = torch.randn(eng.cfg.query_min_window, eng.n_heads, eng.head_dim)
        eng.accumulate_q(q, layer_idx=1)
        assert 1 in eng.q_query_layers_seen
        # Buffer is [n_layers * n_kv_heads, freq_count] — non-zero where
        # layers 0 & 1 wrote, zero elsewhere.
        cb_l1 = 1 * eng.n_kv_heads
        cb_l2 = 2 * eng.n_kv_heads
        assert eng.q_query_real[cb_l1 : cb_l1 + eng.n_kv_heads].abs().sum() > 0
        assert eng.q_query_real[cb_l2 : cb_l2 + eng.n_kv_heads].abs().sum() == 0

    def test_capture_skipped_for_short_prefill(self):
        """Decode-step (1 token) and trivial prefills must not corrupt
        per-turn query buffers — query_min_window gates this."""
        eng = _calibrated_engine(lambda_query_attn=1.0)
        # Layer 1 not yet seen (only the warmup hit layer 0).
        assert 1 not in eng.q_query_layers_seen
        # Push a too-short batch — should NOT get captured.
        q_short = torch.randn(
            eng.cfg.query_min_window - 1, eng.n_heads, eng.head_dim
        )
        eng.accumulate_q(q_short, layer_idx=1)
        assert 1 not in eng.q_query_layers_seen

    def test_capture_disabled_when_lambda_zero(self):
        eng = _calibrated_engine(lambda_query_attn=0.0)
        # warmup happened at lambda=0, so query buffers should never alloc.
        assert eng.q_query_real is None
        assert len(eng.q_query_layers_seen) == 0
        # Even an explicit further pass shouldn't populate them.
        q = torch.randn(eng.cfg.query_min_window, eng.n_heads, eng.head_dim)
        eng.accumulate_q(q, layer_idx=2)
        assert eng.q_query_real is None

    def test_capture_overwrites_per_turn(self):
        """Each new prefill of the same layer overwrites the previous query
        buffer — the user's question changed, the old one is stale."""
        eng = _calibrated_engine(lambda_query_attn=1.0)
        # Force a known input on layer 1.
        q_a = torch.zeros(eng.cfg.query_min_window, eng.n_heads, eng.head_dim)
        q_a[:] = 1.0
        eng.accumulate_q(q_a, layer_idx=1)
        cb = 1 * eng.n_kv_heads
        snapshot_a = eng.q_query_real[cb : cb + eng.n_kv_heads].clone()

        q_b = torch.zeros(eng.cfg.query_min_window, eng.n_heads, eng.head_dim)
        q_b[:] = -3.0
        eng.accumulate_q(q_b, layer_idx=1)
        snapshot_b = eng.q_query_real[cb : cb + eng.n_kv_heads].clone()

        # Layer 1's buffer changed (overwrite, not accumulate).
        assert not torch.allclose(snapshot_a, snapshot_b)


# ---------------------------------------------------------------------------
# Tier 2: maybe_finalize_evict self-gating
# ---------------------------------------------------------------------------


def _begin_round_and_fill(eng: TriAttentionV3Engine, seq_len: int) -> None:
    """Helper: start a score round and accumulate every layer so the
    expected_layers gate passes."""
    device = torch.device("cpu")
    eng.begin_score_round(0, seq_len, device)
    K = torch.randn(seq_len, eng.n_kv_heads, eng.head_dim)
    valid = eng.get_valid_mask(0, seq_len, device)
    max_pos = seq_len - 1
    window_thr = max_pos - eng.cfg.window_size + 1
    for il in range(eng.n_layers):
        eng.accumulate_layer_score(0, il, K, max_pos, window_thr)
        eng._seq_state[0]["pending_layers"].add(il)
    _ = valid


class TestMaybeFinalizeEvict:

    def test_no_engine_returns_zero(self):
        assert hooks.get_engine() is None
        assert backend_helpers.maybe_finalize_evict() == 0

    def test_uncalibrated_returns_zero(self):
        eng = _make_engine()
        hooks.set_engine(eng)
        assert not eng.calibrated
        assert backend_helpers.maybe_finalize_evict() == 0

    def test_no_pending_scores_returns_zero(self):
        eng = _calibrated_engine()
        hooks.set_engine(eng)
        # No begin_score_round call — pending_scores absent.
        assert backend_helpers.maybe_finalize_evict() == 0

    def test_below_budget_returns_zero(self):
        # budget=128 + divide_length=8; seq_len=64 (used=64) is well below
        # the eff_budget + divide gate => should_evict False.
        eng = _calibrated_engine(budget=128)
        hooks.set_engine(eng)
        _begin_round_and_fill(eng, seq_len=64)
        assert backend_helpers.maybe_finalize_evict() == 0
        # Pending state should remain (not finalized).
        assert "pending_scores" in eng._seq_state[0]

    def test_partial_layer_coverage_returns_zero(self):
        """If only some attention layers contributed scores, don't fire —
        the per-segment quota would be wrong on a partial sum."""
        eng = _calibrated_engine(budget=8)
        hooks.set_engine(eng)
        seq_len = 256
        device = torch.device("cpu")
        eng.begin_score_round(0, seq_len, device)
        K = torch.randn(seq_len, eng.n_kv_heads, eng.head_dim)
        max_pos = seq_len - 1
        window_thr = max_pos - eng.cfg.window_size + 1
        # Only layer 0 contributes — need all N_LAYERS for the gate to pass.
        eng.accumulate_layer_score(0, 0, K, max_pos, window_thr)
        eng._seq_state[0]["pending_layers"].add(0)

        # Budget is exceeded (used=128 > budget=8) but expected_layers=4
        # and only 1 seen, so the gate should hold.
        assert backend_helpers.maybe_finalize_evict(
            effective_seq_len=seq_len
        ) == 0
        assert "pending_scores" in eng._seq_state[0]  # not finalized

    def test_expected_layers_override_allows_backend_skipped_layers(self):
        """TurboQuant can skip boundary layers outside the TQ backend.

        The finalizer should wait for the number of layers that actually
        fire V3 hooks, not always the model's total layer count.
        """
        eng = _calibrated_engine(budget=8, expected_layers=2)
        hooks.set_engine(eng)
        seq_len = 256
        device = torch.device("cpu")
        eng.begin_score_round(0, seq_len, device)
        K = torch.randn(seq_len, eng.n_kv_heads, eng.head_dim)
        max_pos = seq_len - 1
        window_thr = max_pos - eng.cfg.window_size + 1
        # Simulate TQ boundary protection: only middle layers fire hooks.
        for il in (1, 2):
            eng.accumulate_layer_score(0, il, K, max_pos, window_thr)
            eng._seq_state[0]["pending_layers"].add(il)

        n = backend_helpers.maybe_finalize_evict(effective_seq_len=seq_len)
        assert n > 0
        assert "pending_scores" not in eng._seq_state[0]

    def test_fires_when_all_conditions_met(self):
        eng = _calibrated_engine(budget=8)
        hooks.set_engine(eng)
        seq_len = 256
        _begin_round_and_fill(eng, seq_len)
        n = backend_helpers.maybe_finalize_evict(effective_seq_len=seq_len)
        assert n > 0
        # Pending state cleared after finalize.
        assert "pending_scores" not in eng._seq_state[0]

    def test_effective_seq_len_path_runs(self):
        """The override branch is exercised: maybe_finalize_evict accepts
        and honours an explicit seq_len for the budget check."""
        eng = _calibrated_engine(budget=8)
        hooks.set_engine(eng)
        _begin_round_and_fill(eng, seq_len=256)
        # Override matches the actual cache size; should evict.
        n = backend_helpers.maybe_finalize_evict(effective_seq_len=256)
        assert n > 0


# ---------------------------------------------------------------------------
# Tier 2: evict-to-longctx callback span grouping + payload
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stub: maps each token id to its string form."""

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{int(i)}" for i in ids)


class _CapturingHttp:
    """Stand-in for `requests.Session` that records POSTs."""

    def __init__(self):
        self.calls: list[dict] = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"ok": True}
        return resp


class TestEvictToLongctxCallback:

    def test_noop_when_endpoint_unset(self):
        # No LONGCTX_ENDPOINT, no tokenizer — callback should silently bail.
        backend_helpers._evict_to_longctx_callback(
            seq_id=0,
            evict_pos=torch.tensor([1, 2, 3], dtype=torch.int32),
            n_evicted=3,
        )
        # No exception, no HTTP session built.
        assert backend_helpers._LONGCTX_HTTP is None

    def test_noop_when_tokenizer_unbound(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        # Reset the cached base url so the env var takes effect.
        monkeypatch.setattr(backend_helpers, "_LONGCTX_BASE_URL", None)
        http = _CapturingHttp()
        monkeypatch.setattr(backend_helpers, "_LONGCTX_HTTP", http)
        # Tokenizer not bound — callback warns and returns.
        backend_helpers._evict_to_longctx_callback(
            seq_id=0,
            evict_pos=torch.tensor([1, 2, 3]),
            n_evicted=3,
        )
        assert http.calls == []

    def test_noop_when_no_token_ids(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        monkeypatch.setattr(backend_helpers, "_LONGCTX_BASE_URL", None)
        backend_helpers.set_tokenizer(_FakeTokenizer())
        http = _CapturingHttp()
        monkeypatch.setattr(backend_helpers, "_LONGCTX_HTTP", http)
        # No prompt token IDs stashed for seq_id=0 -> skip.
        backend_helpers._evict_to_longctx_callback(
            seq_id=0,
            evict_pos=torch.tensor([1, 2, 3]),
            n_evicted=3,
        )
        assert http.calls == []

    def test_groups_contiguous_positions_into_spans(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        monkeypatch.setattr(backend_helpers, "_LONGCTX_BASE_URL", None)
        backend_helpers.set_tokenizer(_FakeTokenizer())
        http = _CapturingHttp()
        monkeypatch.setattr(backend_helpers, "_LONGCTX_HTTP", http)
        # Stash 200 tokens; evict positions [10,11,12,  100,101,  150].
        backend_helpers.set_prompt_token_ids(0, list(range(200)))
        evict = torch.tensor([10, 11, 12, 100, 101, 150], dtype=torch.int32)
        backend_helpers._evict_to_longctx_callback(
            seq_id=0, evict_pos=evict, n_evicted=int(evict.numel()),
        )
        assert len(http.calls) == 1
        body = http.calls[0]["json"]
        assert body["session_id"] == "v3-single-session"
        chunks = body["chunks"]
        assert len(chunks) == 3  # three contiguous runs
        # Each chunk has the BLEED=32 expansion baked into token_range.
        for c in chunks:
            ws, we = c["token_range"]
            assert 0 <= ws <= we <= 200
            assert isinstance(c["text"], str) and c["text"]
            assert c["layer"] == -1

    def test_post_url_is_evict_write(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://lc.test/")
        monkeypatch.setattr(backend_helpers, "_LONGCTX_BASE_URL", None)
        backend_helpers.set_tokenizer(_FakeTokenizer())
        http = _CapturingHttp()
        monkeypatch.setattr(backend_helpers, "_LONGCTX_HTTP", http)
        backend_helpers.set_prompt_token_ids(0, list(range(64)))
        backend_helpers._evict_to_longctx_callback(
            seq_id=0,
            evict_pos=torch.tensor([5, 6]),
            n_evicted=2,
        )
        assert http.calls[0]["url"] == "http://lc.test/evict/write"

    def test_callback_drops_out_of_bounds_positions(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        monkeypatch.setattr(backend_helpers, "_LONGCTX_BASE_URL", None)
        backend_helpers.set_tokenizer(_FakeTokenizer())
        http = _CapturingHttp()
        monkeypatch.setattr(backend_helpers, "_LONGCTX_HTTP", http)
        backend_helpers.set_prompt_token_ids(0, list(range(50)))
        # evict 5,6 (in-range) and 999 (out-of-range, should be dropped).
        backend_helpers._evict_to_longctx_callback(
            seq_id=0,
            evict_pos=torch.tensor([5, 6, 999]),
            n_evicted=3,
        )
        assert len(http.calls) == 1
        chunks = http.calls[0]["json"]["chunks"]
        # Only the 5,6 contiguous run survives.
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Tier 2: install_eviction_to_longctx wiring
# ---------------------------------------------------------------------------


class TestInstallEvictionToLongctx:

    def test_no_engine_returns_false(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        monkeypatch.setattr(backend_helpers, "_LONGCTX_BASE_URL", None)
        assert backend_helpers.install_eviction_to_longctx() is False

    def test_no_endpoint_returns_false(self):
        eng = _calibrated_engine()
        hooks.set_engine(eng)
        # LONGCTX_ENDPOINT not set -> bail.
        assert backend_helpers.install_eviction_to_longctx() is False
        assert eng._eviction_callback is None

    def test_registers_callback_when_both_available(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        monkeypatch.setattr(backend_helpers, "_LONGCTX_BASE_URL", None)
        eng = _calibrated_engine()
        hooks.set_engine(eng)
        assert backend_helpers.install_eviction_to_longctx() is True
        assert eng._eviction_callback is backend_helpers._evict_to_longctx_callback


# ---------------------------------------------------------------------------
# Tier 3: prefill rehydrate hook
# ---------------------------------------------------------------------------


class TestMaybeRehydrateMessages:

    def test_noop_when_endpoint_unset(self):
        msgs = [{"role": "user", "content": "hello"}]
        out, n = prefill_rehydrate.maybe_rehydrate_messages(msgs)
        assert n == 0
        assert out == msgs
        # Must return a fresh list (no aliasing the caller's list).
        assert out is not msgs

    def test_noop_when_no_user_message(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        msgs = [{"role": "system", "content": "you are X"}]
        out, n = prefill_rehydrate.maybe_rehydrate_messages(msgs)
        assert n == 0
        assert out == msgs

    def test_noop_when_retrieve_returns_empty(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        monkeypatch.setattr(
            backend_helpers, "retrieve_evicted_for_query",
            lambda *a, **kw: [],
        )
        msgs = [{"role": "user", "content": "what was the password?"}]
        out, n = prefill_rehydrate.maybe_rehydrate_messages(msgs)
        assert n == 0
        assert out == msgs

    def test_swallows_retrieve_exceptions(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")

        def _boom(*a, **kw):
            raise RuntimeError("longctx-svc unreachable")

        monkeypatch.setattr(
            backend_helpers, "retrieve_evicted_for_query", _boom,
        )
        msgs = [{"role": "user", "content": "anything"}]
        out, n = prefill_rehydrate.maybe_rehydrate_messages(msgs)
        # Returns the original messages, doesn't propagate.
        assert n == 0
        assert out == msgs

    def test_prepends_system_message_with_chunks(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        monkeypatch.setattr(
            backend_helpers, "retrieve_evicted_for_query",
            lambda *a, **kw: [
                {"text": "the password is hunter2",
                 "token_range": (10, 18),
                 "layer": -1, "score": 0.91},
                {"text": "the meeting is at 3pm",
                 "token_range": (50, 56),
                 "layer": -1, "score": 0.82},
            ],
        )
        msgs = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "what was the password?"},
        ]
        out, n = prefill_rehydrate.maybe_rehydrate_messages(msgs)
        assert n == 2
        assert len(out) == 3  # rehydrate sys msg prepended
        assert out[0]["role"] == "system"
        body = out[0]["content"]
        assert "hunter2" in body
        assert "3pm" in body
        assert "Recovered context" in body
        # Original messages preserved in order.
        assert out[1]["role"] == "system" and out[1]["content"] == "be helpful"
        assert out[2]["role"] == "user"

    def test_walks_to_last_user_message(self, monkeypatch):
        """Multi-turn: take the LAST user message, not the first."""
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        captured: dict[str, Any] = {}

        def _capture(query, top_k, score_floor):
            captured["query"] = query
            return []

        monkeypatch.setattr(
            backend_helpers, "retrieve_evicted_for_query", _capture,
        )
        msgs = [
            {"role": "user", "content": "FIRST question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "SECOND question"},
        ]
        prefill_rehydrate.maybe_rehydrate_messages(msgs)
        assert captured["query"] == "SECOND question"

    def test_handles_multimodal_text_parts(self, monkeypatch):
        """Multi-modal content: list of {type, text} dicts."""
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            backend_helpers, "retrieve_evicted_for_query",
            lambda q, **kw: captured.setdefault("query", q) and [] or [],
        )
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "part A"},
                {"type": "image_url", "image_url": "..."},
                {"type": "text", "text": "part B"},
            ],
        }]
        prefill_rehydrate.maybe_rehydrate_messages(msgs)
        assert "part A" in captured["query"]
        assert "part B" in captured["query"]


# ---------------------------------------------------------------------------
# Tier 2/3: longctx session id plumbing
# ---------------------------------------------------------------------------


class TestLongctxSessionIds:

    def test_set_resets_to_default_for_missing_id(self):
        backend_helpers.set_longctx_session_id("prd10m-armC-123")
        assert backend_helpers._LONGCTX_SESSION_ID == "prd10m-armC-123"

        effective = backend_helpers.set_longctx_session_id(None)

        assert effective == "v3-single-session"
        assert backend_helpers._LONGCTX_SESSION_ID == "v3-single-session"

    def test_request_id_round_trip_for_worker_process(self):
        request_id = "chatcmpl-abc123"
        session_id = "prd10m-armC-123/with spaces"

        encoded = backend_helpers.encode_request_id_with_longctx_session(
            request_id, session_id,
        )

        assert encoded.startswith(request_id)
        assert encoded != request_id
        assert backend_helpers.extract_longctx_session_id_from_request_id(
            encoded
        ) == session_id

    def test_default_session_does_not_pollute_request_id(self):
        request_id = "chatcmpl-abc123"

        encoded = backend_helpers.encode_request_id_with_longctx_session(
            request_id, "v3-single-session",
        )

        assert encoded == request_id
        assert backend_helpers.extract_longctx_session_id_from_request_id(
            encoded
        ) == "v3-single-session"


# ---------------------------------------------------------------------------
# Tier 3: stash_prompt_token_ids round-trip
# ---------------------------------------------------------------------------


class TestStashPromptTokenIds:

    def test_noop_when_endpoint_unset(self):
        prefill_rehydrate.stash_prompt_token_ids([1, 2, 3])
        # _PROMPT_TOKEN_IDS should remain empty since the endpoint gate
        # short-circuits before set_prompt_token_ids is invoked.
        assert backend_helpers._PROMPT_TOKEN_IDS == {}

    def test_noop_for_empty_list(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        prefill_rehydrate.stash_prompt_token_ids(None)
        prefill_rehydrate.stash_prompt_token_ids([])
        assert backend_helpers._PROMPT_TOKEN_IDS == {}

    def test_round_trips_token_ids(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        prefill_rehydrate.stash_prompt_token_ids([7, 8, 9, 10])
        # Phase A is single-batch (seq_id=0).
        assert backend_helpers._PROMPT_TOKEN_IDS[0] == [7, 8, 9, 10]

    def test_overwrites_on_new_prefill(self, monkeypatch):
        monkeypatch.setenv("LONGCTX_ENDPOINT", "http://localhost:9999")
        prefill_rehydrate.stash_prompt_token_ids([1, 2, 3])
        prefill_rehydrate.stash_prompt_token_ids([4, 5, 6, 7])
        assert backend_helpers._PROMPT_TOKEN_IDS[0] == [4, 5, 6, 7]
