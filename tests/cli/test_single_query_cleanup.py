"""Regression tests for quiet single-query cleanup.

Quiet `hermes chat -Q -q ...` exits via sys.exit() immediately after printing
machine-readable output. That path must explicitly run cleanup before raising
SystemExit; otherwise memory-provider background writers are left to atexit and
Hindsight retain can race interpreter shutdown.
"""

from __future__ import annotations

import pytest


def test_quiet_single_query_runs_cleanup_before_sys_exit(monkeypatch, capsys):
    import cli as cli_mod

    events: list[str] = []

    class FakeAgent:
        session_id = "session-after-run"
        quiet_mode = False
        suppress_status_output = False
        stream_delta_callback = object()
        tool_gen_callback = object()

        def run_conversation(self, *, user_message, conversation_history):
            events.append("run_conversation")
            assert user_message == "ping"
            assert conversation_history == []
            return {"final_response": "pong"}

    class FakeHermesCLI:
        def __init__(self, **kwargs):
            self.session_id = "session-before-run"
            self.conversation_history = []
            self.tool_progress_mode = "on"
            self.agent = FakeAgent()
            self._active_agent_route_signature = "default"

        def _ensure_runtime_credentials(self):
            return True

        def _preprocess_images_with_vision(self, query, images, announce=False):
            raise AssertionError("no images should be preprocessed")

        def _resolve_turn_agent_config(self, effective_query):
            assert effective_query == "ping"
            return {"signature": "default", "model": None, "runtime": None, "request_overrides": None}

        def _init_agent(self, **kwargs):
            events.append("init_agent")
            return True

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeHermesCLI)
    monkeypatch.setattr(cli_mod, "_collect_query_images", lambda query, image: (query, []))
    monkeypatch.setattr(cli_mod, "_parse_skills_argument", lambda skills: [])
    monkeypatch.setattr(cli_mod, "build_preloaded_skills_prompt", lambda *a, **kw: ("", [], []))
    monkeypatch.setattr(cli_mod, "_run_cleanup", lambda: events.append("cleanup"))
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli_mod,
        "_get_platform_tools",
        lambda *a, **kw: ["terminal"],
        raising=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="ping", quiet=True)

    assert exc_info.value.code == 0
    assert events == ["init_agent", "run_conversation", "cleanup"]
    captured = capsys.readouterr()
    assert "pong" in captured.out
    assert "session_id: session-after-run" in captured.err
