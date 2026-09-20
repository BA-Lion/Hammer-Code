from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from hammer_code.app.runtime import PrimaryAgentFactory
from hammer_code.config import AppConfig, resolve_profile
from hammer_code.domain.events import ModelEvent, ModelRequest
from hammer_code.llm.client import ClientCapabilities, ModelClient
from hammer_code.mcp.manager import McpManager
from hammer_code.memory.store import MemoryStore
from hammer_code.permissions.checker import PermissionChecker
from hammer_code.permissions.models import ApprovalChoice, PermissionMode, PermissionRequest
from hammer_code.permissions.rules import RuleStore
from hammer_code.permissions.service import PermissionService
from hammer_code.session.manager import SessionManager
from hammer_code.tools.registry import ToolRegistry


class _Client(ModelClient):
    @property
    def capabilities(self) -> ClientCapabilities:
        return ClientCapabilities("fake", True, False, False, True)

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        del request
        raise AssertionError("No model request is expected")
        yield  # pragma: no cover

    async def aclose(self) -> None:
        return None


class _UI:
    async def approve(self, request: PermissionRequest, reason: str) -> ApprovalChoice:
        del request, reason
        return ApprovalChoice.DENY

    def info(self, message: str) -> None:
        del message

    def error(self, message: str) -> None:
        del message

    def persistence_warning(self, message: str, *, final: bool = False) -> None:
        del message, final

    def memory_warning(self, message: str) -> None:
        del message

    def skill_warning(self, message: str) -> None:
        del message


def _factory(tmp_path: Path) -> PrimaryAgentFactory:
    config = AppConfig.model_validate(
        {
            "default_profile": "x",
            "profiles": {
                "x": {
                    "protocol": "openai_chat_completions",
                    "model": "m",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "K",
                    "max_output_tokens": 5,
                    "timeout_seconds": 1,
                    "max_retries": 0,
                }
            },
        }
    )
    resolved = resolve_profile(config, None, {"K": "secret"})
    registry = ToolRegistry()
    ui = _UI()
    permissions = PermissionService(
        PermissionChecker(PermissionMode.DEFAULT, RuleStore(tmp_path)), ui, RuleStore(tmp_path)
    )
    return PrimaryAgentFactory(
        workspace_root=tmp_path,
        cwd=tmp_path,
        config=config,
        resolved=resolved,
        client=_Client(),
        ui=ui,  # type: ignore[arg-type]
        registry=registry,
        mcp_manager=McpManager(config.mcp, registry, tmp_path),
        permissions=permissions,
        sessions=SessionManager(tmp_path),
        memory_store=MemoryStore(tmp_path),
        maintenance_lock=asyncio.Lock(),
    )


@pytest.mark.asyncio
async def test_factory_shares_process_resources_and_isolates_agent_resources(
    tmp_path: Path,
) -> None:
    factory = _factory(tmp_path)
    first = await factory.create_new()
    second = await factory.create_new()
    try:
        assert first.client is second.client
        assert first.registry is second.registry
        assert first.mcp_manager is second.mcp_manager
        assert first.permissions is second.permissions
        assert first.memory_service is not None and second.memory_service is not None
        assert first.memory_service.store is second.memory_service.store
        assert first.memory_service._maintenance_lock is second.memory_service._maintenance_lock

        assert first.session_id != second.session_id
        assert first.manager is not second.manager
        assert first.context_window is not second.context_window
        assert first.context_manager is not second.context_manager
        assert first.executor is not second.executor
        assert first.memory_service is not second.memory_service
        assert first.runtime is not None and second.runtime is not None
        assert first.runtime is not second.runtime

        second_result = second.runtime.write_result("second", "keep")
        await first.clear()
        assert second_result.read_text(encoding="utf-8") == "keep"
    finally:
        await first.cancel()
        await second.cancel()


@pytest.mark.asyncio
async def test_factory_resume_keeps_session_identity_and_restores_stale_flag(
    tmp_path: Path,
) -> None:
    factory = _factory(tmp_path)
    original = await factory.create_new()
    session_id = original.session_id
    await original.cancel()

    resumed = await factory.resume(session_id)
    try:
        assert resumed.session_id == session_id
        assert resumed.manager.conversation is not None
        assert resumed.manager.conversation.protocol == "openai_chat_completions"
    finally:
        await resumed.cancel()
