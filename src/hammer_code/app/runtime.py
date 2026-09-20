"""Shared process resources and per-session PrimaryAgent construction."""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from hammer_code.app.agent import PrimaryAgent
from hammer_code.app.context_manager import ContextManager, RecoveryState
from hammer_code.app.context_window import ContextWindow
from hammer_code.app.token_estimator import TokenEstimator
from hammer_code.config import AppConfig, ResolvedProfile
from hammer_code.conversation.manager import ConversationManager
from hammer_code.llm.client import ModelClient
from hammer_code.mcp.manager import McpManager
from hammer_code.memory.service import MemoryService
from hammer_code.memory.store import MemoryStore
from hammer_code.permissions.service import PermissionService
from hammer_code.prompts import PromptRuntimeContext, build_system_prompt
from hammer_code.session.manager import SessionManager
from hammer_code.session.models import RestoreResult
from hammer_code.session.session import Session, SessionCoordinator
from hammer_code.skill.evolution import SkillEvolutionService
from hammer_code.skill.repository import SkillRepository
from hammer_code.skill.service import SkillInvocationService
from hammer_code.tools.base import ToolExecutionContext
from hammer_code.tools.builtin.shell import sanitized_environment
from hammer_code.tools.executor import ToolExecutor
from hammer_code.tools.registry import ToolRegistry
from hammer_code.tools.runtime import RuntimeStore
from hammer_code.ui.console import ConsolePort


class AgentFactory(Protocol):
    async def create_new(self) -> PrimaryAgent: ...

    async def resume(self, session_id: str) -> PrimaryAgent: ...


class PrimaryAgentFactory:
    """Build isolated session state around process-scoped dependencies."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        cwd: Path,
        config: AppConfig,
        resolved: ResolvedProfile,
        client: ModelClient,
        ui: ConsolePort,
        registry: ToolRegistry,
        mcp_manager: McpManager,
        permissions: PermissionService,
        sessions: SessionManager,
        memory_store: MemoryStore,
        maintenance_lock: asyncio.Lock,
        skill_repository: SkillRepository | None = None,
        project_instructions: str = "",
        show_reasoning: bool = False,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.cwd = cwd.resolve()
        self.config = config
        self.resolved = resolved
        self.client = client
        self.ui = ui
        self.registry = registry
        self.mcp_manager = mcp_manager
        self.permissions = permissions
        self.sessions = sessions
        self.memory_store = memory_store
        self.maintenance_lock = maintenance_lock
        self.skill_repository = skill_repository
        self.project_instructions = project_instructions
        self.show_reasoning = show_reasoning
        self._cleanup_old_done = False

    async def create_new(self) -> PrimaryAgent:
        session = await asyncio.to_thread(self.sessions.create, self.resolved.profile.protocol)
        try:
            manager = ConversationManager()
            manager.create(self.resolved)
            return self._build(manager, SessionCoordinator(session, manager), stale_restore=False)
        except Exception:
            # A newly-created empty session is the only construction failure that may be removed.
            try:
                session = Session.open(self.sessions.sessions_root, session.directory)
                if not session.load().records:
                    await asyncio.to_thread(self._delete_empty, session)
            except Exception:
                pass
            raise

    async def resume(self, session_id: str) -> PrimaryAgent:
        session, restored, stale = await asyncio.to_thread(self._load_session, session_id)
        manager = ConversationManager()
        manager.restore(self.resolved, restored.messages, session.meta.total_tokens)
        coordinator = SessionCoordinator.restored(session, manager, restored)
        return self._build(manager, coordinator, stale_restore=stale)

    def _build(
        self,
        manager: ConversationManager,
        coordinator: SessionCoordinator,
        *,
        stale_restore: bool,
    ) -> PrimaryAgent:
        runtime = RuntimeStore(self.workspace_root)
        if not self._cleanup_old_done:
            runtime.cleanup_old()
            self._cleanup_old_done = True
        recovery = RecoveryState(self.config.context)
        prompt = self._prompt(self.permissions.mode)
        context_window = ContextWindow(prompt)
        context_manager = ContextManager(
            manager,
            self.client,
            runtime,
            self.config.context,
            TokenEstimator(),
            recovery,
            prompt,
            self.resolved.profile.max_output_tokens,
        )
        skill_service = (
            SkillInvocationService(self.skill_repository, self.config.skill, self.config.context)
            if self.skill_repository is not None
            else None
        )
        execution_context = ToolExecutionContext(
            self.workspace_root,
            self.cwd,
            runtime.session_dir,
            sanitized_environment(),
            skill_service,
        )
        executor = ToolExecutor(
            self.registry,
            self.permissions,
            execution_context,
            runtime,
            self.config.context,
            TokenEstimator(),
            context_manager.observe_tool_result,
        )
        if skill_service is not None:
            skill_service.bind_fork_runtime(
                client=self.client,
                executor=executor,
                registry=self.registry,
                system_prompt=prompt,
                project_instructions=self.project_instructions,
                mcp_prompt=self.mcp_manager.prompt,
                max_output_tokens=self.resolved.profile.max_output_tokens,
                record_usage=manager.record_maintenance_usage,
            )
        memory = MemoryService(
            self.client,
            self.memory_store,
            coordinator,
            self.ui,
            self.resolved.profile.max_output_tokens,
            executor,
            maintenance_lock=self.maintenance_lock,
        )
        evolution = (
            SkillEvolutionService(
                self.client,
                self.skill_repository,
                self.permissions,
                manager,
                self.config.skill,
                self.resolved.profile.max_output_tokens,
                self.registry.exposed_names(),
                warning=self.ui.skill_warning,
            )
            if self.skill_repository is not None
            else None
        )
        return PrimaryAgent(
            manager,
            self.client,
            self.ui,
            prompt,
            self.resolved.profile.max_output_tokens,
            self.show_reasoning,
            self.registry,
            executor,
            context_window,
            self.mcp_manager,
            context_manager,
            coordinator,
            memory,
            self.project_instructions,
            stale_restore,
            self.permissions,
            self._prompt,
            runtime,
            skill_service,
            evolution,
        )

    def _prompt(self, mode: object) -> str:
        return build_system_prompt(
            PromptRuntimeContext(
                cwd=self.cwd,
                project_root=self.workspace_root,
                platform=sys.platform,
                shell_backend="PowerShell",
                permission_mode=getattr(mode, "value", str(mode)),
                enabled_tools=self.registry.exposed_names(),
            )
        )

    def _load_session(self, session_id: str) -> tuple[Session, RestoreResult, bool]:
        session = self.sessions.open(session_id, self.resolved.profile.protocol)
        old_last_active = session.meta.last_active
        restored = session.load()
        stale = (datetime.now(UTC) - old_last_active.astimezone(UTC)) > timedelta(hours=24)
        session.touch()
        return session, restored, stale

    @staticmethod
    def _delete_empty(session: Session) -> None:
        import shutil

        shutil.rmtree(session.directory)
