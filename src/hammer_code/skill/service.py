"""Per-Agent Skill invocation with complete inline-body guarantees."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from uuid import uuid4

from hammer_code.config import ContextConfig, SkillConfig
from hammer_code.domain.events import ModelRequest, ResponseCompleted, StopReason, UsageUpdated
from hammer_code.domain.messages import Message, Role, TextBlock, ToolCallBlock
from hammer_code.domain.usage import TokenUsage
from hammer_code.llm.client import ModelClient
from hammer_code.skill.models import (
    CatalogSnapshot,
    LoadedSkill,
    SkillDefinition,
    SkillLoadOutcome,
    SkillObservation,
    SkillTurnEvidence,
)
from hammer_code.skill.parser import SkillInvocationError, render_skill
from hammer_code.skill.repository import SkillRepository
from hammer_code.tools.base import ToolExecutionResult
from hammer_code.tools.executor import ToolExecutor
from hammer_code.tools.registry import ToolRegistry
from hammer_code.tools.results import fits_inline_result


@dataclass(frozen=True)
class PreparedUserInvocation:
    prompt: str
    is_fork: bool
    definition: SkillDefinition


@dataclass
class SkillTurnState:
    snapshot: CatalogSnapshot
    retrieved: tuple = ()
    retrieved_prompt: str = ""
    retrieve_event_ids: tuple[str, ...] = ()
    loaded_by_effective_name: dict[str, LoadedSkill] = field(default_factory=dict)
    loaded_by_alias: dict[str, LoadedSkill] = field(default_factory=dict)
    tool_outcomes: dict[str, SkillLoadOutcome] = field(default_factory=dict)
    pending_outcomes: list[SkillLoadOutcome] = field(default_factory=list)


class SkillInvocationService:
    """Holds a single pinned catalog snapshot for one active Agent turn."""

    def __init__(
        self,
        repository: SkillRepository,
        skill_config: SkillConfig,
        context_config: ContextConfig,
    ) -> None:
        self.repository = repository
        self.skill_config = skill_config
        self.context_config = context_config
        self._snapshot: CatalogSnapshot | None = None
        self._state: SkillTurnState | None = None
        self._observation: SkillObservation | None = None
        self._fork_runtime: _ForkRuntime | None = None

    def bind_fork_runtime(
        self,
        *,
        client: ModelClient,
        executor: ToolExecutor,
        registry: ToolRegistry,
        system_prompt: str,
        project_instructions: str,
        mcp_prompt: str,
        max_output_tokens: int,
        record_usage: Callable[[str, str, TokenUsage], None],
    ) -> None:
        self._fork_runtime = _ForkRuntime(
            client,
            executor,
            registry,
            system_prompt,
            project_instructions,
            mcp_prompt,
            max_output_tokens,
            record_usage,
        )

    def begin_turn(
        self,
        snapshot: CatalogSnapshot,
        observation: SkillObservation | None,
        *,
        retrieved_prompt: str = "",
        retrieve_event_ids: tuple[str, ...] = (),
    ) -> None:
        self._snapshot = snapshot
        self._observation = observation
        self._state = SkillTurnState(
            snapshot,
            observation.retrieved if observation else (),
            retrieved_prompt,
            retrieve_event_ids,
        )

    def end_turn(self, *, completed: bool) -> SkillObservation | None:
        observation = self._observation
        self._snapshot = None
        self._observation = None
        self._state = None
        if observation is None:
            return None
        return SkillObservation(
            observation.snapshot_generation, observation.retrieved, observation.used, completed
        )

    async def _resolve(self, name: str, *, user: bool, model: bool) -> SkillDefinition:
        if not self.skill_config.enabled:
            raise SkillInvocationError("Skills are disabled by configuration")
        if self._snapshot is None:
            raise SkillInvocationError("Skill invocation is unavailable outside an active turn")
        state = self._state
        if state is not None and name in state.loaded_by_alias:
            return state.loaded_by_alias[name].ref and await self.repository.resolve_latest(
                name, user=user, model=model
            )
        return await self.repository.resolve_latest(name, user=user, model=model)

    async def invoke(self, name: str, arguments: str) -> ToolExecutionResult:
        try:
            state = self._state
            if state is not None and name in state.loaded_by_alias:
                loaded = state.loaded_by_alias[name]
                state.pending_outcomes.append(SkillLoadOutcome(loaded.ref))
                receipt = (
                    f"Skill already loaded for this turn: {loaded.ref.scope.value}:"
                    f"{loaded.ref.name}@{loaded.ref.version}."
                )
                return ToolExecutionResult(receipt)
            definition = await self._resolve(name, user=False, model=True)
            if state is not None and definition.name in state.loaded_by_effective_name:
                loaded = state.loaded_by_effective_name[definition.name]
                state.loaded_by_alias[name] = loaded
                state.pending_outcomes.append(SkillLoadOutcome(loaded.ref))
                receipt = (
                    f"Skill already loaded for this turn: {loaded.ref.scope.value}:"
                    f"{loaded.ref.name}@{loaded.ref.version}."
                )
                return ToolExecutionResult(receipt)
            rendered = render_skill(definition, arguments)
            if definition.context.value == "inline" and not fits_inline_result(
                rendered, self.context_config
            ):
                return ToolExecutionResult(
                    "Error: Skill body exceeds the complete inline tool-result limit; "
                    "use /skill instead.",
                    True,
                )
            if definition.context.value == "fork" and self._fork_runtime is None:
                return ToolExecutionResult("Error: fork Skill runtime is unavailable.", True)
            event_id = await self.repository.record_loaded(
                definition.ref,
                invocation_kind="fork" if definition.context.value == "fork" else "model",
                catalog_generation=self._snapshot.generation if self._snapshot else 0,
            )
            if state is not None:
                loaded = LoadedSkill(
                    definition.ref,
                    rendered,
                    "fork" if definition.context.value == "fork" else "model",
                    event_id,
                )
                state.loaded_by_effective_name[definition.name] = loaded
                state.loaded_by_alias[name] = loaded
                state.pending_outcomes.append(SkillLoadOutcome(definition.ref))
            if self._observation is not None:
                self._observation = SkillObservation(
                    self._observation.snapshot_generation,
                    self._observation.retrieved,
                    (*self._observation.used, definition.ref),
                )
            if definition.context.value == "fork":
                return await self._invoke_fork(definition, rendered, arguments)
            return ToolExecutionResult(rendered)
        except (SkillInvocationError, ValueError) as exc:
            return ToolExecutionResult(f"Error: {exc}", True)

    async def invoke_user(self, name: str, arguments: str) -> PreparedUserInvocation:
        state = self._state
        if state is not None and name in state.loaded_by_alias:
            loaded = state.loaded_by_alias[name]
            return PreparedUserInvocation(
                loaded.body,
                False,
                await self.repository.resolve_latest(name, user=True, model=False),
            )
        definition = await self._resolve(name, user=True, model=False)
        rendered = render_skill(definition, arguments)
        if len(rendered) > self.skill_config.evolution.max_body_chars:
            raise SkillInvocationError("Skill body exceeds the configured maximum")
        if definition.context.value == "fork" and self._fork_runtime is None:
            raise SkillInvocationError("fork Skill runtime is unavailable")
        event_id = await self.repository.record_loaded(
            definition.ref,
            invocation_kind="direct",
            catalog_generation=self._snapshot.generation if self._snapshot else 0,
        )
        if state is not None:
            loaded = LoadedSkill(definition.ref, rendered, "direct", event_id)
            state.loaded_by_effective_name[definition.name] = loaded
            state.loaded_by_alias[name] = loaded
        return PreparedUserInvocation(rendered, definition.context.value == "fork", definition)

    def observe_tool_result(self, call: ToolCallBlock, arguments: object, raw: object) -> None:
        """Associate successful use_skill results by the executor's real call id."""
        if call.name != "use_skill" or self._state is None:
            return
        if self._state.pending_outcomes:
            self._state.tool_outcomes[call.call_id] = self._state.pending_outcomes.pop(0)

    def finish_turn_evidence(
        self, *, completed: bool, live_messages: tuple[Message, ...]
    ) -> SkillTurnEvidence | None:
        state = self._state
        if state is None:
            return None
        evidence = SkillTurnEvidence(
            tuple(state.retrieved),
            state.retrieved_prompt,
            tuple(state.loaded_by_effective_name.values()),
            live_messages,
            completed,
            state.retrieve_event_ids,
        )
        self.end_turn(completed=completed)
        return evidence

    @property
    def tool_outcomes(self) -> dict[str, SkillLoadOutcome]:
        return {} if self._state is None else dict(self._state.tool_outcomes)

    async def invoke_prepared_fork(
        self, prepared: PreparedUserInvocation, arguments: str
    ) -> ToolExecutionResult:
        if not prepared.is_fork:
            raise SkillInvocationError("Prepared Skill is not a fork Skill")
        return await self._invoke_fork(prepared.definition, prepared.prompt, arguments)

    async def _invoke_fork(
        self, definition: SkillDefinition, rendered: str, arguments: str
    ) -> ToolExecutionResult:
        runtime = self._fork_runtime
        if runtime is None:
            return ToolExecutionResult("Error: fork Skill runtime is unavailable.", True)
        forbidden = {"use_skill", "tool_search"}
        allowed = set(runtime.registry.exposed_names()).difference(forbidden)
        if definition.allowed_tools is not None:
            allowed.intersection_update(definition.allowed_tools)
        registry = runtime.registry.restricted_view(allowed)
        executor = ToolExecutor(
            registry,
            runtime.executor.permissions,
            replace(runtime.executor.context, skill_invoker=None),
            runtime.executor.runtime,
            runtime.executor.context_config,
            runtime.executor.estimator,
        )
        system_prompt = "\n\n".join(
            section.strip()
            for section in (
                runtime.system_prompt,
                runtime.project_instructions,
                runtime.mcp_prompt,
                rendered,
            )
            if section.strip()
        )
        messages: list[Message] = [
            Message(Role.USER, (TextBlock(arguments or "[No additional arguments.]"),))
        ]
        calls = 0
        unknown_calls = 0
        for _ in range(20):
            request_id = f"skill-fork-{uuid4()}"
            request = ModelRequest(
                request_id,
                request_id,
                system_prompt,
                tuple(messages),
                registry.definitions(),
                runtime.max_output_tokens,
            )
            completed: ResponseCompleted | None = None
            async for event in runtime.client.stream(request):
                if event.request_id != request_id:
                    return ToolExecutionResult(
                        "Error: fork received a mismatched model event.", True
                    )
                if isinstance(event, UsageUpdated):
                    runtime.record_usage(request_id, request_id, event.usage)
                elif isinstance(event, ResponseCompleted):
                    if completed is not None:
                        return ToolExecutionResult(
                            "Error: fork received duplicate completion.", True
                        )
                    completed = event
            if completed is None:
                return ToolExecutionResult("Error: fork model request did not complete.", True)
            response = completed.response
            tool_calls = tuple(
                block for block in response.message.content if isinstance(block, ToolCallBlock)
            )
            if response.stop_reason is not StopReason.TOOL_CALL:
                if response.stop_reason is not StopReason.END_TURN:
                    return ToolExecutionResult("Error: fork did not complete normally.", True)
                text = "".join(
                    block.text for block in response.message.content if isinstance(block, TextBlock)
                )
                return ToolExecutionResult(
                    text or "Error: fork did not return visible text.", not bool(text)
                )
            if not tool_calls or calls + len(tool_calls) > 50:
                return ToolExecutionResult("Error: fork tool-call limit reached.", True)
            calls += len(tool_calls)
            messages.append(response.message)
            results = await executor.execute_batch(tool_calls)
            unknown_calls += sum(
                1
                for result in results
                if result.is_error and "unknown, disabled, or unexposed" in result.content[0].text
            )
            if unknown_calls > 2:
                return ToolExecutionResult("Error: fork made too many unknown tool calls.", True)
            messages.append(Message(Role.USER, tuple(results)))
        return ToolExecutionResult("Error: fork request limit reached.", True)


@dataclass(frozen=True)
class _ForkRuntime:
    client: ModelClient
    executor: ToolExecutor
    registry: ToolRegistry
    system_prompt: str
    project_instructions: str
    mcp_prompt: str
    max_output_tokens: int
    record_usage: Callable[[str, str, TokenUsage], None]
