"""The traffic seam: what a plugin sees of an inference request, and how it is asked.

A plugin registers a *traffic observer* through ``PluginContext``. The gateway
builds one provider-neutral :class:`Conversation` per inference request (chat,
messages, or responses shape), asks each observer about it before dispatch, asks
again about every tool call the model produces (buffered per call while a
stream flows), asks once more about the finished response, and writes whatever
the observers annotated onto the usage row.

Every observer call is fenced: an exception is logged with the plugin's name
and skipped, an async call that outlives ``plugins.observer_timeout_ms`` is
cancelled and skipped (a sync one cannot be interrupted, so an overrun is
logged instead), and nothing at all runs when no plugin registered an observer.

Decisions apply. A request ``block`` is a 403 before the provider is called;
``inject_system`` is prepended to the system text of the provider call; a tool
call ``deny`` removes the call from the response and puts the plugin's message
in its place, in a stream by holding the call's fragments back until it is
complete; a response ``block`` replaces a non-streamed response with a 403 and
is only recorded for a streamed one, whose bytes are already gone.
"""

import asyncio
import hashlib
import inspect
import json
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from gateway.log_config import logger

Api = Literal["chat", "messages", "responses"]


@dataclass(frozen=True)
class Caller:
    """Who sent the request, as the gateway resolved it."""

    api_key_id: str | None
    user_id: str | None
    workspace_id: str | None
    organization_id: str | None


@dataclass(frozen=True)
class ToolCall:
    """A tool the model asked the client to run."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """What the client sent back for one tool call."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Turn:
    """One assistant message and the tool results the client returned for it."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    results: tuple[ToolResult, ...] = ()


@dataclass(frozen=True)
class Conversation:
    """The request's conversation, the same shape whichever API carried it."""

    api: Api
    model: str
    system: str
    turns: tuple[Turn, ...]
    session_key: str
    latest_user_text: str = ""

    @property
    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None


@dataclass(frozen=True)
class RequestEvent:
    caller: Caller
    conversation: Conversation


@dataclass
class RequestDecision:
    """What an observer says about a request.

    ``block`` refuses it with a 403 carrying the message; ``inject_system`` is
    prepended to the provider call's system text; ``annotations`` reach the
    usage row.
    """

    inject_system: str | None = None
    block: str | None = None
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallEvent:
    caller: Caller
    conversation: Conversation
    tool_call: ToolCall


@dataclass
class ToolCallDecision:
    """What an observer says about a tool call. ``deny`` replaces the call with the message."""

    deny: str | None = None
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResponseEvent:
    """The model's finished answer: its text and the tool calls that survived."""

    caller: Caller
    conversation: Conversation
    text: str
    tool_calls: tuple[ToolCall, ...]
    streamed: bool


@dataclass
class ResponseDecision:
    """What an observer says about a response. ``block`` applies to a non-streamed one only."""

    block: str | None = None
    annotations: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TrafficObserver(Protocol):
    """What a plugin registers. Every method is optional; each may be sync or async."""

    def on_request(self, event: RequestEvent) -> Any: ...

    def on_tool_call(self, event: ToolCallEvent) -> Any: ...

    def on_response(self, event: ResponseEvent) -> Any: ...


DEFAULT_OBSERVER_TIMEOUT_MS = 250


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TrafficObservers:
    """The observers this process loaded, called with the fences the seam promises."""

    def __init__(self, observers: Iterable[tuple[str, Any]], timeout_ms: int = DEFAULT_OBSERVER_TIMEOUT_MS) -> None:
        self._observers = [(name, observer) for name, observer in observers]
        self._timeout = timeout_ms / 1000

    def __bool__(self) -> bool:
        return bool(self._observers)

    def __len__(self) -> int:
        return len(self._observers)

    async def _ask(self, method_name: str, event: Any) -> list[tuple[str, Any]]:
        """Call ``method_name`` on every observer that has it; return (plugin, decision) pairs."""
        answers: list[tuple[str, Any]] = []
        for name, observer in self._observers:
            method = getattr(observer, method_name, None)
            if method is None:
                continue
            started = time.perf_counter()
            try:
                outcome = method(event)
                if inspect.isawaitable(outcome):
                    outcome = await asyncio.wait_for(outcome, timeout=self._timeout)
            except TimeoutError:
                logger.warning(
                    "Plugin %s: %s took longer than %.0f ms and was skipped", name, method_name, self._timeout * 1000
                )
                continue
            except Exception:  # noqa: BLE001 one plugin's failure must not touch the request
                logger.exception("Plugin %s: %s raised and was skipped", name, method_name)
                continue
            elapsed = time.perf_counter() - started
            if elapsed > self._timeout:
                # A synchronous observer cannot be cut off; say so rather than hide it.
                logger.warning("Plugin %s: %s took %.0f ms synchronously", name, method_name, elapsed * 1000)
            if outcome is not None:
                answers.append((name, outcome))
        return answers

    async def on_request(self, event: RequestEvent) -> list[tuple[str, RequestDecision]]:
        return [(name, d) for name, d in await self._ask("on_request", event) if isinstance(d, RequestDecision)]

    async def on_tool_call(self, event: ToolCallEvent) -> list[tuple[str, ToolCallDecision]]:
        return [(name, d) for name, d in await self._ask("on_tool_call", event) if isinstance(d, ToolCallDecision)]

    async def on_response(self, event: ResponseEvent) -> list[tuple[str, ResponseDecision]]:
        return [(name, d) for name, d in await self._ask("on_response", event) if isinstance(d, ResponseDecision)]


class TrafficHooks:
    """One request's observers, conversation, caller, and what they decided so far.

    Built by ``prepare_gateway_tools`` when at least one observer is loaded, kept
    on the request context, and read at settlement for the usage row.
    """

    def __init__(self, observers: TrafficObservers, caller: Caller, conversation: Conversation) -> None:
        self.observers = observers
        self.caller = caller
        self.conversation = conversation
        self.annotations: dict[str, dict[str, Any]] = {}
        self.decisions: list[tuple[str, RequestDecision | ToolCallDecision | ResponseDecision]] = []
        self.blocked: tuple[str, str] | None = None
        self.injected_system: str = ""
        self.denied: dict[str, str] = {}

    def _merge(self, name: str, annotations: dict[str, Any]) -> None:
        if not annotations:
            return
        # The row's column is JSON, and the writer drops the whole batch it is in
        # when one value cannot be serialized; refuse here, per plugin, instead.
        if not isinstance(annotations, dict):
            logger.warning("Plugin %s: annotations must be a dict, got %s; skipped", name, type(annotations).__name__)
            return
        try:
            json.dumps(annotations)
        except (TypeError, ValueError) as exc:
            logger.warning("Plugin %s: annotations are not JSON-serializable and were skipped: %s", name, exc)
            return
        current = self.annotations.setdefault(name, {})
        for key, value in annotations.items():
            if isinstance(value, list) and isinstance(current.get(key), list):
                current[key] = [*current[key], *value]
            else:
                current[key] = value

    async def request(self) -> None:
        injected: list[str] = []
        for name, decision in await self.observers.on_request(RequestEvent(self.caller, self.conversation)):
            self.decisions.append((name, decision))
            self._merge(name, decision.annotations)
            if decision.inject_system:
                injected.append(decision.inject_system)
                self._merge(name, {"injected_system": True})
            if decision.block and self.blocked is None:
                self.blocked = (name, decision.block)
                self._merge(name, {"blocked": decision.block})
        self.injected_system = "\n\n".join(injected)

    async def tool_call(self, tool_call: ToolCall) -> str | None:
        """Ask about one tool call; return the denial message, if any observer denied it."""
        event = ToolCallEvent(self.caller, self.conversation, tool_call)
        denial: str | None = None
        for name, decision in await self.observers.on_tool_call(event):
            self.decisions.append((name, decision))
            self._merge(name, decision.annotations)
            if decision.deny and denial is None:
                denial = decision.deny
                self.denied[tool_call.id] = denial
                self._merge(
                    name, {"denied": [{"tool_call_id": tool_call.id, "name": tool_call.name, "message": denial}]}
                )
        return denial

    async def response(self, text: str, tool_calls: Iterable[ToolCall], *, streamed: bool) -> str | None:
        """Ask about the finished answer; return a block message, which applies when not streamed."""
        event = ResponseEvent(self.caller, self.conversation, text, tuple(tool_calls), streamed)
        block: str | None = None
        for name, decision in await self.observers.on_response(event):
            self.decisions.append((name, decision))
            self._merge(name, decision.annotations)
            if decision.block and block is None:
                block = decision.block
                self._merge(name, {"would_block" if streamed else "blocked_response": decision.block})
        return block

    async def result(self, api: Api, result: Any) -> tuple[Any, str | None]:
        """Ask about every tool call in a non-streaming result, apply denials, then ask about the answer.

        Returns the result, rewritten where a call was denied, and the response
        block message if an observer blocked the answer.
        """
        denied: dict[str, str] = {}
        for tool_call in tool_calls_from_result(api, result):
            message = await self.tool_call(tool_call)
            if message is not None:
                denied[tool_call.id] = message
        if denied:
            result = deny_in_result(api, result, denied)
        remaining = [call for call in tool_calls_from_result(api, result)]
        block = await self.response(response_text(api, result), remaining, streamed=False)
        return result, block

    def observe_stream(self, api: Api, stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Pass a provider stream through, holding each tool call back until it is judged."""
        gate = stream_gate_for(api)

        async def _observed() -> AsyncIterator[Any]:
            survivors: list[ToolCall] = []
            async for chunk in stream:
                for out in await gate.feed(chunk, self.tool_call):
                    yield out
                survivors.extend(gate.take_survivors())
            for out in await gate.finish(self.tool_call):
                yield out
            survivors.extend(gate.take_survivors())
            await self.response(gate.text(), survivors, streamed=True)

        return _observed()

    def annotations_or_none(self) -> dict[str, dict[str, Any]] | None:
        return self.annotations or None


def denial_text(message: str) -> str:
    """The text that stands in for a denied tool call, so the model and the user both see why."""
    return f"[Otari refused this tool call: {message}]"


def inject_system_text(api: Api, kwargs: dict[str, Any], text: str) -> dict[str, Any]:
    """Prepend ``text`` to the system slot of a provider call, in the API's own shape."""
    if not text:
        return kwargs
    out = {**kwargs}
    if api == "chat":
        messages = list(out.get("messages") or [])
        out["messages"] = [{"role": "system", "content": text}, *messages]
    elif api == "messages":
        existing = out.get("system")
        if existing is None or existing == "":
            out["system"] = text
        elif isinstance(existing, str):
            out["system"] = f"{text}\n\n{existing}"
        elif isinstance(existing, list):
            out["system"] = [{"type": "text", "text": text}, *existing]
        else:
            out["system"] = text
    else:
        existing = out.get("instructions")
        out["instructions"] = f"{text}\n\n{existing}" if isinstance(existing, str) and existing else text
    return out


# ---------------------------------------------------------------------------
# Session identity
# ---------------------------------------------------------------------------


def session_key(*candidates: str | None, system: str = "", first_user_text: str = "") -> str:
    """The first explicit identifier a request carries, else a digest of how it began.

    A digest of the system prompt and first user turn is stable for one agent
    session, since neither changes as the conversation grows, and different for
    two sessions that started differently. It is best effort: two sessions that
    began identically collide.
    """
    for candidate in candidates:
        if candidate:
            return str(candidate)[:200]
    digest = hashlib.sha256(f"{system}\x1f{first_user_text}".encode()).hexdigest()[:16]
    return f"anon-{digest}"


# ---------------------------------------------------------------------------
# Normalizing a request into a Conversation
# ---------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    """Flatten a content value (string, list of blocks, or nested) into text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "input_text", "output_text", "content", "output"):
            if key in content:
                return _text_of(content[key])
        return ""
    if isinstance(content, list):
        return "\n".join(part for part in (_text_of(item) for item in content) if part)
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    return str(content)


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {}


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


class _TurnBuilder:
    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self._open: dict[str, Any] | None = None

    def assistant(self, text: str, tool_calls: list[ToolCall]) -> None:
        self._close()
        self._open = {"text": text, "tool_calls": tool_calls, "results": []}

    def tool_call(self, tool_call: ToolCall) -> None:
        if self._open is None:
            self._open = {"text": "", "tool_calls": [], "results": []}
        self._open["tool_calls"].append(tool_call)

    def result(self, result: ToolResult) -> None:
        if self._open is None:
            self._open = {"text": "", "tool_calls": [], "results": []}
        self._open["results"].append(result)

    def _close(self) -> None:
        if self._open is not None:
            self.turns.append(
                Turn(
                    text=self._open["text"],
                    tool_calls=tuple(self._open["tool_calls"]),
                    results=tuple(self._open["results"]),
                )
            )
            self._open = None

    def build(self) -> tuple[Turn, ...]:
        self._close()
        return tuple(self.turns)


def conversation_from_chat(model: str, messages: list[Any], *, session: str) -> Conversation:
    """OpenAI chat shape: ``system``/``user``/``assistant``/``tool`` roles."""
    system_parts: list[str] = []
    builder = _TurnBuilder()
    first_user = ""
    latest_user = ""
    for message in messages or []:
        role = _get(message, "role")
        content = _get(message, "content")
        if role in ("system", "developer"):
            system_parts.append(_text_of(content))
        elif role == "user":
            text = _text_of(content)
            first_user = first_user or text
            latest_user = text or latest_user
        elif role == "assistant":
            calls: list[ToolCall] = []
            for raw in _get(message, "tool_calls") or []:
                function = _get(raw, "function") or {}
                calls.append(
                    ToolCall(
                        id=str(_get(raw, "id") or ""),
                        name=str(_get(function, "name") or ""),
                        arguments=_parse_arguments(_get(function, "arguments")),
                    )
                )
            builder.assistant(_text_of(content), calls)
        elif role == "tool":
            builder.result(ToolResult(call_id=str(_get(message, "tool_call_id") or ""), content=_text_of(content)))
    system = "\n".join(part for part in system_parts if part)
    return Conversation(
        api="chat",
        model=model,
        system=system,
        turns=builder.build(),
        session_key=session_key(session, system=system, first_user_text=first_user),
        latest_user_text=latest_user,
    )


def conversation_from_messages(model: str, system: Any, messages: list[Any], *, session: str) -> Conversation:
    """Anthropic messages shape: ``tool_use`` and ``tool_result`` content blocks."""
    system_text = _text_of(system)
    builder = _TurnBuilder()
    first_user = ""
    latest_user = ""
    for message in messages or []:
        role = _get(message, "role")
        content = _get(message, "content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": _text_of(content)}]
        if role == "assistant":
            texts: list[str] = []
            calls: list[ToolCall] = []
            for block in blocks:
                kind = _get(block, "type")
                if kind == "tool_use":
                    calls.append(
                        ToolCall(
                            id=str(_get(block, "id") or ""),
                            name=str(_get(block, "name") or ""),
                            arguments=_parse_arguments(_get(block, "input")),
                        )
                    )
                elif kind == "text":
                    texts.append(_text_of(_get(block, "text")))
            builder.assistant("\n".join(t for t in texts if t), calls)
        elif role == "user":
            texts = []
            for block in blocks:
                if _get(block, "type") == "tool_result":
                    builder.result(
                        ToolResult(
                            call_id=str(_get(block, "tool_use_id") or ""),
                            content=_text_of(_get(block, "content")),
                            is_error=bool(_get(block, "is_error", False)),
                        )
                    )
                else:
                    texts.append(_text_of(block))
            text = "\n".join(t for t in texts if t)
            if text:
                first_user = first_user or text
                latest_user = text
    return Conversation(
        api="messages",
        model=model,
        system=system_text,
        turns=builder.build(),
        session_key=session_key(session, system=system_text, first_user_text=first_user),
        latest_user_text=latest_user,
    )


def conversation_from_responses(model: str, instructions: Any, input_data: Any, *, session: str) -> Conversation:
    """OpenAI responses shape: message items plus ``function_call`` / ``function_call_output`` items."""
    system_text = _text_of(instructions)
    builder = _TurnBuilder()
    first_user = ""
    latest_user = ""
    items = input_data if isinstance(input_data, list) else [{"type": "message", "role": "user", "content": input_data}]
    for item in items:
        kind = _get(item, "type") or "message"
        if kind == "message":
            role = _get(item, "role")
            text = _text_of(_get(item, "content"))
            if role == "assistant":
                builder.assistant(text, [])
            elif role in ("system", "developer"):
                system_text = "\n".join(part for part in (system_text, text) if part)
            else:
                first_user = first_user or text
                latest_user = text or latest_user
        elif kind == "function_call":
            builder.tool_call(
                ToolCall(
                    id=str(_get(item, "call_id") or _get(item, "id") or ""),
                    name=str(_get(item, "name") or ""),
                    arguments=_parse_arguments(_get(item, "arguments")),
                )
            )
        elif kind == "function_call_output":
            builder.result(ToolResult(call_id=str(_get(item, "call_id") or ""), content=_text_of(_get(item, "output"))))
    return Conversation(
        api="responses",
        model=model,
        system=system_text,
        turns=builder.build(),
        session_key=session_key(session, system=system_text, first_user_text=first_user),
        latest_user_text=latest_user,
    )


# ---------------------------------------------------------------------------
# Tool calls out of a result, complete or streamed
# ---------------------------------------------------------------------------


def _chat_tool_call(raw: Any) -> ToolCall:
    function = _get(raw, "function") or {}
    return ToolCall(
        id=str(_get(raw, "id") or ""),
        name=str(_get(function, "name") or ""),
        arguments=_parse_arguments(_get(function, "arguments")),
    )


def _messages_tool_call(block: Any) -> ToolCall:
    return ToolCall(
        id=str(_get(block, "id") or ""),
        name=str(_get(block, "name") or ""),
        arguments=_parse_arguments(_get(block, "input")),
    )


def _responses_tool_call(item: Any) -> ToolCall:
    return ToolCall(
        id=str(_get(item, "call_id") or _get(item, "id") or ""),
        name=str(_get(item, "name") or ""),
        arguments=_parse_arguments(_get(item, "arguments")),
    )


def tool_calls_from_result(api: Api, result: Any) -> list[ToolCall]:
    """Every tool call a non-streaming result carries."""
    calls: list[ToolCall] = []
    if api == "chat":
        for choice in _get(result, "choices") or []:
            message = _get(choice, "message")
            calls.extend(_chat_tool_call(raw) for raw in _get(message, "tool_calls") or [])
    elif api == "messages":
        calls.extend(_messages_tool_call(b) for b in _get(result, "content") or [] if _get(b, "type") == "tool_use")
    elif api == "responses":
        calls.extend(
            _responses_tool_call(i) for i in _get(result, "output") or [] if _get(i, "type") == "function_call"
        )
    return calls


def response_text(api: Api, result: Any) -> str:
    """The assistant text a non-streaming result carries."""
    if api == "chat":
        parts = [_text_of(_get(_get(choice, "message"), "content")) for choice in _get(result, "choices") or []]
    elif api == "messages":
        parts = [_text_of(_get(b, "text")) for b in _get(result, "content") or [] if _get(b, "type") == "text"]
    else:
        parts = [_text_of(_get(i, "content")) for i in _get(result, "output") or [] if _get(i, "type") == "message"]
    return "\n".join(part for part in parts if part)


def _set(item: Any, key: str, value: Any) -> None:
    if isinstance(item, dict):
        item[key] = value
    else:
        setattr(item, key, value)


def _text_block(api: Api, text: str, *, call_id: str = "") -> Any:
    """A text block or item in the API's own type, for the place a denied call held."""
    if api == "messages":
        from anthropic.types import TextBlock

        return TextBlock(type="text", text=text)
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    return ResponseOutputMessage(
        id=f"msg_refused_{call_id or 'call'}",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        role="assistant",
        status="completed",
        type="message",
    )


def deny_in_result(api: Api, result: Any, denied: dict[str, str]) -> Any:
    """Rewrite a non-streaming result so each denied call is replaced by its denial text.

    Mutates ``result`` in place and returns it. A shape this does not know is
    returned untouched, with the denial still on the usage row.
    """
    try:
        if api == "chat":
            for choice in _get(result, "choices") or []:
                message = _get(choice, "message")
                calls = _get(message, "tool_calls") or []
                kept = [raw for raw in calls if str(_get(raw, "id") or "") not in denied]
                if len(kept) == len(calls):
                    continue
                texts = [denial_text(denied[str(_get(raw, "id"))]) for raw in calls if str(_get(raw, "id")) in denied]
                existing = _text_of(_get(message, "content"))
                _set(message, "content", "\n".join(part for part in (existing, *texts) if part))
                _set(message, "tool_calls", kept or None)
                if not kept and _get(choice, "finish_reason") == "tool_calls":
                    _set(choice, "finish_reason", "stop")
        elif api == "messages":
            content = list(_get(result, "content") or [])
            rewritten: list[Any] = []
            remaining = 0
            for block in content:
                if _get(block, "type") == "tool_use" and str(_get(block, "id") or "") in denied:
                    rewritten.append(_text_block(api, denial_text(denied[str(_get(block, "id"))])))
                else:
                    remaining += _get(block, "type") == "tool_use"
                    rewritten.append(block)
            _set(result, "content", rewritten)
            if not remaining and _get(result, "stop_reason") == "tool_use":
                _set(result, "stop_reason", "end_turn")
        elif api == "responses":
            output = list(_get(result, "output") or [])
            rewritten = []
            for item in output:
                call_id = str(_get(item, "call_id") or _get(item, "id") or "")
                if _get(item, "type") == "function_call" and call_id in denied:
                    rewritten.append(_text_block(api, denial_text(denied[call_id]), call_id=call_id))
                else:
                    rewritten.append(item)
            _set(result, "output", rewritten)
    except Exception:  # noqa: BLE001 a shape the rewrite cannot handle is served as it came
        logger.exception("Traffic seam: could not apply a tool call denial to a %s result", api)
    return result


@dataclass(frozen=True)
class Completed:
    """A tool call rebuilt from a stream, with the key the stream identifies it by."""

    key: Any
    tool_call: ToolCall


class ToolCallAssembler(Protocol):
    """Rebuilds tool calls from a stream's fragments; ``feed`` returns the ones that just completed."""

    def feed(self, chunk: Any) -> list[Completed]: ...

    def finish(self) -> list[Completed]: ...


class _Partial:
    __slots__ = ("arguments", "id", "name")

    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments: list[str] = []

    def build(self) -> ToolCall:
        return ToolCall(id=self.id, name=self.name, arguments=_parse_arguments("".join(self.arguments)))


class ChatToolCallAssembler:
    """``delta.tool_calls[i]`` fragments, keyed by index, flushed on ``finish_reason``."""

    def __init__(self) -> None:
        self._partials: dict[int, _Partial] = {}

    def feed(self, chunk: Any) -> list[Completed]:
        completed: list[Completed] = []
        for choice in _get(chunk, "choices") or []:
            delta = _get(choice, "delta")
            for raw in _get(delta, "tool_calls") or []:
                index = int(_get(raw, "index") or 0)
                partial = self._partials.setdefault(index, _Partial())
                if _get(raw, "id"):
                    partial.id = str(_get(raw, "id"))
                function = _get(raw, "function")
                if function is not None:
                    if _get(function, "name"):
                        partial.name = str(_get(function, "name"))
                    if _get(function, "arguments"):
                        partial.arguments.append(str(_get(function, "arguments")))
            if _get(choice, "finish_reason"):
                completed.extend(self.finish())
        return completed

    def finish(self) -> list[Completed]:
        done = [Completed(index, self._partials[index].build()) for index in sorted(self._partials)]
        self._partials.clear()
        return done


class MessagesToolCallAssembler:
    """``content_block_start`` (tool_use) through ``content_block_stop``, keyed by block index."""

    def __init__(self) -> None:
        self._partials: dict[int, _Partial] = {}

    def feed(self, chunk: Any) -> list[Completed]:
        kind = _get(chunk, "type")
        index = int(_get(chunk, "index") or 0)
        if kind == "content_block_start":
            block = _get(chunk, "content_block")
            if _get(block, "type") == "tool_use":
                partial = _Partial()
                partial.id = str(_get(block, "id") or "")
                partial.name = str(_get(block, "name") or "")
                initial = _get(block, "input")
                if isinstance(initial, dict) and initial:
                    partial.arguments.append(json.dumps(initial))
                self._partials[index] = partial
        elif kind == "content_block_delta" and index in self._partials:
            delta = _get(chunk, "delta")
            if _get(delta, "type") == "input_json_delta":
                self._partials[index].arguments.append(str(_get(delta, "partial_json") or ""))
        elif kind == "content_block_stop" and index in self._partials:
            return [Completed(index, self._partials.pop(index).build())]
        return []

    def finish(self) -> list[Completed]:
        done = [Completed(index, self._partials[index].build()) for index in sorted(self._partials)]
        self._partials.clear()
        return done


class ResponsesToolCallAssembler:
    """``response.output_item.done`` items of type ``function_call``, which arrive whole."""

    def feed(self, chunk: Any) -> list[Completed]:
        if _get(chunk, "type") != "response.output_item.done":
            return []
        item = _get(chunk, "item")
        if _get(item, "type") != "function_call":
            return []
        return [Completed(int(_get(chunk, "output_index") or 0), _responses_tool_call(item))]

    def finish(self) -> list[Completed]:
        return []


_ASSEMBLERS: dict[str, Callable[[], ToolCallAssembler]] = {
    "chat": ChatToolCallAssembler,
    "messages": MessagesToolCallAssembler,
    "responses": ResponsesToolCallAssembler,
}


def assembler_for(api: Api) -> ToolCallAssembler:
    return _ASSEMBLERS[api]()


# ---------------------------------------------------------------------------
# Stream gates: hold a tool call's fragments until it is judged, then let it
# through or put the denial text in its place
# ---------------------------------------------------------------------------

Judge = Callable[[ToolCall], Any]


def _copy(item: Any, **updates: Any) -> Any:
    if isinstance(item, dict):
        return {**item, **updates}
    return item.model_copy(update=updates)


class StreamGate(Protocol):
    async def feed(self, chunk: Any, judge: Judge) -> list[Any]: ...

    async def finish(self, judge: Judge) -> list[Any]: ...

    def take_survivors(self) -> list[ToolCall]: ...

    def text(self) -> str: ...


class _GateBase:
    def __init__(self) -> None:
        self._survivors: list[ToolCall] = []
        self._text: list[str] = []

    def take_survivors(self) -> list[ToolCall]:
        out, self._survivors = self._survivors, []
        return out

    def text(self) -> str:
        return "".join(self._text)

    async def _judge_all(self, judge: Judge, completed: list[Completed]) -> dict[Any, str]:
        denied: dict[Any, str] = {}
        for done in completed:
            message = await judge(done.tool_call)
            if message is None:
                self._survivors.append(done.tool_call)
            else:
                denied[done.key] = message
        return denied


class ChatStreamGate(_GateBase):
    """Holds every chunk from the first tool-call fragment to the finish, then rewrites by index."""

    def __init__(self) -> None:
        super().__init__()
        self._assembler = ChatToolCallAssembler()
        self._held: list[Any] = []

    def _carries_tool_calls(self, chunk: Any) -> bool:
        return any(_get(_get(choice, "delta"), "tool_calls") for choice in _get(chunk, "choices") or [])

    async def feed(self, chunk: Any, judge: Judge) -> list[Any]:
        for choice in _get(chunk, "choices") or []:
            content = _get(_get(choice, "delta"), "content")
            if isinstance(content, str):
                self._text.append(content)
        completed = self._assembler.feed(chunk)
        holding = bool(self._held) or self._carries_tool_calls(chunk)
        if holding:
            self._held.append(chunk)
        if completed:
            return await self._release(judge, completed)
        return [] if holding else [chunk]

    async def finish(self, judge: Judge) -> list[Any]:
        completed = self._assembler.finish()
        if completed or self._held:
            return await self._release(judge, completed)
        return []

    async def _release(self, judge: Judge, completed: list[Completed]) -> list[Any]:
        denied = await self._judge_all(judge, completed)
        held, self._held = self._held, []
        if not denied:
            return held
        try:
            return self._rewrite(held, denied, all_denied=len(denied) == len(completed))
        except Exception:  # noqa: BLE001 served as it came; the denial is on the usage row
            logger.exception("Traffic seam: could not apply a tool call denial to a chat stream")
            return held

    def _rewrite(self, held: list[Any], denied: dict[Any, str], *, all_denied: bool) -> list[Any]:
        # Survivors are renumbered from zero: a client that accumulates fragments
        # into a list by ``index`` (the OpenAI SDK's stream helper does) cannot
        # take a gap where the denied call was.
        surviving = sorted(
            {
                int(_get(raw, "index") or 0)
                for chunk in held
                for choice in _get(chunk, "choices") or []
                for raw in _get(_get(choice, "delta"), "tool_calls") or []
                if int(_get(raw, "index") or 0) not in denied
            }
        )
        renumbered = {old: new for new, old in enumerate(surviving)}
        out: list[Any] = []
        template: Any = None
        for chunk in held:
            choices = []
            for choice in _get(chunk, "choices") or []:
                delta = _get(choice, "delta")
                calls = _get(delta, "tool_calls") or []
                kept = [
                    _copy(raw, index=renumbered[int(_get(raw, "index") or 0)])
                    for raw in calls
                    if int(_get(raw, "index") or 0) not in denied
                ]
                if len(kept) != len(calls) or any(renumbered[i] != i for i in renumbered):
                    delta = _copy(delta, tool_calls=kept or None)
                    template = template if template is not None or len(kept) == len(calls) else chunk
                finish = _get(choice, "finish_reason")
                if all_denied and finish == "tool_calls":
                    finish = "stop"
                empty = not kept and not _get(delta, "content") and finish is None and not _get(chunk, "usage")
                if empty:
                    continue
                choices.append(_copy(choice, delta=delta, finish_reason=finish))
            if choices or _get(chunk, "usage"):
                out.append(_copy(chunk, choices=choices))
        if template is not None:
            text = "\n".join(denial_text(message) for _, message in sorted(denied.items(), key=lambda kv: str(kv[0])))
            first_choice = (_get(template, "choices") or [None])[0]
            if first_choice is not None:
                delta = _copy(_get(first_choice, "delta"), content=text, tool_calls=None)
                notice = _copy(template, choices=[_copy(first_choice, delta=delta, finish_reason=None)], usage=None)
                # Before the finish chunk, so the finish reason stays last.
                position = next(
                    (i for i, c in enumerate(out) if any(_get(ch, "finish_reason") for ch in _get(c, "choices") or [])),
                    len(out),
                )
                out.insert(position, notice)
        return out


class MessagesStreamGate(_GateBase):
    """Holds a ``tool_use`` block's events by index until its ``content_block_stop``."""

    def __init__(self) -> None:
        super().__init__()
        self._assembler = MessagesToolCallAssembler()
        self._held: dict[int, list[Any]] = {}
        self._seen = 0
        self._denied_count = 0

    async def feed(self, chunk: Any, judge: Judge) -> list[Any]:
        kind = _get(chunk, "type")
        index = int(_get(chunk, "index") or 0)
        if kind == "content_block_delta" and _get(_get(chunk, "delta"), "type") == "text_delta":
            self._text.append(str(_get(_get(chunk, "delta"), "text") or ""))
        if kind == "content_block_start" and _get(_get(chunk, "content_block"), "type") == "tool_use":
            self._held[index] = [chunk]
            self._seen += 1
            self._assembler.feed(chunk)
            return []
        if index in self._held and kind in ("content_block_delta", "content_block_stop"):
            self._held[index].append(chunk)
            completed = self._assembler.feed(chunk)
            if not completed:
                return []
            denied = await self._judge_all(judge, completed)
            held = self._held.pop(index)
            if not denied:
                return held
            self._denied_count += 1
            return self._replacement(index, denied[index], held[0])
        if kind == "message_delta" and self._seen and self._denied_count == self._seen:
            delta = _get(chunk, "delta")
            if _get(delta, "stop_reason") == "tool_use":
                try:
                    return [_copy(chunk, delta=_copy(delta, stop_reason="end_turn"))]
                except Exception:  # noqa: BLE001 the stop reason is cosmetic next to the block itself
                    logger.exception("Traffic seam: could not rewrite a messages stop reason")
        return [chunk]

    async def finish(self, judge: Judge) -> list[Any]:
        out: list[Any] = []
        for index, held in sorted(self._held.items()):
            denied = await self._judge_all(judge, self._assembler.finish() or [])
            out.extend(self._replacement(index, denied[index], held[0]) if index in denied else held)
        self._held.clear()
        return out

    def _replacement(self, index: int, message: str, start_event: Any) -> list[Any]:
        try:
            from anthropic.types import TextDelta
            from any_llm.types.messages import ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent

            return [
                ContentBlockStartEvent(
                    type="content_block_start", index=index, content_block=_text_block("messages", "")
                ),
                ContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=index,
                    delta=TextDelta(type="text_delta", text=denial_text(message)),
                ),
                ContentBlockStopEvent(type="content_block_stop", index=index),
            ]
        except Exception:  # noqa: BLE001 served as it came; the denial is on the usage row
            logger.exception("Traffic seam: could not apply a tool call denial to a messages stream")
            return [start_event]


class ResponsesStreamGate(_GateBase):
    """Holds a ``function_call`` item's events by output index until its ``output_item.done``."""

    def __init__(self) -> None:
        super().__init__()
        self._assembler = ResponsesToolCallAssembler()
        self._held: dict[int, list[Any]] = {}
        self._denied: dict[str, str] = {}

    async def feed(self, chunk: Any, judge: Judge) -> list[Any]:
        kind = _get(chunk, "type")
        if kind == "response.output_text.delta":
            self._text.append(str(_get(chunk, "delta") or ""))
        index = int(_get(chunk, "output_index") or 0) if _get(chunk, "output_index") is not None else None
        if kind == "response.output_item.added" and _get(_get(chunk, "item"), "type") == "function_call":
            self._held[index or 0] = [chunk]
            return []
        if index is not None and index in self._held:
            self._held[index].append(chunk)
            completed = self._assembler.feed(chunk)
            if not completed:
                return []
            denied = await self._judge_all(judge, completed)
            held = self._held.pop(index)
            if not denied:
                return held
            call = completed[0].tool_call
            self._denied[call.id] = denied[index]
            return self._replacement(index, call.id, denied[index], held)
        if kind in ("response.completed", "response.incomplete", "response.failed") and self._denied:
            response = _get(chunk, "response")
            if response is not None:
                deny_in_result("responses", response, self._denied)
        return [chunk]

    async def finish(self, judge: Judge) -> list[Any]:
        out = [chunk for _, held in sorted(self._held.items()) for chunk in held]
        self._held.clear()
        return out

    def _replacement(self, index: int, call_id: str, message: str, held: list[Any]) -> list[Any]:
        try:
            from openai.types.responses import (
                ResponseOutputItemAddedEvent,
                ResponseOutputItemDoneEvent,
                ResponseTextDeltaEvent,
            )

            text = denial_text(message)
            first, last = held[0], held[-1]
            item = _text_block("responses", text, call_id=call_id)
            return [
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    item=_text_block("responses", "", call_id=call_id),
                    output_index=index,
                    sequence_number=int(_get(first, "sequence_number") or 0),
                ),
                ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    content_index=0,
                    delta=text,
                    item_id=item.id,
                    logprobs=[],
                    output_index=index,
                    sequence_number=int(_get(first, "sequence_number") or 0) + 1,
                ),
                ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    item=item,
                    output_index=index,
                    sequence_number=int(_get(last, "sequence_number") or 0),
                ),
            ]
        except Exception:  # noqa: BLE001 served as it came; the denial is on the usage row
            logger.exception("Traffic seam: could not apply a tool call denial to a responses stream")
            return held


_GATES: dict[str, Callable[[], StreamGate]] = {
    "chat": ChatStreamGate,
    "messages": MessagesStreamGate,
    "responses": ResponsesStreamGate,
}


def stream_gate_for(api: Api) -> StreamGate:
    return _GATES[api]()
