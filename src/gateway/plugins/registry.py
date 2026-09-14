"""Load discovered plugins and hold what each one contributed.

A plugin's ``register`` receives a :class:`PluginContext` and adds routers, CLI
groups, migration directories, observers, backends, hooks, and subscriptions to
it. Everything a plugin adds is recorded on its :class:`LoadedPlugin`;
``gateway.api.main`` mounts the routers, ``gateway.main`` mounts the pages and
runs the hooks and migrations, ``gateway.cli`` attaches the groups, and the
services read the backends. A plugin that fails to import or register is
recorded as failed with its error and skipped, so one broken plugin does not
take the gateway down with it; the dashboard shows the failure.
"""

import asyncio
import importlib
import inspect
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import click
from fastapi import APIRouter

from gateway.log_config import logger
from gateway.models.plugins import (
    PLUGIN_API_VERSION,
    PLUGIN_NAME_PATTERN,
    PluginManifest,
    PluginPageManifest,
    PluginsConfig,
    RuntimeMode,
    setting_value_matches,
)
from gateway.plugins.discovery import DiscoveredPlugin, DiscoveryProblem, PluginSource, discover_plugins
from gateway.plugins.events import EventBus, Handler
from gateway.plugins.guardrails import GuardrailBackend, profile_name
from gateway.services.routing.backends import set_plugin_router_backends
from gateway.version import __version__

if TYPE_CHECKING:
    from gateway.container import Container
    from gateway.core.config import GatewayConfig

PluginStatus = Literal["loaded", "failed", "disabled", "pending_restart"]
Hook = Callable[[], Any]
SettingsListener = Callable[[dict[str, Any]], Any]
ToolBackendFactory = Callable[[], Any]


class PluginError(Exception):
    """Raised by :class:`PluginContext` when a contribution is not usable."""


@dataclass(frozen=True)
class PageContribution:
    """One dashboard page a plugin ships, resolved to the directory it is served from."""

    manifest: PluginPageManifest
    directory: Path

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def label(self) -> str:
        return self.manifest.label


@dataclass
class HealthCheck:
    check: Hook
    critical: bool


@dataclass
class LoadedPlugin:
    """One discovered plugin and what loading it produced."""

    manifest: PluginManifest
    source: PluginSource
    package_dir: Path
    install_dir: Path | None
    status: PluginStatus
    error: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    routers: list[APIRouter] = field(default_factory=list)
    cli_groups: list[click.Group] = field(default_factory=list)
    migrations: list[Path] = field(default_factory=list)
    observers: list[Any] = field(default_factory=list)
    pages: list[PageContribution] = field(default_factory=list)
    startup_hooks: list[Hook] = field(default_factory=list)
    shutdown_hooks: list[Hook] = field(default_factory=list)
    health_checks: list[HealthCheck] = field(default_factory=list)
    subscriptions: list[tuple[str, Handler]] = field(default_factory=list)
    guardrails: dict[str, GuardrailBackend] = field(default_factory=dict)
    tools: dict[str, ToolBackendFactory] = field(default_factory=dict)
    router_backends: dict[str, Any] = field(default_factory=dict)
    settings_listeners: list[SettingsListener] = field(default_factory=list)
    health: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def api_prefix(self) -> str:
        """Where the plugin's routers mount, below the API root."""
        return f"/plugins/{self.name}"

    def page_url(self, page: PageContribution) -> str:
        """Where a page is served: the first page at the bare ``/ui/`` mount, the rest under their id."""
        base = f"/plugins/{self.name}/ui/"
        if self.pages and page is not self.pages[0]:
            base = f"{base}{page.id}/"
        return f"{base}{page.manifest.entry}"

    @property
    def ui(self) -> PageContribution | None:
        """The plugin's first page, or ``None`` when it ships none."""
        return self.pages[0] if self.pages else None

    @property
    def ui_url(self) -> str | None:
        return self.page_url(self.pages[0]) if self.pages else None

    def clear_contributions(self) -> None:
        self.routers.clear()
        self.cli_groups.clear()
        self.migrations.clear()
        self.observers.clear()
        self.pages.clear()
        self.startup_hooks.clear()
        self.shutdown_hooks.clear()
        self.health_checks.clear()
        self.subscriptions.clear()
        self.guardrails.clear()
        self.tools.clear()
        self.router_backends.clear()
        self.settings_listeners.clear()


class PluginContext:
    """What a plugin's ``register(ctx)`` is handed.

    ``config`` is the plugin's own block of ``config.yml`` (``plugins.<name>``)
    over the manifest's setting defaults. It is the live dict: a dashboard edit
    updates it in place and calls every ``on_settings_change`` listener.
    ``container`` is the composition root, so a plugin can resolve a port; it is
    ``None`` when plugins are loaded for the CLI alone, where no app exists.
    """

    def __init__(self, plugin: LoadedPlugin, config: dict[str, Any], container: "Container | None") -> None:
        self._plugin = plugin
        self.name = plugin.name
        self.config = config
        self.container = container

    def add_router(self, router: APIRouter) -> None:
        """Mount ``router`` under ``/api/v1/plugins/<name>``.

        The router's own prefix, if any, applies below that. Authentication is
        the plugin's to declare per route, as the core routers do; mounting adds
        none.
        """
        if not isinstance(router, APIRouter):
            msg = f"plugin {self.name!r} added a router that is not an APIRouter: {router!r}"
            raise PluginError(msg)
        self._plugin.routers.append(router)

    def add_cli(self, group: click.Group) -> None:
        """Attach ``group`` to the ``otari`` command line as a top-level group."""
        if not isinstance(group, click.Group):
            msg = f"plugin {self.name!r} added a CLI group that is not a click.Group: {group!r}"
            raise PluginError(msg)
        if not group.name:
            msg = f"plugin {self.name!r} added a CLI group with no name"
            raise PluginError(msg)
        self._plugin.cli_groups.append(group)

    def add_migrations(self, script_location: Path | str) -> None:
        """Register an Alembic script directory, run against the plugin's own version table."""
        path = Path(script_location)
        if not path.is_dir():
            msg = f"plugin {self.name!r} named a migrations directory that does not exist: {path}"
            raise PluginError(msg)
        self._plugin.migrations.append(path)

    def add_traffic_observer(self, observer: Any) -> None:
        """Watch inference traffic: see ``gateway.plugins.traffic``.

        ``observer`` implements any of ``on_request``, ``on_tool_call``, and
        ``on_response``, sync or async. Its annotations reach the usage row;
        its decisions (block, inject a system text, deny a tool call) apply.
        """
        methods = ("on_request", "on_tool_call", "on_response")
        if not any(callable(getattr(observer, name, None)) for name in methods):
            msg = f"plugin {self.name!r} added a traffic observer with none of {', '.join(methods)}: {observer!r}"
            raise PluginError(msg)
        self._plugin.observers.append(observer)

    def on_startup(self, hook: Hook) -> None:
        """Run ``hook`` (sync or async, no arguments) when the gateway starts, after migrations."""
        self._plugin.startup_hooks.append(_callable(self, hook, "startup hook"))

    def on_shutdown(self, hook: Hook) -> None:
        """Run ``hook`` when the gateway stops."""
        self._plugin.shutdown_hooks.append(_callable(self, hook, "shutdown hook"))

    def add_health_check(self, check: Hook, *, critical: bool = False) -> None:
        """Report into ``/health``: ``check`` returns truthy for healthy, or raises.

        A ``critical`` check that fails takes ``/health/readiness`` to 503; a
        plain one is reported and changes nothing else.
        """
        self._plugin.health_checks.append(HealthCheck(_callable(self, check, "health check"), critical))

    def subscribe(self, event: str, handler: Handler) -> None:
        """Run ``handler(event)`` for every event named ``event`` (or ``"*"``); see ``gateway.plugins.events``."""
        if not isinstance(event, str) or not event:
            msg = f"plugin {self.name!r} subscribed to an event with no name"
            raise PluginError(msg)
        self._plugin.subscriptions.append((event, _callable(self, handler, "event handler")))

    def add_guardrail(self, name: str, backend: GuardrailBackend) -> None:
        """Offer a guardrail profile named ``<plugin>:<name>``; see ``gateway.plugins.guardrails``."""
        _valid_key(self, name, "guardrail")
        if not callable(getattr(backend, "check", None)):
            msg = f"plugin {self.name!r} added a guardrail {name!r} with no check method"
            raise PluginError(msg)
        self._plugin.guardrails[name] = backend

    def add_tool(self, name: str, factory: ToolBackendFactory) -> None:
        """Offer a tool backend the gateway runs for the model when a request opts in.

        ``factory`` is called per request and returns an object with the
        ``ToolBackend`` members (``openai_tools``, ``owns_tool``, ``call_tool``,
        ``purpose_hints``); an async context manager is entered for the request.
        A request opts in with a tool entry ``{"type": "plugin", "name": "<plugin>:<name>"}``.
        """
        _valid_key(self, name, "tool")
        if not callable(factory):
            msg = f"plugin {self.name!r} added a tool {name!r} whose factory is not callable"
            raise PluginError(msg)
        self._plugin.tools[name] = factory

    def add_router_backend(self, name: str, backend: Any) -> None:
        """Offer a router backend a routing policy names as ``backend: <plugin>:<name>``."""
        _valid_key(self, name, "router backend")
        if not callable(getattr(backend, "rank", None)):
            msg = f"plugin {self.name!r} added a router backend {name!r} with no rank method"
            raise PluginError(msg)
        self._plugin.router_backends[name] = backend

    def on_settings_change(self, listener: SettingsListener) -> None:
        """Run ``listener(config)`` after the dashboard changes the plugin's settings."""
        self._plugin.settings_listeners.append(_callable(self, listener, "settings listener"))


def _callable(ctx: PluginContext, value: Any, what: str) -> Any:
    if not callable(value):
        msg = f"plugin {ctx.name!r} added a {what} that is not callable: {value!r}"
        raise PluginError(msg)
    return value


def _valid_key(ctx: PluginContext, name: str, what: str) -> None:
    if not isinstance(name, str) or not PLUGIN_NAME_PATTERN.fullmatch(name):
        msg = f"plugin {ctx.name!r} added a {what} whose name {name!r} does not match {PLUGIN_NAME_PATTERN.pattern}"
        raise PluginError(msg)


class PluginRegistry:
    """Every plugin this process discovered, loaded or not, in load order."""

    def __init__(
        self,
        directory: Path,
        plugins: list[LoadedPlugin],
        problems: list[DiscoveryProblem],
        *,
        event_timeout_ms: int = 5_000,
        hook_timeout_ms: int = 30_000,
        health_timeout_ms: int = 5_000,
        health_cache_seconds: float = 10.0,
    ) -> None:
        self.directory = directory
        self._plugins = plugins
        self.problems = problems
        self._hook_timeout = hook_timeout_ms / 1000
        self._health_timeout = health_timeout_ms / 1000
        self._health_cache_seconds = health_cache_seconds
        self._health_report: tuple[float, dict[str, str], bool] | None = None
        self.events = EventBus(
            ((plugin.name, name, handler) for plugin in self.loaded() for name, handler in plugin.subscriptions),
            timeout_ms=event_timeout_ms,
        )

    def __iter__(self) -> Iterator[LoadedPlugin]:
        return iter(self._plugins)

    def __len__(self) -> int:
        return len(self._plugins)

    def get(self, name: str) -> LoadedPlugin | None:
        return next((plugin for plugin in self._plugins if plugin.name == name), None)

    def loaded(self) -> list[LoadedPlugin]:
        return [plugin for plugin in self._plugins if plugin.status == "loaded"]

    def traffic_observers(self) -> list[tuple[str, Any]]:
        """Every loaded plugin's traffic observers, as (plugin name, observer) pairs."""
        return [(plugin.name, observer) for plugin in self.loaded() for observer in plugin.observers]

    def guardrail_backends(self) -> dict[str, GuardrailBackend]:
        """Every loaded plugin's guardrails, keyed by the profile that names them."""
        return {
            profile_name(plugin.name, name): backend
            for plugin in self.loaded()
            for name, backend in plugin.guardrails.items()
        }

    def tool_backends(self) -> dict[str, ToolBackendFactory]:
        """Every loaded plugin's tools, keyed ``<plugin>:<name>``."""
        return {f"{plugin.name}:{name}": factory for plugin in self.loaded() for name, factory in plugin.tools.items()}

    def router_backends(self) -> dict[str, Any]:
        """Every loaded plugin's router backends, keyed ``<plugin>:<name>``."""
        return {
            f"{plugin.name}:{name}": backend
            for plugin in self.loaded()
            for name, backend in plugin.router_backends.items()
        }

    def pages(self) -> list[tuple[LoadedPlugin, PageContribution]]:
        return [(plugin, page) for plugin in self.loaded() for page in plugin.pages]

    async def _run_hook(self, hook: Hook, timeout: float | None = None) -> Any:
        outcome = hook()
        if inspect.isawaitable(outcome):
            outcome = await asyncio.wait_for(outcome, timeout=timeout if timeout is not None else self._hook_timeout)
        return outcome

    def _withdraw(self, plugin: LoadedPlugin) -> None:
        """Take a plugin that failed after load out of the registries built at load."""
        self.events.remove_plugin(plugin.name)
        set_plugin_router_backends(self.router_backends())

    async def startup(self) -> None:
        """Run every loaded plugin's startup hooks; a plugin whose hook fails is marked failed."""
        for plugin in self.loaded():
            for hook in plugin.startup_hooks:
                try:
                    await self._run_hook(hook)
                except Exception as error:  # noqa: BLE001 one plugin's failure must not stop the boot
                    logger.error("Plugin %s failed at startup: %s\n%s", plugin.name, error, traceback.format_exc())
                    plugin.status = "failed"
                    plugin.error = f"startup hook: {type(error).__name__}: {error}"
                    self._withdraw(plugin)
                    break

    async def shutdown(self) -> None:
        for plugin in self._plugins:
            for hook in plugin.shutdown_hooks:
                try:
                    await self._run_hook(hook)
                except Exception:  # noqa: BLE001 shutdown runs every hook regardless
                    logger.exception("Plugin %s failed at shutdown", plugin.name)
        await self.events.drain()

    async def _check_plugin(self, plugin: LoadedPlugin) -> tuple[str, bool]:
        problems: list[str] = []
        critical_failure = False
        for check in plugin.health_checks:
            try:
                outcome = await self._run_hook(check.check, self._health_timeout)
                if outcome is not None and not outcome:
                    raise RuntimeError("check returned a falsy value")  # noqa: TRY301
            except Exception as error:  # noqa: BLE001 a failing check is a report, not a crash
                problems.append(f"{type(error).__name__}: {error}")
                critical_failure = critical_failure or check.critical
        status = "ok" if not problems else "failing: " + "; ".join(problems)
        plugin.health = {"status": status}
        return status, critical_failure

    async def check_health(self) -> tuple[dict[str, str], bool]:
        """Every loaded plugin's health, and whether a critical check failed.

        Plugins are checked concurrently, each check under ``plugins.health_timeout_ms``,
        and the report is served from memory for a few seconds afterwards: the
        health routes are unauthenticated, so a caller must not be able to fan
        the checks out as fast as it can send requests.
        """
        now = time.monotonic()
        if self._health_report is not None and now - self._health_report[0] < self._health_cache_seconds:
            return self._health_report[1], self._health_report[2]
        checked = [plugin for plugin in self.loaded() if plugin.health_checks]
        outcomes = await asyncio.gather(*(self._check_plugin(plugin) for plugin in checked))
        report = {plugin.name: status for plugin, (status, _) in zip(checked, outcomes, strict=True)}
        critical_failure = any(critical for _, critical in outcomes)
        self._health_report = (now, report, critical_failure)
        return report, critical_failure

    async def apply_settings(self, name: str, values: dict[str, Any]) -> LoadedPlugin | None:
        """Update a loaded plugin's live config in place and tell its listeners."""
        plugin = self.get(name)
        if plugin is None or plugin.status != "loaded":
            return plugin
        plugin.config.update(values)
        for listener in plugin.settings_listeners:
            try:
                outcome = listener(plugin.config)
                if inspect.isawaitable(outcome):
                    await asyncio.wait_for(outcome, timeout=self._hook_timeout)
            except Exception:  # noqa: BLE001 the write already happened; the listener is best effort
                logger.exception("Plugin %s: settings listener raised", plugin.name)
        return plugin

    def record_pending(self, manifest: PluginManifest, package_dir: Path, install_dir: Path) -> LoadedPlugin:
        """Record a plugin installed into the directory since startup.

        It takes effect on the next start; until then it is listed as pending so
        the dashboard can say so rather than show nothing.
        """
        pending = LoadedPlugin(
            manifest=manifest,
            source="directory",
            package_dir=package_dir,
            install_dir=install_dir,
            status="pending_restart",
        )
        self._plugins = [plugin for plugin in self._plugins if plugin.name != manifest.name]
        self._plugins.append(pending)
        return pending

    def forget(self, name: str) -> None:
        """Drop a plugin from the listing after its directory was removed.

        A loaded plugin's routes stay mounted until restart; the entry is kept
        as pending so the listing says a restart is owed.
        """
        for plugin in self._plugins:
            if plugin.name == name and plugin.status == "loaded":
                plugin.status = "pending_restart"
                plugin.error = "removed; restart the gateway to unload it"
                return
        self._plugins = [plugin for plugin in self._plugins if plugin.name != name]

    @property
    def summary(self) -> str:
        loaded = ", ".join(f"{plugin.name} {plugin.manifest.version}" for plugin in self.loaded())
        failed = ", ".join(plugin.name for plugin in self._plugins if plugin.status == "failed")
        parts = [f"loaded {loaded}" if loaded else "no plugins loaded"]
        if failed:
            parts.append(f"failed {failed}")
        if self.problems:
            parts.append(f"{len(self.problems)} undescribable")
        return "; ".join(parts)


def _actual_contributions(plugin: LoadedPlugin) -> set[str]:
    """What a plugin registered, in the manifest's vocabulary."""
    actual: set[str] = set()
    if plugin.routers:
        actual.add("routes")
    if plugin.cli_groups:
        actual.add("cli")
    if plugin.migrations:
        actual.add("migrations")
    if plugin.observers:
        actual.add("traffic")
    if plugin.manifest.pages:
        actual.add("ui")
    if plugin.startup_hooks or plugin.shutdown_hooks or plugin.health_checks:
        actual.add("lifecycle")
    if plugin.subscriptions:
        actual.add("events")
    if plugin.guardrails:
        actual.add("guardrails")
    if plugin.tools:
        actual.add("tools")
    if plugin.router_backends:
        actual.add("routing")
    return actual


def _undeclared_contributions(plugin: LoadedPlugin) -> set[str]:
    return _actual_contributions(plugin) - set(plugin.manifest.contributes)


def _version_tuple(text: str) -> tuple[int, ...]:
    """The leading numeric components of a version string, for a soft comparison."""
    numbers: list[int] = []
    for part in text.split("."):
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers)


def _resolve_register(manifest: PluginManifest) -> Callable[[PluginContext], None]:
    module = importlib.import_module(manifest.package)
    register = getattr(module, manifest.entrypoint, None)
    if register is None:
        msg = f"package {manifest.package!r} has no attribute {manifest.entrypoint!r}"
        raise PluginError(msg)
    if not callable(register):
        msg = f"{manifest.package}.{manifest.entrypoint} is not callable"
        raise PluginError(msg)
    if inspect.iscoroutinefunction(register):
        msg = f"{manifest.package}.{manifest.entrypoint} is async; register must be a plain def"
        raise PluginError(msg)
    return register  # type: ignore[no-any-return]


def build_plugin_config(manifest: PluginManifest, raw: dict[str, Any]) -> dict[str, Any]:
    """The manifest's setting defaults under the operator's block, type-checked against the manifest.

    Raises:
        PluginError: When a declared setting is present with a value of the wrong type.

    """
    config: dict[str, Any] = {**manifest.setting_defaults, **raw}
    for key, spec in manifest.settings.items():
        value = config.get(key)
        if value is not None and not setting_value_matches(spec.type, value):
            msg = f"setting {key!r} must be of type {spec.type}, got {type(value).__name__}"
            raise PluginError(msg)
    declared = set(manifest.all_config_keys)
    unknown = sorted(key for key in raw if declared and key not in declared)
    if unknown:
        logger.warning("Plugin %s: config keys %s are not declared in its manifest", manifest.name, ", ".join(unknown))
    return config


def runtime_mode(config: "GatewayConfig") -> RuntimeMode:
    if config.is_hybrid_mode:
        return "hybrid"
    if config.is_hosted_mode:
        return "hosted"
    return "standalone"


def _load_one(
    discovered: DiscoveredPlugin,
    settings: dict[str, Any],
    container: "Container | None",
    mode: RuntimeMode,
) -> LoadedPlugin:
    manifest = discovered.manifest
    plugin = LoadedPlugin(
        manifest=manifest,
        source=discovered.source,
        package_dir=discovered.package_dir,
        install_dir=discovered.install_dir,
        status="loaded",
    )
    if manifest.plugin_api > PLUGIN_API_VERSION:
        plugin.status = "failed"
        plugin.error = f"needs plugin API {manifest.plugin_api}; this gateway provides {PLUGIN_API_VERSION}"
        logger.error("Plugin %s: %s", manifest.name, plugin.error)
        return plugin
    if mode not in manifest.modes:
        plugin.status = "disabled"
        plugin.error = f"not for {mode} mode (supports {', '.join(manifest.modes)})"
        logger.info("Plugin %s: %s", manifest.name, plugin.error)
        return plugin
    if manifest.min_otari_version and _version_tuple(__version__) < _version_tuple(manifest.min_otari_version):
        logger.warning(
            "Plugin %s wants otari >= %s and this is %s; loading it anyway",
            manifest.name,
            manifest.min_otari_version,
            __version__,
        )
    if discovered.install_dir is not None:
        # The directory holding the package, so the package name imports.
        entry = str(discovered.package_dir.parent)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    try:
        plugin.config = build_plugin_config(manifest, settings)
        register = _resolve_register(manifest)
        outcome = register(PluginContext(plugin, plugin.config, container))
        if inspect.isawaitable(outcome):
            outcome.close()
            msg = f"{manifest.package}.{manifest.entrypoint} returned an awaitable; register must run when called"
            raise PluginError(msg)
    except Exception as error:  # noqa: BLE001 one plugin's failure must not stop the load
        logger.error("Plugin %s failed to load: %s\n%s", manifest.name, error, traceback.format_exc())
        plugin.status = "failed"
        plugin.error = f"{type(error).__name__}: {error}"
        plugin.clear_contributions()
        return plugin
    undeclared = _undeclared_contributions(plugin)
    if undeclared:
        # The declaration is what an operator read before installing; a plugin
        # that does more than it said is refused rather than trusted.
        plugin.status = "failed"
        plugin.error = (
            f"registered {', '.join(sorted(undeclared))} without declaring it in the manifest's "
            f"contributes list ({', '.join(manifest.contributes) or 'empty'})"
        )
        logger.error("Plugin %s: %s", manifest.name, plugin.error)
        plugin.clear_contributions()
        return plugin
    for declared in manifest.contributes:
        # "ui" is the manifest's own page, checked at parse time; the rest is what register did.
        if declared not in _actual_contributions(plugin) and (declared != "ui" or manifest.ui is None):
            logger.warning("Plugin %s declares %r but registered nothing of the kind", manifest.name, declared)
    for page in manifest.pages:
        page_dir = (discovered.package_dir / page.path).resolve()
        if page_dir.is_dir() and (page_dir / "index.html").is_file():
            plugin.pages.append(PageContribution(manifest=page, directory=page_dir))
        else:
            logger.warning(
                "Plugin %s declares page %s at %s but there is no index.html there", manifest.name, page.id, page_dir
            )
    return plugin


def plugins_directory(config: "GatewayConfig") -> Path:
    return Path(config.plugins.directory).expanduser().resolve()


def load_plugins(config: "GatewayConfig", container: "Container | None" = None) -> PluginRegistry:
    """Discover and load every plugin for this process.

    Called once per app in ``create_app`` and once per CLI invocation. Never
    raises for a plugin's own failure; a plugin that cannot load is listed as
    failed.
    """
    plugins_config: PluginsConfig = config.plugins
    directory = plugins_directory(config)
    if not plugins_config.enabled:
        registry = PluginRegistry(directory, [], [])
        logger.info("Plugins: disabled by configuration")
        return registry
    discovered, problems = discover_plugins(directory)
    for problem in problems:
        logger.warning("Plugin at %s (%s) could not be described: %s", problem.location, problem.source, problem.error)
    disabled = set(plugins_config.disabled)
    mode = runtime_mode(config)
    loaded: list[LoadedPlugin] = []
    for candidate in discovered:
        if candidate.manifest.name in disabled:
            loaded.append(
                LoadedPlugin(
                    manifest=candidate.manifest,
                    source=candidate.source,
                    package_dir=candidate.package_dir,
                    install_dir=candidate.install_dir,
                    status="disabled",
                )
            )
            continue
        loaded.append(_load_one(candidate, plugins_config.plugin_settings(candidate.manifest.name), container, mode))
    registry = PluginRegistry(
        directory,
        loaded,
        problems,
        event_timeout_ms=plugins_config.event_timeout_ms,
        hook_timeout_ms=plugins_config.hook_timeout_ms,
        health_timeout_ms=plugins_config.health_timeout_ms,
    )
    set_plugin_router_backends(registry.router_backends())
    logger.info("Plugins: %s", registry.summary)
    return registry
