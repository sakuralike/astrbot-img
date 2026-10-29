from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import aiohttp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import At, Plain, Reply
from astrbot.api.star import Context, Star
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .providers import (
    ImageProviderError,
    ProviderSettings,
    fetch_model_ids,
    generate_image,
    manual_provider_settings,
    settings_from_astrbot_provider,
    settings_from_manual,
)

PLUGIN_ID = "astrbot_img"
DEFAULT_TIMEOUT = 120
DEFAULT_COOLDOWN = 60
DEFAULT_CONCURRENCY = 2
DEFAULT_MAX_PROMPT_CHARS = 1000
DEFAULT_GENERATE_COMMAND = "生图"
DEFAULT_MODELS_COMMAND = "获取生图模型"
MAX_IMAGE_BYTES = 25 * 1024 * 1024


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return min(maximum, max(minimum, value))


def _command_name(value: Any, default: str) -> str:
    if not isinstance(value, str):
        return default
    command = value.strip().lstrip("/").strip()
    if not command or any(char.isspace() for char in command):
        return default
    return command


def extract_prompt(message: str, command: str = "生图") -> str:
    text = str(message or "").strip()
    command = command.strip().lstrip("/").strip() if isinstance(command, str) else ""
    if not command or any(char.isspace() for char in command):
        command = "生图"
    prefixes = {"/生图", f"/{command}"}
    for prefix in sorted(prefixes, key=len, reverse=True):
        if text == prefix:
            return ""
        if text.startswith(f"{prefix} "):
            return text[len(prefix) :].strip()
    return text


def _display_error(error: ImageProviderError) -> str:
    return {
        "not_configured": "生图功能尚未配置，请管理员选择 Provider 或配置手动供应商。",
        "invalid_config": "生图配置无效，请管理员检查接口地址、密钥和模型 ID。",
        "unsupported_provider": "所选 Provider 不支持明确的 OpenAI Images/Responses 生图协议。",
        "unauthorized": "生图供应商鉴权失败，请管理员检查密钥。",
        "rate_limited": "生图请求过于频繁，请稍后再试。",
        "timeout": "生图请求超时，请稍后再试。",
        "network_error": "生图供应商暂时无法连接，请稍后再试。",
        "provider_error": "生图供应商暂时不可用，请稍后再试。",
        "invalid_response": "生图供应商返回了无法识别的数据。",
        "unsupported_response": "生图供应商返回格式不受支持。",
        "async_unsupported": "该供应商仅返回异步任务，本插件暂不支持任务轮询。",
    }.get(error.code, "生图失败，请稍后再试。")


class ImageGenerationPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._cooldowns: dict[str, float] = {}
        self._semaphore = asyncio.Semaphore(
            _bounded_int(
                config.get("max_concurrency", DEFAULT_CONCURRENCY),
                DEFAULT_CONCURRENCY,
                1,
                8,
            )
        )
        self._session: aiohttp.ClientSession | None = None
        self._temp_dir = (
            Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_ID / "images"
        )
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._terminated = False
        self.generate_command = _command_name(
            config.get("generate_command", DEFAULT_GENERATE_COMMAND),
            DEFAULT_GENERATE_COMMAND,
        )
        self.models_command = _command_name(
            config.get("models_command", DEFAULT_MODELS_COMMAND),
            DEFAULT_MODELS_COMMAND,
        )
        self._model_cache: dict[str, list[str]] = {}
        self._configure_commands()

    def _configure_commands(self) -> None:
        try:
            from astrbot.core.star.star_handler import star_handlers_registry
        except (ImportError, AttributeError):
            return

        command_names = {
            "generate": (self.generate_command, DEFAULT_GENERATE_COMMAND),
            "refresh_models": (self.models_command, DEFAULT_MODELS_COMMAND),
        }
        for handler_name, (command_name, default_name) in command_names.items():
            handler = star_handlers_registry.get_handler_by_full_name(
                f"{__name__}_{handler_name}"
            )
            if handler is None:
                continue
            for filter_ref in handler.event_filters:
                if not hasattr(filter_ref, "command_name"):
                    continue
                base_aliases = getattr(filter_ref, "_astrbot_img_base_aliases", None)
                if base_aliases is None:
                    base_aliases = set(getattr(filter_ref, "alias", set()))
                    filter_ref._astrbot_img_base_aliases = set(base_aliases)
                aliases = set(base_aliases)
                if command_name != default_name:
                    aliases.add(default_name)
                filter_ref.command_name = command_name
                filter_ref.alias = aliases
                filter_ref._cmpl_cmd_names = None
                break

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _scope_allowed(self, event: AstrMessageEvent) -> bool:
        group_id = str(event.get_group_id() or "").strip()
        if not group_id:
            return False
        allowed = self.config.get("allowed_groups", [])
        if not isinstance(allowed, list) or not allowed:
            return True
        platform_id = str(event.get_platform_id() or "").strip()
        bot_id = str(event.get_self_id() or "").strip()
        candidates = {
            group_id,
            f"{platform_id}|{group_id}",
            f"{platform_id}|{bot_id}|{group_id}",
        }
        return any(
            isinstance(item, str) and item.strip() in candidates for item in allowed
        )

    def _cooldown_key(self, event: AstrMessageEvent) -> str:
        return "|".join(
            (
                str(event.get_platform_id() or ""),
                str(event.get_group_id() or ""),
                str(event.get_sender_id() or ""),
            )
        )

    def _cooldown_remaining(self, event: AstrMessageEvent, now: float) -> int:
        key = self._cooldown_key(event)
        deadline = self._cooldowns.get(key, 0.0)
        if deadline <= now:
            self._cooldowns.pop(key, None)
            return 0
        return max(1, int(deadline - now + 0.999))

    def _mark_cooldown(self, event: AstrMessageEvent, now: float) -> None:
        cooldown = _bounded_int(
            self.config.get("cooldown_seconds", DEFAULT_COOLDOWN),
            DEFAULT_COOLDOWN,
            0,
            3600,
        )
        if cooldown:
            self._cooldowns[self._cooldown_key(event)] = now + cooldown

    def _clear_cooldown(self, event: AstrMessageEvent) -> None:
        self._cooldowns.pop(self._cooldown_key(event), None)

    def _framework_settings(self) -> ProviderSettings:
        provider_id = str(self.config.get("astrbot_provider_id", "") or "").strip()
        provider = (
            self.context.get_provider_by_id(provider_id) if provider_id else None
        )
        if provider is None and not provider_id:
            providers = self.context.get_all_providers()
            provider = providers[0] if providers else None
            if provider is not None:
                try:
                    provider_id = str(provider.meta().id or "")
                except (AttributeError, TypeError, ValueError):
                    provider_id = ""
        return settings_from_astrbot_provider(
            provider_id,
            provider,
            str(self.config.get("astrbot_model_id", "") or ""),
            _bounded_int(
                self.config.get("request_timeout", DEFAULT_TIMEOUT),
                DEFAULT_TIMEOUT,
                5,
                600,
            ),
        )

    def _provider_settings(self) -> ProviderSettings:
        mode = str(self.config.get("provider_mode", "astrbot") or "astrbot")
        timeout = _bounded_int(
            self.config.get("request_timeout", DEFAULT_TIMEOUT),
            DEFAULT_TIMEOUT,
            5,
            600,
        )
        if mode == "manual":
            return manual_provider_settings(
                self.config.get("manual_providers", []),
                str(self.config.get("manual_provider_id", "") or ""),
                timeout,
                self._model_cache,
            )
        if mode != "astrbot":
            raise ImageProviderError("invalid_config", "生图供应商模式无效。")
        return self._framework_settings()

    def _manual_provider_items(self) -> list[Mapping[str, Any]]:
        providers = self.config.get("manual_providers", [])
        if not isinstance(providers, list):
            return []
        return [
            item
            for item in providers
            if isinstance(item, Mapping)
            and item.get("enabled", True) is not False
            and str(item.get("id", "") or "").strip()
        ]

    def _manual_model_endpoints(self, item: Mapping[str, Any]) -> list[str]:
        configured = str(item.get("models_endpoint", "") or "").strip()
        if configured:
            return [configured]
        base_url = str(item.get("base_url", "") or "").rstrip("/")
        if not base_url:
            return []
        endpoints = [f"{base_url}/models"]
        if not base_url.lower().endswith("/v1"):
            endpoints.append(f"{base_url}/v1/models")
        return endpoints

    async def _fetch_manual_models(self, item: Mapping[str, Any]) -> list[str]:
        timeout = _bounded_int(
            self.config.get("request_timeout", DEFAULT_TIMEOUT),
            DEFAULT_TIMEOUT,
            5,
            600,
        )
        settings = settings_from_manual(item, timeout, require_model=False)
        session = await self._http()
        last_error: ImageProviderError | None = None
        for endpoint in self._manual_model_endpoints(item):
            try:
                models = await fetch_model_ids(session, settings, endpoint)
                if models:
                    return models
                last_error = ImageProviderError(
                    "invalid_response", "供应商没有返回可用模型。"
                )
            except ImageProviderError as error:
                last_error = error
        raise last_error or ImageProviderError(
            "invalid_config", "未配置模型列表接口。"
        )

    def _cache_manual_models(self, provider_id: str, models: list[str]) -> None:
        self._model_cache[provider_id] = list(dict.fromkeys(models))

    async def _persist_manual_models(
        self, provider_id: str, models: list[str]
    ) -> None:
        providers = self.config.get("manual_providers", [])
        if isinstance(providers, list):
            for item in providers:
                if (
                    isinstance(item, dict)
                    and str(item.get("id", "") or "").strip() == provider_id
                ):
                    item["models"] = list(dict.fromkeys(models))
                    break
        save_config = getattr(self.config, "save_config", None)
        if not callable(save_config):
            return
        result = save_config()
        if inspect.isawaitable(result):
            await result

    async def _download_image(self, url: str, settings: ProviderSettings) -> bytes:
        parsed = urlparse(url)
        if parsed.scheme not in {"https", "http"}:
            raise ImageProviderError("unsupported_response", "供应商图片地址无效。")
        if parsed.scheme != "https" and not settings.allow_insecure_http:
            raise ImageProviderError("unsupported_response", "供应商返回了不安全的图片地址。")
        session = await self._http()
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=settings.timeout),
            ) as response:
                if response.status in {401, 403}:
                    raise ImageProviderError("unauthorized", "供应商图片下载鉴权失败。")
                if response.status == 429:
                    raise ImageProviderError("rate_limited", "供应商图片下载请求频率受限。")
                if response.status >= 400:
                    raise ImageProviderError("provider_error", "供应商图片下载失败。")
                data = await response.content.read(MAX_IMAGE_BYTES + 1)
        except ImageProviderError:
            raise
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
            raise ImageProviderError("timeout", "供应商图片下载超时。")
        except (aiohttp.ClientError, OSError):
            raise ImageProviderError("network_error", "供应商图片下载失败。")
        if len(data) > MAX_IMAGE_BYTES:
            raise ImageProviderError("provider_error", "生成图片超过大小限制。")
        if not data:
            raise ImageProviderError("unsupported_response", "供应商返回了空图片。")
        return data

    async def _send_text(self, event: AstrMessageEvent, text: str) -> None:
        await event.send(MessageChain().message(text).use_markdown(False))

    async def _send_image(self, event: AstrMessageEvent, path: Path) -> None:
        await event.send(MessageChain().file_image(str(path)).use_markdown(False))

    async def _generate(self, prompt: str, settings: ProviderSettings) -> bytes:
        session = await self._http()
        image_bytes, image_url = await generate_image(session, settings, prompt)
        if image_bytes:
            return image_bytes
        if image_url:
            return await self._download_image(image_url, settings)
        raise ImageProviderError("unsupported_response", "供应商没有返回图片。")

    @filter.command("生图")
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def generate(self, event: AstrMessageEvent) -> None:
        if (
            self._terminated
            or self.config.get("enabled", True) is not True
            or not self._scope_allowed(event)
        ):
            return
        event.should_call_llm(False)
        event.stop_event()
        prompt = extract_prompt(event.get_message_str(), self.generate_command)
        max_chars = _bounded_int(
            self.config.get("max_prompt_chars", DEFAULT_MAX_PROMPT_CHARS),
            DEFAULT_MAX_PROMPT_CHARS,
            1,
            4000,
        )
        if not prompt:
            await self._send_text(
                event, f"用法：/{self.generate_command} 你的提示词"
            )
            return
        if len(prompt) > max_chars:
            await self._send_text(event, f"提示词过长，请控制在 {max_chars} 字以内。")
            return
        now = time.monotonic()
        remaining = self._cooldown_remaining(event, now)
        if remaining:
            await self._send_text(event, f"你还在生图冷却中，请 {remaining} 秒后再试。")
            return
        if self._semaphore.locked():
            await self._send_text(event, "当前已有较多生图任务，请稍后再试。")
            return
        try:
            settings = self._provider_settings()
        except ImageProviderError as error:
            await self._send_text(event, _display_error(error))
            return
        self._mark_cooldown(event, now)
        request_succeeded = False
        async with self._semaphore:
            try:
                image_bytes = await self._generate(prompt, settings)
                path = self._temp_dir / f"{uuid4().hex}.png"
                path.write_bytes(image_bytes)
                try:
                    await self._send_image(event, path)
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        path.unlink()
                request_succeeded = True
            except ImageProviderError as error:
                logger.warning("AstrBot 生图请求失败：%s", error.code)
                await self._send_text(event, _display_error(error))
            except (OSError, ValueError, TypeError):
                logger.exception("AstrBot 生图消息发送失败")
                await self._send_text(event, "图片生成成功，但发送图片失败。")
            finally:
                if not request_succeeded:
                    self._clear_cooldown(event)

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        try:
            return event.is_admin() is True
        except (AttributeError, TypeError, ValueError):
            return False

    @filter.command(DEFAULT_MODELS_COMMAND)
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def refresh_models(self, event: AstrMessageEvent) -> None:
        if (
            self._terminated
            or self.config.get("enabled", True) is not True
            or not self._scope_allowed(event)
        ):
            return
        event.should_call_llm(False)
        event.stop_event()
        if not self._is_admin(event):
            await self._send_text(event, "只有管理员可以获取生图模型。")
            return
        if str(self.config.get("provider_mode", "astrbot") or "") != "manual":
            await self._send_text(event, "请先将生图供应商来源设置为插件手动供应商。")
            return
        results: list[str] = []
        for item in self._manual_provider_items():
            provider_id = str(item.get("id", "") or "").strip()
            try:
                models = await self._fetch_manual_models(item)
                self._cache_manual_models(provider_id, models)
                await self._persist_manual_models(provider_id, models)
                results.append(f"{provider_id}：{', '.join(models)}")
            except ImageProviderError as error:
                results.append(f"{provider_id}：获取失败（{_display_error(error)}）")
        if results:
            await self._send_text(event, "生图模型列表：\n" + "\n".join(results))
        else:
            await self._send_text(event, "未找到启用的手动生图供应商。")

    @filter.on_decorating_result()
    async def qq_official_plain_reply(self, event: AstrMessageEvent) -> None:
        if self._terminated or event.get_platform_name() != "qq_official":
            return
        result = event.get_result()
        if result is None or not result.chain:
            return
        if all(isinstance(component, (Plain, At, Reply)) for component in result.chain):
            result.use_markdown(False)

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self) -> None:
        if (
            self.config.get("enabled", True) is not True
            or str(self.config.get("provider_mode", "astrbot") or "astrbot")
            != "manual"
        ):
            return
        for item in self._manual_provider_items():
            try:
                models = await self._fetch_manual_models(item)
                provider_id = str(item.get("id", "") or "").strip()
                self._cache_manual_models(provider_id, models)
                logger.info(
                    "AstrBot 生图供应商 %s 可用模型数：%d",
                    provider_id,
                    len(models),
                )
            except (ImageProviderError, aiohttp.ClientError, OSError, ValueError):
                logger.debug(
                    "AstrBot 生图供应商模型列表拉取失败：%s",
                    str(item.get("id", "")),
                )

    async def terminate(self) -> None:
        self._terminated = True
        self._cooldowns.clear()
        with contextlib.suppress(Exception):
            if self._session is not None and not self._session.closed:
                await self._session.close()
        self._session = None
