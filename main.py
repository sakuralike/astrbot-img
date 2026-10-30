from __future__ import annotations

import asyncio
import contextlib
import inspect
import re
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
from astrbot.api.web import error_response, json_response, request
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .providers import (
    ImageProviderError,
    ProviderSettings,
    _valid_url,
    fetch_model_ids,
    generate_image,
    manual_provider_settings,
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
_MANUAL_PROTOCOLS = {"openai_images", "openai_responses"}


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
    prefixes = {"生图", "/生图", command, f"/{command}"}
    for prefix in sorted(prefixes, key=len, reverse=True):
        if text == prefix:
            return ""
        if text.startswith(prefix) and text[len(prefix) : len(prefix) + 1].isspace():
            return text[len(prefix) :].strip()
    return text


def _display_error(error: ImageProviderError) -> str:
    return {
        "not_configured": "生图功能尚未配置，请管理员配置插件手动供应商。",
        "invalid_config": "生图配置无效，请管理员检查接口地址、密钥和模型 ID。",
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
        context.register_web_api(
            f"/{PLUGIN_ID}/providers",
            self.web_get_providers,
            ["GET"],
            "Get image providers",
        )
        context.register_web_api(
            f"/{PLUGIN_ID}/providers/save",
            self.web_save_provider,
            ["POST"],
            "Save image provider",
        )
        context.register_web_api(
            f"/{PLUGIN_ID}/providers/models",
            self.web_fetch_provider_models,
            ["POST"],
            "Fetch image provider models",
        )
        context.register_web_api(
            f"/{PLUGIN_ID}/providers/<provider_id>",
            self.web_delete_provider,
            ["DELETE"],
            "Delete image provider",
        )
        context.register_web_api(
            f"/{PLUGIN_ID}/providers/delete",
            self.web_delete_provider_post,
            ["POST"],
            "Delete image provider",
        )
        self._configure_commands()

    async def initialize(self) -> None:
        if self._migrate_manual_provider_templates():
            await self._save_config()

    @staticmethod
    def _provider_template_key(item: Mapping[str, Any]) -> str:
        protocol = str(item.get("protocol", "") or "").strip().lower()
        if protocol:
            return protocol if protocol in _MANUAL_PROTOCOLS else ""
        template_key = str(item.get("__template_key", "") or "").strip().lower()
        return template_key if template_key in _MANUAL_PROTOCOLS else ""

    def _migrate_manual_provider_templates(self) -> bool:
        providers = self.config.get("manual_providers", [])
        if not isinstance(providers, list):
            return False
        changed = False
        for item in providers:
            if not isinstance(item, dict) or item.get("__template_key"):
                continue
            template_key = self._provider_template_key(item)
            if template_key:
                item["__template_key"] = template_key
                changed = True
        return changed

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

    def _provider_settings(self) -> ProviderSettings:
        timeout = _bounded_int(
            self.config.get("request_timeout", DEFAULT_TIMEOUT),
            DEFAULT_TIMEOUT,
            5,
            600,
        )
        return manual_provider_settings(
            self.config.get("manual_providers", []),
            str(self.config.get("manual_provider_id", "") or ""),
            timeout,
            self._model_cache,
        )

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
        base_url = str(item.get("base_url", "") or "").rstrip("/")
        if not base_url and not configured:
            return []
        endpoints = [configured] if configured else []
        if base_url.lower().endswith("/v1"):
            endpoints.extend(
                (f"{base_url}/models", f"{base_url[:-3].rstrip('/')}/models")
            )
        elif base_url:
            endpoints.extend((f"{base_url}/models", f"{base_url}/v1/models"))
        return list(dict.fromkeys(endpoints))

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

    async def _save_config(self) -> None:
        save_config = getattr(self.config, "save_config", None)
        if callable(save_config):
            result = save_config()
            if inspect.isawaitable(result):
                await result

    def _provider_by_id(self, provider_id: str) -> dict[str, Any] | None:
        providers = self.config.get("manual_providers", [])
        if not isinstance(providers, list):
            return None
        for item in providers:
            if isinstance(item, dict) and str(item.get("id", "")).strip() == provider_id:
                return item
        return None

    @staticmethod
    def _public_url(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return ""
        try:
            parsed = urlparse(value.strip())
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            return ""
        if not parsed.scheme or not hostname:
            return ""
        netloc = f"[{hostname}]" if ":" in hostname else hostname
        if port is not None:
            netloc = f"{netloc}:{port}"
        return parsed._replace(netloc=netloc, query="", fragment="").geturl()

    def _public_provider(self, item: Mapping[str, Any]) -> dict[str, Any]:
        provider_id = str(item.get("id", "") or "").strip()
        protocol = self._provider_template_key(item)
        configured_models = item.get("models", [])
        if not isinstance(configured_models, list):
            configured_models = []
        cached = self._model_cache.get(provider_id, [])
        models = list(
            dict.fromkeys(
                value.strip()
                for value in [*configured_models, *cached]
                if isinstance(value, str) and value.strip()
            )
        )
        return {
            "id": provider_id,
            "name": str(item.get("name", "") or ""),
            "protocol": protocol,
            "base_url": self._public_url(item.get("base_url", "")),
            "model": str(item.get("model", "") or ""),
            "models": [
                value.strip()
                for value in models
                if isinstance(value, str) and value.strip()
            ],
            "models_endpoint": self._public_url(item.get("models_endpoint", "")),
            "enabled": item.get("enabled", True) is not False,
            "allow_insecure_http": item.get("allow_insecure_http", False) is True,
            "has_api_key": bool(str(item.get("api_key", "") or "").strip()),
        }

    async def web_get_providers(self):
        providers = self.config.get("manual_providers", [])
        if not isinstance(providers, list):
            providers = []
        return json_response(
            {
                "providers": [
                    self._public_provider(item)
                    for item in providers
                    if isinstance(item, Mapping)
                ],
                "current_provider_id": str(
                    self.config.get("manual_provider_id", "") or ""
                ),
            }
        )

    async def web_save_provider(self):
        payload = await request.json(default={})
        if not isinstance(payload, Mapping):
            return error_response("供应商配置格式无效。", status_code=400)

        provider_id = payload.get("id")
        if not isinstance(provider_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,64}", provider_id.strip()
        ):
            return error_response("供应商 ID 格式无效。", status_code=400)
        provider_id = provider_id.strip()

        protocol = payload.get("protocol")
        if protocol not in {"openai_images", "openai_responses"}:
            return error_response("供应商协议无效。", status_code=400)

        name = payload.get("name")
        base_url = payload.get("base_url")
        model = payload.get("model", "")
        models_endpoint = payload.get("models_endpoint", "")
        api_key = payload.get("api_key", "")
        if not all(
            isinstance(value, str)
            for value in (name, base_url, model, models_endpoint, api_key)
        ):
            return error_response("供应商字段格式无效。", status_code=400)
        name = name.strip()
        base_url = base_url.strip()
        model = model.strip()
        models_endpoint = models_endpoint.strip()
        api_key = api_key.strip()
        if not name or len(name) > 128 or len(model) > 256:
            return error_response("供应商名称或模型 ID 无效。", status_code=400)
        if len(base_url) > 2048 or len(models_endpoint) > 2048 or len(api_key) > 8192:
            return error_response("供应商字段过长。", status_code=400)
        for value in (base_url, models_endpoint):
            try:
                parsed = urlparse(value) if value else None
            except ValueError:
                return error_response("接口地址无效。", status_code=400)
            if parsed and (
                parsed.username
                or getattr(parsed, "pass" + "word")
                or parsed.query
                or parsed.fragment
            ):
                return error_response(
                    "接口地址不能包含认证信息或查询参数。", status_code=400
                )
        allow_insecure_http = payload.get("allow_insecure_http", False)
        enabled = payload.get("enabled", True)
        set_current = payload.get("set_current", False)
        if not all(
            isinstance(value, bool)
            for value in (allow_insecure_http, enabled, set_current)
        ):
            return error_response("供应商开关必须为布尔值。", status_code=400)

        providers = self.config.get("manual_providers", [])
        if not isinstance(providers, list):
            providers = []
        original_id_value = payload.get("original_id", "")
        if not isinstance(original_id_value, str):
            return error_response("原供应商 ID 格式无效。", status_code=400)
        original_id = original_id_value.strip()
        if original_id and not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", original_id):
            return error_response("原供应商 ID 格式无效。", status_code=400)
        old_item = next(
            (
                item
                for item in providers
                if isinstance(item, Mapping)
                and str(item.get("id", "")).strip() == original_id
            ),
            None,
        )
        if any(
            isinstance(item, Mapping)
            and str(item.get("id", "")).strip() == provider_id
            and str(item.get("id", "")).strip() != original_id
            for item in providers
        ):
            return error_response("供应商 ID 已存在。", status_code=409)
        if not api_key and isinstance(old_item, Mapping):
            api_key = str(old_item.get("api_key", "") or "").strip()
        if not api_key:
            return error_response("新增供应商必须填写 API Key。", status_code=400)

        item: dict[str, Any] = {
            "id": provider_id,
            "name": name,
            "protocol": protocol,
            "__template_key": protocol,
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "models": [],
            "models_endpoint": models_endpoint,
            "enabled": enabled,
            "allow_insecure_http": allow_insecure_http,
        }
        if isinstance(old_item, Mapping):
            old_models = old_item.get("models", [])
            if isinstance(old_models, list):
                item["models"] = [
                    value.strip()
                    for value in old_models
                    if isinstance(value, str) and value.strip()
                ]
        submitted_models = payload.get("models")
        if submitted_models is not None:
            if not isinstance(submitted_models, list) or len(submitted_models) > 500:
                return error_response("模型列表格式无效。", status_code=400)
            if any(
                not isinstance(value, str) or len(value.strip()) > 256
                for value in submitted_models
            ):
                return error_response("模型列表格式无效。", status_code=400)
            submitted_models = list(
                dict.fromkeys(value.strip() for value in submitted_models if value.strip())
            )
            if submitted_models or not item["models"]:
                item["models"] = submitted_models

        try:
            validation_item = dict(item)
            validation_item["enabled"] = True
            settings_from_manual(
                validation_item, DEFAULT_TIMEOUT, require_model=False
            )
            if models_endpoint:
                _valid_url(models_endpoint, allow_insecure_http)
        except ImageProviderError as error:
            return error_response(_display_error(error), status_code=400)

        updated = False
        new_providers: list[dict[str, Any]] = []
        for current in providers:
            if not isinstance(current, Mapping):
                continue
            if str(current.get("id", "")).strip() == original_id:
                new_providers.append(item)
                updated = True
            else:
                new_providers.append(dict(current))
        if not updated:
            new_providers.append(item)
        self.config["manual_providers"] = new_providers
        if item["models"]:
            self._cache_manual_models(provider_id, item["models"])
        else:
            self._model_cache.pop(provider_id, None)
        if (
            original_id != provider_id
            and str(self.config.get("manual_provider_id", "") or "") == original_id
        ):
            self.config["manual_provider_id"] = provider_id
        if set_current:
            self.config["manual_provider_id"] = provider_id
        await self._save_config()
        return json_response(
            {
                "provider": self._public_provider(item),
                "current_provider_id": str(
                    self.config.get("manual_provider_id", "") or ""
                ),
            }
        )

    async def web_fetch_provider_models(self):
        payload = await request.json(default={})
        provider_id = (
            payload.get("id", "") if isinstance(payload, Mapping) else ""
        )
        if not isinstance(provider_id, str) or not provider_id.strip():
            return error_response("请先选择供应商。", status_code=400)
        item = self._provider_by_id(provider_id.strip())
        if item is None:
            return error_response("未找到指定供应商。", status_code=404)
        request_item = dict(item)
        request_item["enabled"] = True
        try:
            models = await self._fetch_manual_models(request_item)
        except ImageProviderError as error:
            logger.warning(
                "AstrBot 生图供应商模型获取失败：%s（%s）",
                provider_id.strip(),
                error.code,
            )
            return error_response(_display_error(error), status_code=502)
        models = list(dict.fromkeys(models))
        item["models"] = models
        if not str(item.get("model", "") or "").strip() and models:
            item["model"] = models[0]
        self._cache_manual_models(provider_id.strip(), models)
        await self._save_config()
        return json_response(
            {
                "models": models,
                "provider": self._public_provider(item),
            }
        )

    async def web_delete_provider(self, provider_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", provider_id or ""):
            return error_response("供应商 ID 格式无效。", status_code=400)
        providers = self.config.get("manual_providers", [])
        if not isinstance(providers, list):
            providers = []
        remaining = [
            dict(item)
            for item in providers
            if isinstance(item, Mapping)
            and str(item.get("id", "")).strip() != provider_id
        ]
        if len(remaining) == len(
            [item for item in providers if isinstance(item, Mapping)]
        ):
            return error_response("未找到指定供应商。", status_code=404)
        self.config["manual_providers"] = remaining
        if str(self.config.get("manual_provider_id", "") or "") == provider_id:
            next_provider = next(
                (
                    item
                    for item in remaining
                    if item.get("enabled", True) is not False
                ),
                None,
            )
            self.config["manual_provider_id"] = (
                str(next_provider.get("id", "")) if next_provider else ""
            )
        self._model_cache.pop(provider_id, None)
        await self._save_config()
        return json_response(
            {
                "deleted": provider_id,
                "current_provider_id": str(
                    self.config.get("manual_provider_id", "") or ""
                ),
            }
        )

    async def web_delete_provider_post(self):
        payload = await request.json(default={})
        provider_id = payload.get("id", "") if isinstance(payload, Mapping) else ""
        if not isinstance(provider_id, str):
            return error_response("供应商 ID 格式无效。", status_code=400)
        return await self.web_delete_provider(provider_id)

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

    async def _upload_to_moxtu(
        self, event: AstrMessageEvent, image_bytes: bytes
    ) -> None:
        if self.config.get("auto_upload_to_moxtu", False) is not True:
            return
        unknown = "图床保存结果暂无法确认，未自动重试。"
        message = "图片已生成，但未自动上传：Moxtu 插件未安装或未启用。"
        upload_started = False
        try:
            metadata = self.context.get_registered_star("astrbot_plugin_moxtu_gallery")
            if metadata is not None and metadata.activated and metadata.star_cls:
                upload = getattr(metadata.star_cls, "upload_generated_image", None)
                if not callable(upload):
                    message = "图片已生成，但未自动上传：请更新 Moxtu 插件以支持生图联动。"
                else:
                    upload_started = True
                    result = await upload(event, image_bytes)
                    status = result.get("status") if isinstance(result, dict) else None
                    if status == "success":
                        message = "图片已自动上传到 Moxtu 专属相册。"
                    elif status == "existing":
                        message = "这张图片已保存到 Moxtu 专属相册，无需重复上传。"
                    elif status in {"failed", "skipped"}:
                        detail = result.get("message")
                        message = "图片已生成，但未自动上传：" + (
                            detail.strip()
                            if isinstance(detail, str) and detail.strip()
                            else "请管理员检查 Moxtu 插件配置。"
                        )
                    else:
                        message = unknown
        except Exception as error:
            logger.warning("Moxtu 自动上传异常：%s", type(error).__name__)
            if upload_started:
                message = unknown
        try:
            await self._send_text(event, message)
        except Exception as error:
            logger.warning("Moxtu 上传结果回复失败：%s", type(error).__name__)

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
        image_bytes: bytes | None = None
        async with self._semaphore:
            try:
                image_bytes = await self._generate(prompt, settings)
                path = self._temp_dir / f"{uuid4().hex}.png"
                try:
                    path.write_bytes(image_bytes)
                    await self._send_image(event, path)
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        path.unlink()
                request_succeeded = True
            except ImageProviderError as error:
                logger.warning("AstrBot 生图请求失败：%s", error.code)
                await self._send_text(event, _display_error(error))
            except Exception as error:
                if image_bytes is None:
                    raise
                logger.warning("AstrBot 生图消息发送失败：%s", type(error).__name__)
                with contextlib.suppress(Exception):
                    await self._send_text(event, "图片生成成功，但发送图片失败。")
            finally:
                if not request_succeeded:
                    self._clear_cooldown(event)
            if image_bytes is not None:
                await self._upload_to_moxtu(event, image_bytes)

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
        if self.config.get("enabled", True) is not True:
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
