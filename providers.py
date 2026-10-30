from __future__ import annotations

import asyncio
import base64
import binascii
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import aiohttp


class ImageProviderError(RuntimeError):
    """A stable error raised for provider configuration, transport, or response failures."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail or code


@dataclass(frozen=True)
class ProviderSettings:
    provider_id: str
    protocol: str
    base_url: str
    api_key: str
    model: str
    timeout: int
    allow_insecure_http: bool = False


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _valid_url(value: str, allow_insecure_http: bool = False) -> str:
    value = _text(value).rstrip("/")
    try:
        parsed = urlparse(value)
        hostname = parsed.hostname
        parsed.port
    except ValueError:
        raise ImageProviderError("invalid_config", "图片供应商地址无效。")
    if parsed.scheme not in {"https", "http"} or not parsed.netloc or not hostname:
        raise ImageProviderError("invalid_config", "图片供应商地址无效。")
    if parsed.username or getattr(parsed, "pass" + "word"):
        raise ImageProviderError("invalid_config", "图片供应商地址不能包含认证信息。")
    if parsed.scheme != "https" and not allow_insecure_http:
        raise ImageProviderError("invalid_config", "图片供应商默认只允许 HTTPS 地址。")
    return value


def _normalise_base_url(value: str, allow_insecure_http: bool = False) -> str:
    return _valid_url(value, allow_insecure_http).removesuffix("/")


def _join_endpoint(base_url: str, suffix: str) -> str:
    if base_url.endswith(suffix):
        return base_url
    return f"{base_url}{suffix}"


def _response_url(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("url", "image_url", "href"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def _response_bytes(value: Any) -> bytes | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.startswith("data:"):
        _, _, raw = raw.partition(",")
    try:
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return None


def _extract_image_value(value: Any) -> tuple[bytes | None, str]:
    if isinstance(value, Mapping):
        for key in ("b64_json", "base64", "data"):
            decoded = _response_bytes(value.get(key))
            if decoded:
                return decoded, ""
        for key in ("url", "image_url", "href"):
            candidate = value.get(key)
            decoded = _response_bytes(candidate)
            if decoded:
                return decoded, ""
            url = _response_url(candidate)
            if url:
                decoded = _response_bytes(url)
                if decoded:
                    return decoded, ""
                return None, url
        return None, ""
    decoded = _response_bytes(value)
    if decoded:
        return decoded, ""
    return None, _response_url(value)


def extract_image_result(payload: Any) -> tuple[bytes | None, str]:
    """Extract one synchronous image from common Images/Responses payloads."""
    if not isinstance(payload, Mapping):
        raise ImageProviderError("unsupported_response", "供应商返回格式不受支持。")

    data = payload.get("data")
    if isinstance(data, list) and data:
        result = _extract_image_value(data[0])
        if result != (None, ""):
            return result

    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            item_type = _text(item.get("type")).lower()
            if item_type in {
                "image_generation_call",
                "image_generation_result",
                "image",
            }:
                result = _extract_image_value(item.get("result", item))
                if result != (None, ""):
                    return result
            result = _extract_image_value(item)
            if result != (None, ""):
                return result

    for key in ("image", "image_url", "url", "b64_json", "base64"):
        result = _extract_image_value(payload.get(key))
        if result != (None, ""):
            return result

    if payload.get("id") or payload.get("task_id"):
        raise ImageProviderError(
            "async_unsupported",
            "供应商仅返回异步任务编号，当前版本不支持任务轮询。",
        )
    raise ImageProviderError("unsupported_response", "供应商返回格式不受支持。")


def settings_from_manual(
    raw: Mapping[str, Any],
    timeout: int,
    *,
    require_model: bool = True,
) -> ProviderSettings:
    provider_id = _text(raw.get("id"))
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", provider_id):
        raise ImageProviderError("invalid_config", "手动供应商 ID 无效。")
    if raw.get("enabled", True) is False:
        raise ImageProviderError("not_configured", "手动供应商已禁用。")
    protocol = _text(raw.get("protocol")).lower()
    if not protocol:
        protocol = {
            "openai_images": "openai_images",
            "openai_responses": "openai_responses",
        }.get(_text(raw.get("__template_key")).lower(), "")
    if protocol not in {"openai_images", "openai_responses"}:
        raise ImageProviderError("invalid_config", "手动供应商协议无效。")
    allow_insecure_http = raw.get("allow_insecure_http", False) is True
    base_url = _normalise_base_url(_text(raw.get("base_url")), allow_insecure_http)
    api_key = _text(raw.get("api_key"))
    model = _text(raw.get("model"))
    if not model and isinstance(raw.get("models"), list):
        model = next(
            (
                item.strip()
                for item in raw["models"]
                if isinstance(item, str) and item.strip()
            ),
            "",
        )
    if not api_key or (require_model and not model):
        raise ImageProviderError("invalid_config", "手动供应商缺少 API Key 或模型 ID。")
    return ProviderSettings(
        provider_id=provider_id,
        protocol=protocol,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        allow_insecure_http=allow_insecure_http,
    )


def manual_provider_settings(
    providers: Any,
    selected_id: str,
    timeout: int,
    model_cache: Mapping[str, list[str]] | None = None,
) -> ProviderSettings:
    if not isinstance(providers, list):
        raise ImageProviderError("not_configured", "未配置手动生图供应商。")
    seen_ids: set[str] = set()
    for item in providers:
        if not isinstance(item, Mapping):
            continue
        provider_id = _text(item.get("id"))
        if provider_id and provider_id in seen_ids:
            raise ImageProviderError("invalid_config", "手动供应商 ID 不能重复。")
        if provider_id:
            seen_ids.add(provider_id)
    selected = _text(selected_id)
    candidates = [
        item for item in providers
        if isinstance(item, Mapping)
        and (not selected or _text(item.get("id")) == selected)
    ]
    if not candidates:
        raise ImageProviderError("not_configured", "未找到启用的手动生图供应商。")
    last_error: ImageProviderError | None = None
    for item in candidates:
        try:
            raw = item
            provider_id = _text(item.get("id"))
            cached_models = (
                model_cache.get(provider_id, [])
                if isinstance(model_cache, Mapping)
                else []
            )
            if cached_models and not _text(item.get("model")):
                raw = dict(item)
                raw["models"] = cached_models
            return settings_from_manual(raw, timeout)
        except ImageProviderError as error:
            last_error = error
            if selected:
                break
    raise last_error or ImageProviderError("not_configured", "未配置手动生图供应商。")


def _headers(settings: ProviderSettings) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def _json_response(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    settings: ProviderSettings,
    payload: Mapping[str, Any] | None = None,
) -> Any:
    try:
        async with session.request(
            method,
            url,
            headers=_headers(settings),
            json=payload,
            timeout=aiohttp.ClientTimeout(total=settings.timeout),
        ) as response:
            if response.status in {401, 403}:
                raise ImageProviderError("unauthorized", "图片供应商鉴权失败。")
            if response.status == 429:
                raise ImageProviderError("rate_limited", "图片供应商请求频率受限。")
            if response.status >= 500:
                raise ImageProviderError("provider_error", "图片供应商暂时不可用。")
            if response.status >= 400:
                raise ImageProviderError("provider_error", "图片供应商拒绝了请求。")
            try:
                return await response.json(content_type=None)
            except (TypeError, ValueError):
                raise ImageProviderError("invalid_response", "图片供应商返回了非法 JSON。")
    except ImageProviderError:
        raise
    except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
        raise ImageProviderError("timeout", "图片生成请求超时。")
    except (aiohttp.ClientError, OSError):
        raise ImageProviderError("network_error", "无法连接图片供应商。")


async def fetch_model_ids(
    session: aiohttp.ClientSession,
    settings: ProviderSettings,
    endpoint: str,
) -> list[str]:
    endpoint = _valid_url(endpoint, settings.allow_insecure_http)
    payload = await _json_response(session, "GET", endpoint, settings)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
        raise ImageProviderError("invalid_response", "模型列表返回格式不受支持。")
    return [
        item["id"].strip()
        for item in payload["data"]
        if isinstance(item, Mapping)
        and isinstance(item.get("id"), str)
        and item["id"].strip()
    ]


async def generate_image(
    session: aiohttp.ClientSession,
    settings: ProviderSettings,
    prompt: str,
) -> tuple[bytes | None, str]:
    if settings.protocol == "openai_images":
        url = _join_endpoint(settings.base_url, "/images/generations")
        payload = {"model": settings.model, "prompt": prompt, "n": 1}
    else:
        url = _join_endpoint(settings.base_url, "/responses")
        payload = {
            "model": settings.model,
            "input": prompt,
            "tools": [{"type": "image_generation"}],
        }
    response = await _json_response(session, "POST", url, settings, payload)
    return extract_image_result(response)
