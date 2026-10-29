from __future__ import annotations

import asyncio
import base64
import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import aiohttp
import pytest
from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "astrbot_img_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    f"{PACKAGE}.providers",
    ROOT / "providers.py",
    submodule_search_locations=[],
)
providers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = providers
assert spec.loader is not None
spec.loader.exec_module(providers)


def test_extract_images_and_responses_payload():
    encoded = base64.b64encode(b"png-data").decode()
    assert providers.extract_image_result({"data": [{"b64_json": encoded}]}) == (
        b"png-data",
        "",
    )
    assert providers.extract_image_result(
        {"output": [{"type": "image_generation_call", "result": encoded}]}
    ) == (b"png-data", "")


def test_extract_data_url_from_image_fields():
    encoded = base64.b64encode(b"png-data").decode()
    data_url = f"data:image/png;base64,{encoded}"
    assert providers.extract_image_result({"data": [{"url": data_url}]}) == (
        b"png-data",
        "",
    )
    assert providers.extract_image_result(
        {"output": [{"type": "image_generation_call", "result": data_url}]}
    ) == (b"png-data", "")


def test_extract_rejects_async_task():
    with pytest.raises(providers.ImageProviderError) as error:
        providers.extract_image_result({"id": "task-1"})
    assert error.value.code == "async_unsupported"


def test_manual_template_provider_uses_first_model():
    settings = providers.settings_from_manual(
        {
            "__template_key": "openai_images",
            "id": "demo",
            "base_url": "https://example.com/v1",
            "api_key": "secret",
            "models": ["image-model"],
        },
        timeout=30,
    )
    assert settings.protocol == "openai_images"
    assert settings.model == "image-model"
    assert settings.base_url == "https://example.com/v1"


def test_astrbot_provider_reads_merged_connection_info():
    class Provider:
        provider_config: ClassVar[dict[str, object]] = {
            "type": "openai_responses",
            "api_base": "https://example.com/v1",
            "model": "image-model",
            "key": ["secret"],
        }

        def get_current_key(self):
            return "secret"

    settings = providers.settings_from_astrbot_provider(
        "provider-1",
        Provider(),
        "",
        30,
    )
    assert settings.protocol == "openai_responses"
    assert settings.model == "image-model"


def test_astrbot_provider_accepts_string_api_key():
    class Provider:
        provider_config: ClassVar[dict[str, object]] = {
            "type": "openai_images",
            "api_base": "https://example.com/v1",
            "model": "image-model",
            "api_key": "secret",
        }

    settings = providers.settings_from_astrbot_provider(
        "provider-1",
        Provider(),
        "",
        30,
    )
    assert settings.api_key == "secret"


def test_plain_chat_provider_is_not_assumed_to_generate_images():
    class Provider:
        provider_config: ClassVar[dict[str, object]] = {
            "type": "openai_chat_completion",
            "api_base": "https://example.com/v1",
            "model": "chat-model",
            "key": ["secret"],
        }

        def get_current_key(self):
            return "secret"

    with pytest.raises(providers.ImageProviderError) as error:
        providers.settings_from_astrbot_provider("provider-1", Provider(), "", 30)
    assert error.value.code == "unsupported_provider"


def test_http_url_is_rejected_by_default():
    with pytest.raises(providers.ImageProviderError):
        providers.settings_from_manual(
            {
                "id": "demo",
                "protocol": "openai_images",
                "base_url": "http://example.com/v1",
                "api_key": "secret",
                "model": "image-model",
            },
            30,
        )


def test_duplicate_manual_provider_ids_are_rejected():
    with pytest.raises(providers.ImageProviderError) as error:
        providers.manual_provider_settings(
            [
                {
                    "id": "demo",
                    "protocol": "openai_images",
                    "base_url": "https://example.com/v1",
                    "api_key": "secret",
                    "model": "image-model",
                },
                {
                    "id": "demo",
                    "protocol": "openai_images",
                    "base_url": "https://example.com/v1",
                    "api_key": "secret2",
                    "model": "image-model",
                },
            ],
            "",
            30,
        )
    assert error.value.code == "invalid_config"


def _run(coro):
    return asyncio.run(coro)


async def _with_test_server(callback):
    app = web.Application()

    async def image_generation(request):
        body = await request.json()
        assert request.headers["Authorization"] == "Bearer secret"
        assert body["prompt"] == "一只猫"
        assert body["n"] == 1
        return web.json_response(
            {"data": [{"b64_json": base64.b64encode(b"png-data").decode()}]}
        )

    async def responses_generation(request):
        body = await request.json()
        assert request.headers["Authorization"] == "Bearer secret"
        assert body["input"] == "一只猫"
        assert body["tools"] == [{"type": "image_generation"}]
        return web.json_response(
            {
                "output": [
                    {
                        "type": "image_generation_call",
                        "result": (
                            "data:image/png;base64,"
                            + base64.b64encode(b"response-image").decode()
                        ),
                    }
                ]
            }
        )

    async def status_response(request):
        return web.Response(status=int(request.match_info["status"]), text="error")

    async def invalid_json(request):
        return web.Response(text="not-json")

    async def delayed_response(request):
        await asyncio.sleep(0.2)
        return web.json_response({"data": []})

    app.router.add_post("/images/generations", image_generation)
    app.router.add_post("/responses", responses_generation)
    app.router.add_get("/status/{status}", status_response)
    app.router.add_get("/invalid-json", invalid_json)
    app.router.add_get("/delayed", delayed_response)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets
    assert sockets
    port = sockets[0].getsockname()[1]
    try:
        return await callback(f"http://127.0.0.1:{port}")
    finally:
        await runner.cleanup()


def test_openai_images_base64_request_and_responses_data_url():
    async def run():
        async def check(base_url):
            settings = providers.ProviderSettings(
                provider_id="demo",
                protocol="openai_images",
                base_url=base_url,
                api_key="secret",
                model="image-model",
                timeout=5,
                allow_insecure_http=True,
            )
            async with aiohttp.ClientSession() as session:
                image_bytes, image_url = await providers.generate_image(
                    session, settings, "一只猫"
                )
                assert image_bytes == b"png-data"
                assert image_url == ""

            responses = replace(settings, protocol="openai_responses")
            async with aiohttp.ClientSession() as session:
                image_bytes, image_url = await providers.generate_image(
                    session, responses, "一只猫"
                )
                assert image_bytes == b"response-image"
                assert image_url == ""

        await _with_test_server(check)

    _run(run())


def test_provider_http_errors_and_timeout_are_stable():
    async def run():
        async def check(base_url):
            settings = providers.ProviderSettings(
                provider_id="demo",
                protocol="openai_images",
                base_url=base_url,
                api_key="secret",
                model="image-model",
                timeout=5,
                allow_insecure_http=True,
            )
            expected = {
                401: "unauthorized",
                403: "unauthorized",
                429: "rate_limited",
                500: "provider_error",
            }
            async with aiohttp.ClientSession() as session:
                for status, code in expected.items():
                    with pytest.raises(providers.ImageProviderError) as error:
                        await providers._json_response(
                            session,
                            "GET",
                            f"{base_url}/status/{status}",
                            settings,
                        )
                    assert error.value.code == code

                with pytest.raises(providers.ImageProviderError) as error:
                    await providers._json_response(
                        session,
                        "GET",
                        f"{base_url}/invalid-json",
                        settings,
                    )
                assert error.value.code == "invalid_response"

                short_timeout = replace(settings, timeout=0.05)
                with pytest.raises(providers.ImageProviderError) as error:
                    await providers._json_response(
                        session,
                        "GET",
                        f"{base_url}/delayed",
                        short_timeout,
                    )
                assert error.value.code == "timeout"

        await _with_test_server(check)

    _run(run())


def test_custom_models_endpoint_obeys_https_policy():
    settings = providers.ProviderSettings(
        provider_id="demo",
        protocol="openai_images",
        base_url="https://example.com/v1",
        api_key="secret",
        model="image-model",
        timeout=5,
    )
    with pytest.raises(providers.ImageProviderError) as error:
        _run(providers.fetch_model_ids(None, settings, "http://example.com/models"))
    assert error.value.code == "invalid_config"
