from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "astrbot_img_web_test"


def _identity_decorator(*args, **kwargs):
    return lambda handler: handler


def _install_astrbot_stubs():
    astrbot = types.ModuleType("astrbot")
    astrbot.__path__ = []
    api = types.ModuleType("astrbot.api")
    api.__path__ = []
    api.AstrBotConfig = dict
    api.logger = logging.getLogger("astrbot_img_web_test")

    event = types.ModuleType("astrbot.api.event")
    event.AstrMessageEvent = type("AstrMessageEvent", (), {})
    event.MessageChain = type("MessageChain", (), {})
    event.filter = types.SimpleNamespace(
        EventMessageType=types.SimpleNamespace(GROUP_MESSAGE=object()),
        command=_identity_decorator,
        event_message_type=_identity_decorator,
        on_decorating_result=_identity_decorator,
        on_astrbot_loaded=_identity_decorator,
    )

    components = types.ModuleType("astrbot.api.message_components")
    components.At = type("At", (), {})
    components.Plain = type("Plain", (), {})
    components.Reply = type("Reply", (), {})

    star = types.ModuleType("astrbot.api.star")
    star.Context = type("Context", (), {})
    star.Star = type("Star", (), {"__init__": lambda self, context: None})

    web_api = types.ModuleType("astrbot.api.web")
    web_api.request = None
    web_api.json_response = lambda data=None, **kwargs: FakeResponse(data, **kwargs)
    web_api.error_response = lambda message, **kwargs: FakeResponse(
        {"status": "error", "message": message, "data": kwargs.get("data", {})},
        status_code=kwargs.get("status_code", 400),
    )

    core = types.ModuleType("astrbot.core")
    core.__path__ = []
    utils = types.ModuleType("astrbot.core.utils")
    utils.__path__ = []
    astrbot_path = types.ModuleType("astrbot.core.utils.astrbot_path")
    astrbot_path.get_astrbot_data_path = lambda: str(ROOT / "data")

    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.message_components": components,
            "astrbot.api.star": star,
            "astrbot.api.web": web_api,
            "astrbot.core": core,
            "astrbot.core.utils": utils,
            "astrbot.core.utils.astrbot_path": astrbot_path,
        }
    )


class FakeResponse:
    def __init__(self, data, *, status_code=200):
        self.data = data
        self.status_code = status_code
        self.body = json.dumps(data, ensure_ascii=False).encode()


class FakeRequest:
    def __init__(self, payload=None):
        self.payload = payload or {}

    async def json(self, default=None):
        return self.payload if self.payload is not None else default


_install_astrbot_stubs()
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
for module_name in ("providers", "main"):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.{module_name}",
        ROOT / f"{module_name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
main_module = sys.modules[f"{PACKAGE}.main"]


class FakeConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_count = 0

    def save_config(self):
        self.save_count += 1


def _plugin(providers=None, current_id=""):
    plugin = object.__new__(main_module.ImageGenerationPlugin)
    plugin.config = FakeConfig(
        manual_providers=list(providers or []),
        manual_provider_id=current_id,
        request_timeout=10,
    )
    plugin._model_cache = {}
    plugin._session = None
    plugin._terminated = False
    plugin._cooldowns = {}
    return plugin


def _payload(**overrides):
    data = {
        "id": "demo",
        "name": "Demo",
        "protocol": "openai_images",
        "base_url": "https://images.example/v1",
        "api_key": "new-secret",
        "model": "",
        "models": [],
        "models_endpoint": "",
        "enabled": True,
        "allow_insecure_http": False,
        "set_current": True,
    }
    data.update(overrides)
    return data


def _json(response):
    return json.loads(response.body.decode())


def test_provider_get_redacts_api_key_and_sensitive_url_parts():
    plugin = _plugin(
        [
            {
                **_payload(
                    api_key="secret-api-key",
                    base_url="https://user@example.com/v1?token=query-secret",
                    models_endpoint="https://user@example.com/models?key=query-secret",
                ),
            }
        ]
    )

    result = _json(asyncio.run(plugin.web_get_providers()))
    serialized = json.dumps(result)

    assert "secret-api-key" not in serialized
    assert "query-secret" not in serialized
    assert "user@" not in serialized
    assert result["providers"][0]["has_api_key"] is True
    assert result["providers"][0]["base_url"] == "https://example.com/v1"
    assert result["providers"][0]["models_endpoint"] == "https://example.com/models"


def test_provider_get_response_does_not_expose_framework_source():
    plugin = _plugin([_payload(api_key="secret")], current_id="demo")

    result = _json(asyncio.run(plugin.web_get_providers()))

    assert "provider_mode" not in result


@pytest.mark.parametrize("protocol", ["openai_images", "openai_responses"])
def test_save_provider_adds_manual_provider_and_sets_current(protocol):
    plugin = _plugin()
    main_module.request = FakeRequest(
        _payload(protocol=protocol, model="custom-image-model")
    )

    response = asyncio.run(plugin.web_save_provider())

    assert response.status_code == 200
    assert "provider_mode" not in plugin.config
    assert plugin.config["manual_provider_id"] == "demo"
    saved = plugin.config["manual_providers"][0]
    assert saved["api_key"] == "new-secret"
    assert saved["model"] == "custom-image-model"
    assert saved["protocol"] == saved["__template_key"] == protocol
    assert _json(response)["provider"]["has_api_key"] is True
    assert "new-secret" not in response.body.decode()
    assert plugin.config.save_count == 1


@pytest.mark.parametrize("protocol", ["openai_images", "openai_responses"])
def test_get_providers_is_read_only_and_recognizes_native_templates(protocol):
    native = _payload(
        protocol=protocol,
        __template_key=protocol,
        api_key="existing-secret",
        model="image-model",
        models=["image-model", "other-model"],
    )
    native.pop("protocol")
    legacy = _payload(id="legacy", protocol=protocol, model="old-model")
    plugin = _plugin([native, legacy], current_id="demo")
    before = json.loads(json.dumps(plugin.config))

    result = _json(asyncio.run(plugin.web_get_providers()))

    assert plugin.config == before
    assert plugin.config.save_count == 0
    assert result["providers"][0]["protocol"] == protocol
    assert result["providers"][0]["model"] == "image-model"
    assert result["providers"][0]["models"] == ["image-model", "other-model"]
    assert result["providers"][1]["protocol"] == protocol
    assert result["current_provider_id"] == "demo"
    assert "existing-secret" not in json.dumps(result)
    assert plugin._provider_settings().protocol == protocol
    assert plugin._provider_settings().model == "image-model"


@pytest.mark.parametrize("protocol", ["openai_images", "openai_responses"])
@pytest.mark.parametrize("enabled", [True, False])
def test_initialize_repairs_legacy_templates_once_without_requests(protocol, enabled):
    plugin = _plugin(
        [
            _payload(
                protocol=protocol,
                api_key="existing-secret",
                model="chosen-model",
                models=["other-model", "chosen-model"],
            ),
            _payload(id="empty", protocol=protocol, __template_key=""),
        ],
        current_id="demo",
    )
    plugin.config["enabled"] = enabled
    expected = json.loads(json.dumps(plugin.config))
    for item in expected["manual_providers"]:
        item["__template_key"] = protocol
    plugin._http = AsyncMock(side_effect=AssertionError("Unexpected HTTP request"))

    asyncio.run(plugin.initialize())

    assert plugin.config == expected
    assert plugin.config.save_count == 1
    asyncio.run(plugin.initialize())
    assert plugin.config == expected
    assert plugin.config.save_count == 1
    plugin._http.assert_not_awaited()
    assert plugin._provider_settings().model == "chosen-model"


def test_initialize_preserves_existing_templates_and_unknown_protocols():
    plugin = _plugin(
        [
            _payload(__template_key="openai_images"),
            _payload(id="native", protocol="", __template_key="openai_responses"),
            _payload(id="unknown-template", __template_key="custom_template"),
            _payload(id="unknown-protocol", protocol="unsupported"),
            _payload(id="conflicting", __template_key="openai_responses"),
        ]
    )
    before = json.loads(json.dumps(plugin.config))

    asyncio.run(plugin.initialize())

    assert plugin.config == before
    assert plugin.config.save_count == 0


@pytest.mark.parametrize("protocol", ["openai_images", "openai_responses"])
def test_edit_preserves_empty_key_and_current_selection(protocol):
    plugin = _plugin(
        [
            {
                **_payload(api_key="existing-secret", model="old-model"),
            }
        ],
        current_id="demo",
    )
    main_module.request = FakeRequest(
        _payload(
            original_id="demo",
            protocol=protocol,
            api_key="",
            model="custom-model",
            set_current=False,
        )
    )

    response = asyncio.run(plugin.web_save_provider())

    assert response.status_code == 200
    assert plugin.config["manual_providers"][0]["api_key"] == "existing-secret"
    assert plugin.config["manual_providers"][0]["model"] == "custom-model"
    assert plugin.config["manual_providers"][0]["protocol"] == protocol
    assert plugin.config["manual_providers"][0]["__template_key"] == protocol
    assert plugin.config["manual_provider_id"] == "demo"
    assert "existing-secret" not in response.body.decode()


@pytest.mark.parametrize("protocol", ["openai_images", "openai_responses"])
def test_saved_provider_survives_reload_and_generates_with_selected_model(protocol):
    async def run():
        calls = []

        async def handler(request):
            calls.append((request.path, await request.json()))
            if protocol == "openai_images":
                return web.json_response({"data": [{"b64_json": "cG5nLWRhdGE="}]})
            return web.json_response(
                {"output": [{"type": "image_generation_call", "result": "cG5nLWRhdGE="}]}
            )

        app = web.Application()
        app.router.add_post("/{tail:.*}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        restored = None
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            with tempfile.TemporaryDirectory(prefix=".test-providers-", dir=ROOT) as temp:
                path = Path(temp) / "config.json"

                class PersistentConfig(FakeConfig):
                    def save_config(self):
                        super().save_config()
                        path.write_text(json.dumps(self), encoding="utf-8")

                plugin = _plugin()
                plugin.config = PersistentConfig(plugin.config)
                main_module.request = FakeRequest(
                    _payload(
                        protocol=protocol,
                        base_url=f"http://127.0.0.1:{port}/v1",
                        allow_insecure_http=True,
                        model="chosen-model",
                        models=["other-model", "chosen-model"],
                    )
                )
                response = await plugin.web_save_provider()
                assert response.status_code == 200
                stored = json.loads(path.read_text(encoding="utf-8"))
                assert stored["manual_providers"][0]["__template_key"] == protocol

                restored = _plugin()
                restored.config = PersistentConfig(stored)
                await restored.initialize()
                assert restored.config.save_count == 0
                result = _json(await restored.web_get_providers())
                assert result["current_provider_id"] == "demo"
                assert result["providers"][0]["models"] == [
                    "other-model",
                    "chosen-model",
                ]
                assert result["providers"][0]["model"] == "chosen-model"
                assert result["providers"][0]["protocol"] == protocol
                assert not calls
                image = await restored._generate(
                    "draw a cat", restored._provider_settings()
                )
                assert image == b"png-data"
                assert len(calls) == 1
                endpoint, payload = calls[0]
                assert payload["model"] == "chosen-model"
                if protocol == "openai_images":
                    assert endpoint == "/v1/images/generations"
                    assert payload["prompt"] == "draw a cat"
                else:
                    assert endpoint == "/v1/responses"
                    assert payload["input"] == "draw a cat"
                assert json.loads(path.read_text(encoding="utf-8")) == stored
        finally:
            if restored is not None:
                await restored.terminate()
            await runner.cleanup()

    asyncio.run(run())


def test_legacy_framework_fields_are_ignored_for_runtime_selection():
    plugin = _plugin([_payload(api_key="manual-secret", model="manual-model")], current_id="demo")
    plugin.config.update(
        provider_mode="astrbot",
        astrbot_provider_id="framework-provider",
        astrbot_model_id="framework-model",
    )

    settings = plugin._provider_settings()

    assert settings.provider_id == "demo"
    assert settings.model == "manual-model"
    assert settings.api_key == "manual-secret"


def test_missing_manual_provider_has_manual_configuration_error():
    plugin = _plugin()

    with pytest.raises(main_module.ImageProviderError) as error:
        plugin._provider_settings()

    assert error.value.code == "not_configured"
    assert "插件手动供应商" in main_module._display_error(error.value)


def test_save_provider_does_not_clear_existing_models_on_empty_form_list():
    plugin = _plugin(
        [
            _payload(
                api_key="existing-secret",
                model="old-model",
                models=["old-model", "second-model"],
            )
        ],
        current_id="demo",
    )
    main_module.request = FakeRequest(
        _payload(
            original_id="demo",
            api_key="",
            model="old-model",
            models=[],
            set_current=False,
        )
    )

    response = asyncio.run(plugin.web_save_provider())

    assert response.status_code == 200
    assert plugin.config["manual_providers"][0]["models"] == [
        "old-model",
        "second-model",
    ]
    settings = main_module.manual_provider_settings(
        plugin.config["manual_providers"],
        plugin.config["manual_provider_id"],
        30,
    )
    assert settings.model == "old-model"


def test_renaming_provider_preserves_key_and_updates_current_id():
    plugin = _plugin(
        [
            {
                **_payload(api_key="existing-secret"),
            }
        ],
        current_id="demo",
    )
    main_module.request = FakeRequest(
        _payload(original_id="demo", id="renamed", api_key="", set_current=False)
    )

    response = asyncio.run(plugin.web_save_provider())

    assert response.status_code == 200
    assert plugin.config["manual_providers"][0]["id"] == "renamed"
    assert plugin.config["manual_providers"][0]["api_key"] == "existing-secret"
    assert plugin.config["manual_provider_id"] == "renamed"


@pytest.mark.parametrize(
    ("overrides", "status_code"),
    [
        ({"protocol": "chat"}, 400),
        ({"base_url": "http://images.example/v1"}, 400),
        ({"base_url": "https://["}, 400),
        ({"base_url": "https://images.example/v1?key=secret"}, 400),
        ({"api_key": ""}, 400),
        ({"allow_insecure_http": "true"}, 400),
    ],
)
def test_save_provider_rejects_invalid_config(overrides, status_code):
    plugin = _plugin()
    main_module.request = FakeRequest(_payload(**overrides))

    response = asyncio.run(plugin.web_save_provider())

    assert response.status_code == status_code
    assert not plugin.config.get("manual_providers")


def test_save_provider_rejects_duplicate_id():
    plugin = _plugin([_payload(api_key="first-secret")])
    main_module.request = FakeRequest(_payload(api_key="second-secret"))

    response = asyncio.run(plugin.web_save_provider())

    assert response.status_code == 409
    assert plugin.config["manual_providers"][0]["api_key"] == "first-secret"


def test_fetch_models_falls_back_and_persists_models_and_default():
    async def run():
        async def handler(request):
            if request.path == "/v1/models":
                return web.json_response({"data": [{"id": "image-v1"}]})
            return web.Response(status=404, text="not found")

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        plugin = _plugin(
            [
                _payload(
                    base_url=f"http://127.0.0.1:{port}",
                    models_endpoint=f"http://127.0.0.1:{port}/missing",
                    allow_insecure_http=True,
                    api_key="test-secret",
                )
            ]
        )
        main_module.request = FakeRequest({"id": "demo"})
        try:
            async with aiohttp.ClientSession() as session:
                plugin._session = session
                response = await plugin.web_fetch_provider_models()
            assert response.status_code == 200
            assert plugin.config["manual_providers"][0]["models"] == ["image-v1"]
            assert plugin.config["manual_providers"][0]["model"] == "image-v1"
            assert plugin._model_cache["demo"] == ["image-v1"]
            assert _json(response)["models"] == ["image-v1"]
            assert "test-secret" not in response.body.decode()
            assert plugin.config.save_count == 1
        finally:
            await plugin.terminate()
            await runner.cleanup()

    asyncio.run(run())


def test_fetch_model_failure_preserves_existing_model_cache():
    async def run():
        async def handler(request):
            return web.Response(status=404, text="not found")

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        plugin = _plugin(
            [
                _payload(
                    base_url=f"http://127.0.0.1:{port}",
                    models_endpoint=f"http://127.0.0.1:{port}/missing",
                    allow_insecure_http=True,
                    api_key="test-secret",
                    models=["old-model"],
                    model="old-model",
                )
            ]
        )
        main_module.request = FakeRequest({"id": "demo"})
        try:
            async with aiohttp.ClientSession() as session:
                plugin._session = session
                response = await plugin.web_fetch_provider_models()
            assert response.status_code == 502
            assert plugin.config["manual_providers"][0]["models"] == ["old-model"]
            assert plugin.config["manual_providers"][0]["model"] == "old-model"
            assert plugin.config.save_count == 0
            assert "test-secret" not in response.body.decode()
        finally:
            await plugin.terminate()
            await runner.cleanup()

    asyncio.run(run())


def test_delete_provider_clears_current_and_secret_cache():
    plugin = _plugin(
        [
            _payload(api_key="secret"),
            _payload(id="next", name="Next", api_key="other-secret"),
        ],
        current_id="demo",
    )
    plugin._model_cache = {"demo": ["image-model"]}
    main_module.request = FakeRequest({"id": "demo"})

    response = asyncio.run(plugin.web_delete_provider_post())

    assert response.status_code == 200
    assert [item["id"] for item in plugin.config["manual_providers"]] == ["next"]
    assert plugin.config["manual_provider_id"] == "next"
    assert "demo" not in plugin._model_cache
