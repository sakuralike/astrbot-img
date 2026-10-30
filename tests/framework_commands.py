from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import AsyncMock, patch


async def verify(source: Path, initial_config: dict) -> int:
    from astrbot.api.message_components import At, Image, Plain
    from astrbot.core.pipeline.context import PipelineContext
    from astrbot.core.pipeline.process_stage.method.star_request import (
        StarRequestSubStage,
    )
    from astrbot.core.pipeline.waking_check.stage import WakingCheckStage
    from astrbot.core.platform.astr_message_event import AstrMessageEvent
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.platform.platform_metadata import PlatformMetadata
    from astrbot.core.star import session_plugin_manager
    from astrbot.core.star.star import star_map
    from astrbot.core.star.star_handler import star_handlers_registry

    package_name = "image_command_framework_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.main", source / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    metadata = star_map[module.__name__]
    metadata.name = "astrbot_img"
    metadata.activated = True

    class Config(dict):
        def save_config(self):
            self.saves = getattr(self, "saves", 0) + 1

    class Event(AstrMessageEvent):
        def __init__(self, text, platform, admin, send_error=False):
            message = AstrBotMessage()
            message.type = MessageType.GROUP_MESSAGE
            message.self_id = "test-bot"
            message.session_id = "test-group"
            message.message_id = "isolated-message"
            message.group_id = "test-group"
            message.sender = MessageMember(
                user_id="test-admin" if admin else "test-member",
                nickname="test-user",
            )
            message.message = [Plain(text)]
            if not text.startswith("/"):
                message.message.insert(0, At(qq=message.self_id))
            message.message_str = text
            message.raw_message = {}
            super().__init__(
                text,
                message,
                PlatformMetadata(platform, "isolated", "test-platform"),
                message.session_id,
            )
            self.sent = []
            self.send_error = send_error

        async def send(self, chain):
            assert chain.use_markdown_ is False
            for component in chain.chain:
                if isinstance(component, Image):
                    assert Path(component.path).read_bytes() == b"mock-image-bytes"
                    if self.send_error:
                        raise OSError("isolated send failure")
            self.sent.append(chain)

    pipeline_context = PipelineContext(
        astrbot_config={
            "admins_id": ["test-admin"],
            "wake_prefix": ["/"],
            "plugin_set": ["astrbot_img"],
            "platform_settings": {},
            "provider_settings": {"prompt_prefix": "", "identifier": False},
        },
        plugin_manager=None,
        astrbot_config_id="isolated",
    )
    waking = WakingCheckStage()
    await waking.initialize(pipeline_context)
    dispatch = StarRequestSubStage()
    await dispatch.initialize(pipeline_context)

    cases = [
        ("slash-empty", "/生图", "", "usage"),
        ("normalized-empty", "生图", "", "usage"),
        ("slash-prompt", "/生图 cat", "", "image"),
        ("normalized-prompt", "生图 cat", "", "image"),
        ("tab-prompt", "/生图\tcat", "", "image"),
        ("newline-prompt", "/生图\ncat", "", "image"),
        ("custom-empty", "/画图", "画图", "usage"),
        ("custom-slash", "/画图 cat", "画图", "image"),
        ("custom-normalized", "画图 cat", "画图", "image"),
        ("custom-default-alias", "/生图 cat", "画图", "image"),
        ("boundary", "/生图片段 cat", "", "unmatched"),
        ("models-admin", "/获取生图模型", "", "models"),
        ("models-member", "/获取生图模型", "", "permission"),
        ("models-custom", "/刷新图片模型", "画图", "models"),
        ("disabled-generate", "/生图 cat", "", "disabled"),
        ("disabled-models", "/获取生图模型", "", "disabled"),
        ("timeout", "/生图 cat", "", "timeout"),
        ("provider-error", "/生图 cat", "", "provider_error"),
        ("send-error", "/生图 cat", "", "send_error"),
        ("onebot-image", "/生图 cat", "", "image"),
    ]
    for name, text, custom, expected in cases:
        config = Config(copy.deepcopy(initial_config))
        config.update(
            enabled=expected != "disabled",
            generate_command=custom or "生图",
            models_command="刷新图片模型" if custom else "获取生图模型",
            allowed_groups=[],
            cooldown_seconds=60,
            max_concurrency=2,
        )
        plugin = module.ImageGenerationPlugin(
            types.SimpleNamespace(register_web_api=lambda *args: None), config
        )
        metadata.star_cls = plugin
        for handler in star_handlers_registry.get_handlers_by_module_name(
            module.__name__
        ):
            handler.handler = getattr(plugin, handler.handler_name)
        await plugin.initialize()
        config.saves = 0
        plugin._http = AsyncMock(side_effect=AssertionError("network forbidden"))
        plugin._generate = AsyncMock(return_value=b"mock-image-bytes")
        plugin._fetch_manual_models = AsyncMock(return_value=["mock-image-model"])
        if expected in ("timeout", "provider_error"):
            plugin._generate.side_effect = module.ImageProviderError(
                expected, "isolated provider error"
            )
        event = Event(
            text,
            "aiocqhttp" if name == "onebot-image" else "qq_official",
            expected == "models" or name == "disabled-models",
            send_error=expected == "send_error",
        )
        try:
            with (
                patch.object(
                    session_plugin_manager.sp, "get_async", AsyncMock(return_value={})
                ),
                patch.object(module.logger, "exception"),
            ):
                await waking.process(event)
                handlers = event.get_extra("activated_handlers")
                assert len(handlers) == (0 if expected == "unmatched" else 1), name
                async for _ in dispatch.process(event):
                    pass
            replies = "".join(
                part.text
                for chain in event.sent
                for part in chain.chain
                if isinstance(part, Plain)
            )
            if expected in ("disabled", "unmatched"):
                assert not event.sent, name
            elif expected == "usage":
                assert replies.startswith(f"用法：/{plugin.generate_command} "), name
            elif expected == "permission":
                assert replies == "只有管理员可以获取生图模型。", name
                plugin._fetch_manual_models.assert_not_awaited()
            elif expected == "models":
                assert "生图模型列表：" in replies, name
                assert "mock-image-model" in replies, name
                assert config.saves > 0, name
                assert all(
                    item["models"] == ["mock-image-model"]
                    for item in plugin._manual_provider_items()
                ), name
            elif expected == "image":
                assert len(event.sent) == 1, name
                assert len(event.sent[0].chain) == 2, name
                assert isinstance(event.sent[0].chain[0], At), name
                assert event.sent[0].chain[0].qq == event.get_sender_id(), name
                assert isinstance(event.sent[0].chain[1], Image), name
                assert plugin._cooldowns, name
            else:
                expected_reply = {
                    "timeout": "生图请求超时，请稍后再试。",
                    "provider_error": "生图供应商暂时不可用，请稍后再试。",
                    "send_error": "图片生成成功，但发送图片失败。",
                }[expected]
                assert replies == expected_reply, name
                assert not plugin._cooldowns, name
            if expected in ("image", "timeout", "provider_error", "send_error"):
                plugin._generate.assert_awaited_once()
                assert plugin._generate.await_args.args[0] == "cat", name
                assert (
                    plugin._generate.await_args.args[1].model
                    == plugin._provider_settings().model
                ), name
            else:
                plugin._generate.assert_not_awaited()
            if expected not in ("disabled", "unmatched"):
                assert event.is_stopped() and event.call_llm is False, name
            assert plugin._semaphore._value == 2, name
            assert not list(plugin._temp_dir.iterdir()), name
            plugin._http.assert_not_awaited()
            print(f"PASS {name}")
        finally:
            await plugin.terminate()
    return len(cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--temp-parent", type=Path, required=True)
    args = parser.parse_args()
    if args.config:
        original_config = args.config.read_bytes()
        config = json.loads(original_config.decode("utf-8-sig"))
    else:
        config = {
            "manual_provider_id": "test",
            "manual_providers": [
                {
                    "id": "test",
                    "protocol": "openai_images",
                    "__template_key": "openai_images",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "test-only-key",
                    "model": "mock-image-model",
                    "models": ["mock-image-model"],
                    "enabled": True,
                }
            ],
        }
    with tempfile.TemporaryDirectory(prefix="image-command-test-", dir=args.temp_parent) as root:
        os.environ["ASTRBOT_ROOT"] = root
        os.environ["TMPDIR"] = root
        tempfile.tempdir = root
        count = asyncio.run(verify(args.source, config))
    if args.config:
        assert args.config.read_bytes() == original_config, "production config changed"
    print(f"PASS {count} real-framework command cases; no network or live sends")


if __name__ == "__main__":
    main()
