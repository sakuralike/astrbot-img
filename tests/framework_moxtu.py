from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import io
import os
import sqlite3
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import AsyncMock, patch


async def verify(image_source: Path, moxtu_source: Path, run_root: Path) -> int:
    import aiohttp
    from PIL import Image as PillowImage

    from astrbot.api.message_components import At, Image, Plain
    from astrbot.api.star import Context
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
    from astrbot.core.star.star import star_map, star_registry
    from astrbot.core.star.star_handler import star_handlers_registry

    def load(name, source):
        package = types.ModuleType(name)
        package.__path__ = [str(source)]
        sys.modules[name] = package
        spec = importlib.util.spec_from_file_location(f"{name}.main", source / "main.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    image_module = load("image_moxtu_framework_test", image_source)
    moxtu_module = load("moxtu_framework_test", moxtu_source)
    image_metadata = star_map[image_module.__name__]
    image_metadata.name = "astrbot_img"
    moxtu_metadata = star_map[moxtu_module.__name__]
    moxtu_metadata.name = "astrbot_plugin_moxtu_gallery"

    data = io.BytesIO()
    PillowImage.new("RGB", (7, 9), (21, 93, 175)).save(data, format="PNG")
    image_bytes = data.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    command = "/\u751f\u56fe cat"
    timeline = []

    class Config(dict):
        def save_config(self):
            raise AssertionError("Unexpected configuration persistence")

    class Event(AstrMessageEvent):
        def __init__(self, case, *, suffix="", platform="qq_official", send_error=False):
            message = AstrBotMessage()
            message.type = MessageType.GROUP_MESSAGE
            message.self_id = "test-bot"
            message.session_id = "test-group"
            message.message_id = f"test-message-{case}{suffix}"
            message.group_id = "test-group"
            message.sender = MessageMember(user_id=f"test-user{suffix}", nickname="Tester")
            message.message = [Plain(command)]
            message.message_str = command
            message.raw_message = {}
            super().__init__(
                command,
                message,
                PlatformMetadata(platform, "isolated", "test-platform"),
                message.session_id,
            )
            self.sent = []
            self.send_error = send_error

        async def send(self, chain):
            assert chain.use_markdown_ is False
            if any(isinstance(part, Image) for part in chain.chain):
                mentions = [part.qq for part in chain.chain if isinstance(part, At)]
                assert mentions == [self.get_sender_id()]
                timeline.append("send-image")
                part = next(part for part in chain.chain if isinstance(part, Image))
                assert Path(part.path).read_bytes() == image_bytes
                if self.send_error:
                    raise OSError("isolated image delivery failure")
            else:
                timeline.append("send-text")
            self.sent.append(chain)

    framework_config = {
        "admins_id": [],
        "wake_prefix": ["/"],
        "plugin_set": ["astrbot_img", "astrbot_plugin_moxtu_gallery"],
        "platform_settings": {},
        "provider_settings": {"prompt_prefix": "", "identifier": False},
    }
    context = Context(
        event_queue=asyncio.Queue(),
        config=framework_config,
        db=None,
        provider_manager=None,
        platform_manager=None,
        conversation_manager=None,
        message_history_manager=None,
        persona_manager=None,
        astrbot_config_mgr=None,
        knowledge_base_manager=None,
        cron_manager=None,
    )
    context.registered_web_apis = []
    pipeline_context = PipelineContext(
        astrbot_config=framework_config,
        plugin_manager=None,
        astrbot_config_id="isolated",
    )
    waking = WakingCheckStage()
    dispatch = StarRequestSubStage()
    await waking.initialize(pipeline_context)
    await dispatch.initialize(pipeline_context)

    async def run_event(event):
        with patch.object(
            session_plugin_manager.sp, "get_async", AsyncMock(return_value={})
        ):
            await waking.process(event)
            handlers = event.get_extra("activated_handlers")
            assert len(handlers) == 1 and handlers[0].handler_name == "generate"
            async for _ in dispatch.process(event):
                pass

    cases = (
        "success",
        "onebot",
        "off",
        "missing",
        "inactive",
        "old-version",
        "disabled",
        "group-denied",
        "caption-failed",
        "upload-failed",
        "upload-unknown",
        "send-image-failed",
        "reload",
        "cancel",
        "existing",
    )
    for case in cases:
        timeline.clear()
        case_root = run_root / case
        case_root.mkdir()
        os.environ["ASTRBOT_ROOT"] = str(case_root)
        image_metadata.activated = True
        moxtu_metadata.activated = case != "inactive"
        if moxtu_metadata not in star_registry:
            star_registry.append(moxtu_metadata)
        if case == "missing":
            star_registry.remove(moxtu_metadata)
        image_config = Config(
            enabled=True,
            auto_upload_to_moxtu=case != "off",
            cooldown_seconds=60,
            max_concurrency=2,
            manual_provider_id="test",
            manual_providers=[
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
        )
        moxtu_config = Config(
            enabled=case != "disabled",
            allowed_groups=["other-group"] if case == "group-denied" else ["test-group"],
            api_base_url="https://example.invalid",
            api_version="v2",
            api_token="test-only-token",
            album_id=42,
            storage_id=7,
            upload_is_public=False,
            upload_cooldown=0,
            group_upload_cooldown=0,
            caption_enabled=True,
            caption_failure_policy="reject",
        )
        upload_started = asyncio.Event()
        upload_release = asyncio.Event()
        upload_paths = []
        plugins = []

        async def fake_upload(path, **kwargs):
            timeline.append("upload")
            upload_paths.append(Path(path))
            assert Path(path).read_bytes() == image_bytes
            assert kwargs["album_id"] == 42 and kwargs["is_public"] is False
            assert kwargs["content_type"] == "image/png"
            if case == "upload-failed":
                raise moxtu_module.LskyError("test token must not leak", code="unauthorized")
            if case == "upload-unknown":
                raise moxtu_module.LskyError("test token must not leak", code="timeout")
            if case == "cancel":
                upload_started.set()
                await upload_release.wait()
            return moxtu_module.LskyPhoto(
                id=len(upload_paths),
                public_url=f"https://example.invalid/{len(upload_paths)}.png",
                filename="generated.png",
                visibility="private",
            )

        async def new_moxtu():
            plugin = moxtu_module.MoxtuGalleryPlugin(context, moxtu_config)
            plugins.append(plugin)
            await plugin._client.close()
            plugin._client = types.SimpleNamespace(
                upload=AsyncMock(side_effect=fake_upload),
                close=AsyncMock(),
                _resolved_version="v2",
            )
            plugin._caption_image = AsyncMock(
                return_value=moxtu_module.CaptionResult(
                    status="failed" if case == "caption-failed" else "ok",
                    caption="" if case == "caption-failed" else "A blue square.",
                    error_code="timeout" if case == "caption-failed" else "",
                )
            )
            plugin._send_text = AsyncMock(side_effect=AssertionError("Moxtu sent a reply"))
            plugin._send_image = AsyncMock(side_effect=AssertionError("Moxtu sent an image"))
            return plugin

        moxtu = await new_moxtu()
        moxtu_metadata.star_cls = object() if case == "old-version" else moxtu
        image = image_module.ImageGenerationPlugin(context, image_config)
        image_metadata.star_cls = image
        for handler in star_handlers_registry.get_handlers_by_module_name(
            image_module.__name__
        ):
            handler.handler = getattr(image, handler.handler_name)
        image._generate = AsyncMock(return_value=image_bytes)
        event = Event(
            case,
            platform="aiocqhttp" if case == "onebot" else "qq_official",
            send_error=case == "send-image-failed",
        )
        task = None
        try:
            with (
                patch.object(
                    context,
                    "get_registered_star",
                    wraps=context.get_registered_star,
                ) as lookup,
                patch.object(
                    aiohttp.ClientSession,
                    "_request",
                    AsyncMock(side_effect=AssertionError("Network forbidden")),
                ) as network,
                patch.object(image_module.logger, "exception"),
                patch.object(moxtu_module.logger, "exception"),
            ):
                if case == "cancel":
                    task = asyncio.create_task(run_event(event))
                    await asyncio.wait_for(upload_started.wait(), timeout=5)
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    else:
                        raise AssertionError("Upload cancellation was swallowed")
                else:
                    await run_event(event)
                if case == "reload":
                    await moxtu.terminate()
                    replacement = await new_moxtu()
                    moxtu_metadata.star_cls = replacement
                    await run_event(Event(case, suffix="-second"))
                    assert moxtu._client is None
                    replacement._client.upload.assert_awaited_once()
                if case == "existing":
                    result = await moxtu.upload_generated_image(event, image_bytes)
                    assert result["status"] == "existing", result
                if case == "off":
                    lookup.assert_not_called()
                else:
                    assert lookup.call_count == (2 if case == "reload" else 1)
                network.assert_not_awaited()
            expected_uploads = (
                0
                if case in {
                    "off", "missing", "inactive", "old-version", "disabled",
                    "group-denied", "caption-failed",
                }
                else 2 if case == "reload" else 1
            )
            assert len(upload_paths) == expected_uploads, case
            assert image._generate.await_count == (2 if case == "reload" else 1), case
            assert image._generate.await_args.args[0] == "cat", case
            if upload_paths:
                assert timeline.index("send-image") < timeline.index("upload"), timeline
            if case != "send-image-failed":
                assert image._cooldowns, case
            else:
                assert not image._cooldowns, case
            assert image._semaphore._value == 2, case
            assert not list(image._temp_dir.iterdir()), case
            assert all(not path.exists() for path in upload_paths), case
            for plugin in plugins:
                assert not plugin._running, case
                assert not plugin._temp_dir.exists() or not list(plugin._temp_dir.iterdir())
                plugin._send_text.assert_not_awaited()
                plugin._send_image.assert_not_awaited()
            with sqlite3.connect(moxtu._store.path) as db:
                db.row_factory = sqlite3.Row
                records = [dict(row) for row in db.execute("SELECT * FROM uploaded_images")]
            if case in {"success", "onebot", "send-image-failed", "reload", "existing"}:
                assert len(records) == (2 if case == "reload" else 1), case
                for record in records:
                    assert record["sender_id"].startswith("test-user"), record
                    assert record["sender_name"] == "Tester", record
                    assert record["group_id"] == "test-group", record
                    assert record["platform_id"] == "test-platform", record
                    assert record["platform_name"] == (
                        "aiocqhttp" if case == "onebot" else "qq_official"
                    ), record
                    assert record["bot_self_id"] == "test-bot", record
                    assert record["album_id"] == "42", record
                    assert record["sha256"] == image_hash, record
                    assert record["width"] == 7 and record["height"] == 9, record
                    assert record["mime_type"] == "image/png", record
                    assert record["size_bytes"] == len(image_bytes), record
                    assert record["caption_status"] == "ok", record
                    assert record["visibility"] == "private", record
                    assert record["upload_status"] == "success", record
            else:
                assert not any(record["upload_status"] == "success" for record in records)
            text = "".join(
                part.text for chain in event.sent for part in chain.chain
                if isinstance(part, Plain)
            )
            assert "test token must not leak" not in text
            if case not in {"off", "cancel"}:
                assert text, case
            print(f"PASS {case}")
        finally:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await image.terminate()
            for plugin in plugins:
                await plugin.terminate()
    return len(cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-source", type=Path, required=True)
    parser.add_argument("--moxtu-source", type=Path, required=True)
    parser.add_argument("--temp-parent", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(
        prefix="image-moxtu-framework-", dir=args.temp_parent
    ) as root:
        os.environ["ASTRBOT_ROOT"] = root
        os.environ["TMPDIR"] = root
        tempfile.tempdir = root
        count = asyncio.run(verify(args.image_source, args.moxtu_source, Path(root)))
    print(f"PASS {count} real-framework Moxtu cases; no network or live sends")


if __name__ == "__main__":
    main()
