from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from test_web_api import _payload, _plugin, main_module


def _event():
    return types.SimpleNamespace(
        get_group_id=lambda: "group",
        get_platform_id=lambda: "platform",
        get_self_id=lambda: "bot",
        get_sender_id=lambda: "member",
        get_message_str=lambda: "生图 cat",
        should_call_llm=Mock(),
        stop_event=Mock(),
    )


def _configured_plugin(tmp_path, *, enabled=True):
    plugin = _plugin([_payload(model="image-model")])
    plugin.config.update(enabled=True, auto_upload_to_moxtu=enabled)
    plugin.generate_command = "生图"
    plugin._semaphore = asyncio.Semaphore(2)
    plugin._temp_dir = tmp_path
    plugin._generate = AsyncMock(return_value=b"generated-image")
    plugin._send_image = AsyncMock()
    plugin._send_text = AsyncMock()
    plugin.context = types.SimpleNamespace(get_registered_star=Mock())
    return plugin


def _metadata(upload):
    return types.SimpleNamespace(
        activated=True,
        star_cls=types.SimpleNamespace(upload_generated_image=upload),
    )


def test_moxtu_upload_is_opt_in():
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["auto_upload_to_moxtu"]["default"] is False


def test_disabled_integration_does_not_lookup_plugin(tmp_path):
    plugin = _configured_plugin(tmp_path, enabled=False)
    asyncio.run(plugin.generate(_event()))
    plugin.context.get_registered_star.assert_not_called()
    plugin._send_text.assert_not_awaited()
    plugin._send_image.assert_awaited_once()


def test_image_is_sent_before_upload_and_cooldown_is_preserved(tmp_path):
    async def run():
        plugin = _configured_plugin(tmp_path)
        event = _event()
        order = []

        async def send_image(actual_event, path):
            assert actual_event is event
            assert path.read_bytes() == b"generated-image"
            order.append("image")

        async def upload(actual_event, data):
            assert actual_event is event
            assert data == b"generated-image"
            assert list(tmp_path.iterdir()) == []
            assert plugin._semaphore._value == 1
            order.append("upload")
            return {"status": "success", "photo_id": "1"}

        plugin._send_image.side_effect = send_image
        target = AsyncMock(side_effect=upload)
        plugin.context.get_registered_star.return_value = _metadata(target)
        await plugin.generate(event)
        assert order == ["image", "upload"]
        target.assert_awaited_once()
        plugin.context.get_registered_star.assert_called_once_with(
            "astrbot_plugin_moxtu_gallery"
        )
        plugin._send_text.assert_awaited_once_with(
            event, "图片已自动上传到 Moxtu 专属相册。"
        )
        assert plugin._cooldowns and plugin._semaphore._value == 2
        assert list(tmp_path.iterdir()) == []

    asyncio.run(run())


@pytest.mark.parametrize(
    ("result", "text"),
    [
        ({"status": "existing"}, "已保存"),
        ({"status": "skipped", "message": "当前群不允许上传。"}, "当前群不允许上传"),
        ({"status": "failed", "message": "图片超过大小限制。"}, "图片超过大小限制"),
        ({"status": "unknown"}, "保存结果暂无法确认"),
        ({}, "保存结果暂无法确认"),
    ],
)
def test_upload_result_does_not_change_generation_success(tmp_path, result, text):
    plugin = _configured_plugin(tmp_path)
    target = AsyncMock(return_value=result)
    plugin.context.get_registered_star.return_value = _metadata(target)
    asyncio.run(plugin.generate(_event()))
    assert text in plugin._send_text.await_args.args[1]
    plugin._generate.assert_awaited_once()
    assert plugin._cooldowns and plugin._semaphore._value == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("metadata", "text"),
    [
        (None, "未安装或未启用"),
        (types.SimpleNamespace(activated=False, star_cls=None), "未安装或未启用"),
        (types.SimpleNamespace(activated=True, star_cls=object()), "更新 Moxtu"),
    ],
)
def test_missing_disabled_or_old_moxtu_keeps_image_reply(tmp_path, metadata, text):
    plugin = _configured_plugin(tmp_path)
    plugin.context.get_registered_star.return_value = metadata
    asyncio.run(plugin.generate(_event()))
    plugin._send_image.assert_awaited_once()
    assert text in plugin._send_text.await_args.args[1]
    assert plugin._cooldowns


def test_upload_exception_does_not_leak_error_or_retry(tmp_path):
    plugin = _configured_plugin(tmp_path)
    target = AsyncMock(side_effect=RuntimeError("private-token"))
    plugin.context.get_registered_star.return_value = _metadata(target)
    asyncio.run(plugin.generate(_event()))
    assert "private-token" not in plugin._send_text.await_args.args[1]
    assert "保存结果暂无法确认" in plugin._send_text.await_args.args[1]
    target.assert_awaited_once()
    plugin._generate.assert_awaited_once()
    assert plugin._cooldowns


def test_image_send_failure_still_uploads_even_if_error_reply_fails(tmp_path):
    plugin = _configured_plugin(tmp_path)
    plugin._send_image.side_effect = RuntimeError("send failure")
    plugin._send_text.side_effect = OSError("text failure")
    target = AsyncMock(return_value={"status": "success"})
    plugin.context.get_registered_star.return_value = _metadata(target)
    asyncio.run(plugin.generate(_event()))
    target.assert_awaited_once()
    plugin._generate.assert_awaited_once()
    assert not plugin._cooldowns
    assert plugin._semaphore._value == 2
    assert list(tmp_path.iterdir()) == []


def test_generation_failure_does_not_upload(tmp_path):
    plugin = _configured_plugin(tmp_path)
    plugin._generate.side_effect = main_module.ImageProviderError("timeout")
    asyncio.run(plugin.generate(_event()))
    plugin.context.get_registered_star.assert_not_called()
    assert not plugin._cooldowns


def test_lookup_is_refreshed_for_each_request(tmp_path):
    async def run():
        plugin = _configured_plugin(tmp_path)
        plugin.config["cooldown_seconds"] = 0
        first = AsyncMock(return_value={"status": "success"})
        second = AsyncMock(return_value={"status": "success"})
        plugin.context.get_registered_star.side_effect = [
            _metadata(first), _metadata(second)
        ]
        await plugin.generate(_event())
        await plugin.generate(_event())
        first.assert_awaited_once()
        second.assert_awaited_once()

    asyncio.run(run())


def test_cancelled_upload_releases_slot_and_preserves_delivered_cooldown(tmp_path):
    async def run():
        plugin = _configured_plugin(tmp_path)
        entered = asyncio.Event()

        async def upload(*args):
            entered.set()
            await asyncio.Event().wait()

        plugin.context.get_registered_star.return_value = _metadata(upload)
        task = asyncio.create_task(plugin.generate(_event()))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert plugin._cooldowns
        assert plugin._semaphore._value == 2
        assert list(tmp_path.iterdir()) == []

    asyncio.run(run())
