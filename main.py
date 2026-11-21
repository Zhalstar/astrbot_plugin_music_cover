import traceback
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.star.star import StarMetadata
from astrbot.core.utils.session_waiter import SessionController, session_waiter

from .core.config import PluginConfig
from .core.service import CoverClip, CoverResult, MusicCoverService

REVERB_FLAGS = {"大混响", "rv", "reverb", "bigreverb"}
SELECTION_MODES = {"高潮": True, "全曲": False}


class MusicCoverPlugin(Star):
    """Expose music cover business operations as AstrBot commands."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.cfg = PluginConfig(config)
        self.service = MusicCoverService(self.cfg)

    async def initialize(self):
        """Initialize available dependencies without requiring RVC to be online."""
        try:
            await self.service.initialize()
        except Exception as exc:
            logger.warning(
                f"Music cover loaded without RVC WebUI: {type(exc).__name__}: {exc}"
            )
        metadata = self.context.get_registered_star("astrbot_plugin_music")
        if metadata is not None:
            await self.on_plugin_loaded(metadata)
        else:
            logger.warning("Music cover is waiting for astrbot_plugin_music to load")

    @filter.on_plugin_loaded()
    async def on_plugin_loaded(self, metadata: StarMetadata) -> None:
        if metadata.name != "astrbot_plugin_music":
            return
        music_api = getattr(metadata.star_cls, "music_api", None)
        if music_api is None or getattr(music_api, "version", 0) < 2:
            logger.error("Music cover requires astrbot_plugin_music with MusicApi v2")
            return
        self.service.attach_music(music_api)

    @filter.on_plugin_unloaded()
    async def on_plugin_unloaded(self, metadata: StarMetadata) -> None:
        if metadata.name == "astrbot_plugin_music":
            self.service.detach_music()

    async def terminate(self):
        await self.service.close()

    async def _wait_for_selection(
        self,
        event: AstrMessageEvent,
        song_count: int,
        model_count: int,
    ) -> tuple[int, int, bool, int, bool] | None:
        """Wait for one reply containing the complete cover selection.

        Args:
            event: Command event that starts the selection session.
            song_count: Number of displayed song choices.
            model_count: Number of displayed voice model choices.

        Returns:
            Zero-based song and model indexes plus range, pitch, and reverb
            options, or ``None`` when the session times out.
        """
        sender_id = event.get_sender_id()
        selection = None

        @session_waiter(timeout=self.cfg.selection_timeout)
        async def selection_waiter(
            controller: SessionController, reply_event: AstrMessageEvent
        ) -> None:
            nonlocal selection
            if reply_event.get_sender_id() != sender_id:
                return
            values = reply_event.message_str.strip().split()
            if (
                not values
                or not values[0].isdigit()
                or not 1 <= int(values[0]) <= song_count
            ):
                return

            model_index = 0
            option_start = 1
            if len(values) > 1 and values[1].isdigit():
                if not 1 <= int(values[1]) <= model_count:
                    return
                model_index = int(values[1]) - 1
                option_start = 2

            chorus_only = None
            key_shift = None
            big_reverb = False
            for value in values[option_start:]:
                flag = value.casefold().lstrip("-")
                if value in SELECTION_MODES and chorus_only is None:
                    chorus_only = SELECTION_MODES[value]
                elif value.lstrip("+-").isdigit() and key_shift is None:
                    candidate = int(value)
                    if not -12 <= candidate <= 12:
                        return
                    key_shift = candidate
                elif flag in REVERB_FLAGS and not big_reverb:
                    big_reverb = True
                else:
                    return

            selection = (
                int(values[0]) - 1,
                model_index,
                SELECTION_MODES[self.cfg.default_cover_mode]
                if chorus_only is None
                else chorus_only,
                key_shift or 0,
                big_reverb,
            )
            controller.stop()

        try:
            await selection_waiter(event)
        except TimeoutError:
            return None
        return selection

    @filter.command("翻唱")
    async def cover(self, event: AstrMessageEvent):
        """翻唱 <歌名>"""
        query = event.message_str.removeprefix("翻唱").strip()
        if not query:
            yield event.plain_result("用法: /翻唱 <歌名>")
            return

        try:
            songs = await self.service.search_songs(query)
        except Exception as exc:
            logger.error(traceback.format_exc())
            yield event.plain_result(str(exc))
            return
        if not songs:
            yield event.plain_result("没有找到匹配的歌曲")
            return

        models = self.service.models()
        if not models:
            yield event.plain_result("当前后端没有可用的翻唱模型")
            return

        song_lines = []
        for index, result in enumerate(songs, start=1):
            song = result["song"]
            song_lines.append(
                f"{index}. {song.get('name') or '未知歌曲'} - "
                f"{song.get('artists') or '未知歌手'}"
            )
        model_lines = [
            f"{index}. {model.display_name}"
            for index, model in enumerate(models, start=1)
        ]
        msg = "\n".join(
            [
                "●命令示例：1 2 全曲 -2 大混响",
                "\n【歌曲】",
                "\n".join(song_lines),
                "\n【音色】",
                "\n".join(model_lines),
            ]
        )
        await self.service.send_selection(event, msg)
        selection = await self._wait_for_selection(event, len(songs), len(models))
        if selection is None:
            yield event.plain_result("选择超时，操作已取消")
            return

        song_index, model_index, chorus_only, key_shift, big_reverb = selection
        selected_model = models[model_index]
        clip = None
        if chorus_only:
            clip = await self.service.resolve_chorus_clip(songs[song_index])

        sent, message = await self._send_cover(
            event,
            songs[song_index],
            selected_model.name,
            key_shift,
            big_reverb,
            clip,
        )
        if not sent:
            yield event.plain_result(message)
        event.stop_event()

    @filter.llm_tool()
    async def cover_song_by_name(
        self,
        event: AstrMessageEvent,
        song_name: str,
        model_name: str = "",
        cover_mode: str = "",
        key_shift: int = 0,
        big_reverb: bool = False,
    ) -> str:
        """当用户想听 AI 翻唱时，根据歌名搜索歌曲并生成翻唱。

        Args:
            song_name(string): 歌曲名称或包含歌手的搜索关键词。
            model_name(string): 可选的翻唱音色名称；留空时使用第一个可用音色。
            cover_mode(string): 可选的翻唱范围，支持“高潮”或“全曲”；留空时使用插件配置。
            key_shift(int): 可选的升降调，范围 -12 到 12，0 表示不调整。
            big_reverb(bool): 是否启用大混响，默认 False。

        Returns:
            翻唱生成和发送结果。
        """
        query = song_name.strip()
        if not query:
            return "未提供要翻唱的歌曲名称"
        if not -12 <= key_shift <= 12:
            return "升降调必须在 -12 到 12 之间"

        mode = cover_mode.strip() or self.cfg.default_cover_mode
        if mode not in SELECTION_MODES:
            return "翻唱范围仅支持“高潮”或“全曲”"

        models = self.service.models()
        if not models:
            return "当前后端没有可用的翻唱模型"
        requested_model = model_name.strip().casefold()
        selected_model = models[0]
        if requested_model:
            selected_model = next(
                (
                    model
                    for model in models
                    if requested_model
                    in {
                        model.name.casefold(),
                        model.display_name.casefold(),
                        Path(model.name).name.casefold(),
                    }
                ),
                None,
            )
            if selected_model is None:
                available = "、".join(model.display_name for model in models)
                return f"未找到翻唱音色“{model_name}”；可用音色：{available}"

        try:
            songs = await self.service.search_songs(query, limit=1)
        except Exception as exc:
            logger.error(traceback.format_exc())
            return str(exc)
        if not songs:
            return f"没有找到匹配的歌曲：{query}"

        clip = None
        if SELECTION_MODES[mode]:
            clip = await self.service.resolve_chorus_clip(songs[0])
        _, message = await self._send_cover(
            event,
            songs[0],
            selected_model.name,
            key_shift,
            big_reverb,
            clip,
        )
        return message

    async def _send_cover(
        self,
        event: AstrMessageEvent,
        search_result: dict,
        model_name: str,
        key_shift: int,
        big_reverb: bool,
        clip: CoverClip | None,
    ) -> tuple[bool, str]:
        """Generate a cover through the service and send its audio result.

        Args:
            event: Event used to send result messages.
            search_result: Selected source-owned MusicApi search result.
            model_name: Selected raw voice model name.
            key_shift: Semitone adjustment in the inclusive range -12 to 12.
            big_reverb: Whether to enable the strong reverb preset.
            clip: Optional chorus interval to keep from the generated cover.

        Returns:
            Whether generation and sending succeeded, plus a user-facing result.
        """
        result: CoverResult | None = None
        try:
            result = await self.service.generate_cover(
                search_result,
                model_name,
                key_shift=key_shift,
                big_reverb=big_reverb,
                clip=clip,
            )
            minutes, seconds = divmod(result.elapsed_seconds, 60)
            elapsed = f"{minutes} 分 {seconds:02d} 秒" if minutes else f"{seconds} 秒"
            logger.debug(event.plain_result(f"翻唱完成，总用时：{elapsed}"))
            song = search_result.get("song", {})
            sent = await self.service.send_audio(
                event,
                result.path,
                name=f"{song.get('name') or '翻唱'}（翻唱）",
                artist=Path(model_name).stem,
            )
            if not sent:
                return False, "翻唱已生成，但音频发送失败"
            title = song.get("name") or "未知歌曲"
            artist = song.get("artists") or "未知歌手"
            return True, f"已发送《{title} - {artist}》的翻唱"
        except Exception as exc:
            logger.error(traceback.format_exc())
            if isinstance(exc, TimeoutError) or "timeout" in str(exc).casefold():
                return False, "生成超时，请调高推理超时时长"
            return False, f"生成失败: {exc}"
        finally:
            if result:
                self.service.release_result(result)
