import asyncio
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from astrbot.api import logger

from .chorus import locate_chorus
from .config import PluginConfig
from .rvc_client import RvcClient


@dataclass(frozen=True, slots=True)
class CoverModel:
    name: str
    display_name: str


@dataclass(frozen=True, slots=True)
class CoverResult:
    path: Path
    elapsed_seconds: int
    workspace: Path


@dataclass(frozen=True, slots=True)
class CoverClip:
    start_seconds: float | None
    duration_seconds: float


class MusicCoverService:
    """Business operations shared by command and Web/API application layers."""

    def __init__(self, config: PluginConfig):
        self.cfg = config
        self._client: RvcClient | None = None
        self._music_api = None
        self._inference_lock = asyncio.Lock()

    @property
    def client(self) -> RvcClient:
        """Return the initialized RVC client.

        Returns:
            The RVC client owned by this service.

        Raises:
            RuntimeError: If the service has not been initialized.
        """
        if self._client is None:
            raise RuntimeError("RVC WebUI 尚未初始化")
        return self._client

    async def initialize(self) -> None:
        """Connect to RVC WebUI and persist its available models."""
        if self._client is not None:
            return
        base_url = str(self.cfg.backend_config.base_url).strip()
        if not base_url:
            raise ValueError("RVC WebUI 地址不能为空")

        client = RvcClient(base_url, self.cfg.inference_timeout)
        try:
            await client.initialize()
            changed = False
            if client.models != self.cfg.backend_config.models:
                self.cfg.backend_config.models = client.models
                changed = True
            if client.separator_models != self.cfg.backend_config.separator_models:
                self.cfg.backend_config.separator_models = client.separator_models
                changed = True

            default_model = self.cfg.default_model
            if default_model and default_model not in client.models:
                logger.warning(
                    f"Configured default voice model is unavailable: {default_model}"
                )

            separator_model = str(self.cfg.inference_config.separator_model).strip()
            if separator_model not in client.separator_models:
                if "HP5_only_main_vocal" not in client.separator_models:
                    raise RuntimeError(
                        f"RVC WebUI 没有配置的人声分离模型: {separator_model}"
                    )
                self.cfg.inference_config.separator_model = "HP5_only_main_vocal"
                changed = True
            if changed:
                self.cfg.save_config()
            self._client = client
        except BaseException:
            await client.close()
            raise

    async def close(self) -> None:
        """Close the RVC client and clear attached music state."""
        self.detach_music()
        if self._client is not None:
            await self._client.close()
            self._client = None

    def attach_music(self, music_api: Any) -> None:
        """Attach the current MusicApi instance.

        Args:
            music_api: Versioned API exposed by the music plugin.
        """
        self._music_api = music_api

    def detach_music(self) -> None:
        """Detach the old MusicApi while the music plugin reloads."""
        self._music_api = None

    async def send_audio(
        self,
        event: Any,
        path: Path,
        *,
        name: str = "",
        artist: str = "",
    ) -> bool:
        """Send generated audio through the music plugin's fallback chain.

        Args:
            event: AstrBot message event receiving the audio.
            path: Generated audio path owned by this service's caller.
            name: Display name used for file messages.
            artist: Optional artist used for file messages.

        Returns:
            Whether the music plugin sent the audio successfully.

        Raises:
            RuntimeError: If the music plugin is unavailable.
        """
        music_api = self._music_api
        if music_api is None:
            raise RuntimeError("点歌插件未加载或正在重载，请稍后再试")
        return await music_api.send_audio(
            event,
            path=path,
            name=name,
            artist=artist,
        )

    async def send_selection(self, event: Any, text: str) -> None:
        """Send a recallable selection through the music plugin.

        Args:
            event: AstrBot message event receiving the selection.
            text: Complete cover selection prompt and candidate list.

        Raises:
            RuntimeError: If the music plugin is unavailable.
        """
        music_api = self._music_api
        if music_api is None:
            raise RuntimeError("点歌插件未加载或正在重载，请稍后再试")
        await music_api.send_selection(event, text)

    def models(self) -> list[CoverModel]:
        """Return voice models refreshed from RVC WebUI."""
        models = [
            CoverModel(name=name, display_name=Path(name).stem)
            for item in self.cfg.backend_config.models
            if (name := str(item).strip())
        ]
        default_model = self.cfg.default_model
        for index, model in enumerate(models):
            if model.name == default_model:
                models.insert(0, models.pop(index))
                break
        return models

    async def search_songs(
        self, query: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Search through the music plugin's configured fallback chain.

        Args:
            query: Song title, artist, or other search text.
            limit: Optional maximum result count.

        Returns:
            Source-owned song results from ``MusicApi.search``.

        Raises:
            RuntimeError: If the music plugin is unavailable or search fails.
        """
        music_api = self._music_api
        if music_api is None:
            raise RuntimeError("点歌插件未加载或正在重载，请稍后再试")
        try:
            return await music_api.search(
                query,
                limit=limit or self.cfg.song_limit,
            )
        except Exception as exc:
            raise RuntimeError(f"歌曲搜索失败: {exc}") from exc

    async def resolve_chorus_clip(self, search_result: dict[str, Any]) -> CoverClip:
        """Resolve the chorus interval for one MusicApi search result.

        Args:
            search_result: Source-owned result returned by :meth:`search_songs`.

        Returns:
            A detected or fallback interval. An unresolved start is calculated
            from the downloaded audio's actual duration during generation.
        """
        music_api = self._music_api
        if music_api is None:
            return CoverClip(None, self.cfg.chorus_duration)

        source = str(search_result.get("source") or "").strip()
        song = search_result.get("song")
        if not source or not isinstance(song, dict):
            raise RuntimeError("歌曲搜索结果缺少音源信息")
        try:
            details = await music_api.track_details(
                source,
                song,
                lyrics_timeout=self.cfg.selection_timeout,
            )
            lyrics = str(details.get("lyrics") or "")
            total = float(details.get("duration") or 0) / 1000 or None
            start, duration = locate_chorus(
                lyrics,
                duration=self.cfg.chorus_duration,
                total=total,
            )
        except Exception as exc:
            logger.warning(
                f"Chorus metadata unavailable, using audio-duration fallback: {exc}"
            )
            return CoverClip(None, self.cfg.chorus_duration)
        return CoverClip(start_seconds=start, duration_seconds=duration)

    async def generate_cover(
        self,
        search_result: dict[str, Any],
        model_name: str,
        *,
        key_shift: int = 0,
        big_reverb: bool = False,
        clip: CoverClip | None = None,
    ) -> CoverResult:
        """Download, separate, convert, and mix one cover.

        Args:
            search_result: Source-owned result returned by :meth:`search_songs`.
            model_name: RVC voice model filename.
            key_shift: Semitone adjustment from -12 to 12.
            big_reverb: Whether to apply the strong reverb preset.
            clip: Optional source interval to process before separation.

        Returns:
            Generated audio path and elapsed processing time.

        Raises:
            ValueError: If request parameters are invalid.
            RuntimeError: If downloading, separation, conversion, or mixing fails.
        """
        source = str(search_result.get("source") or "").strip()
        song = search_result.get("song")
        if not source or not isinstance(song, dict):
            raise ValueError("歌曲搜索结果缺少音源信息")
        if not -12 <= key_shift <= 12:
            raise ValueError("升降调必须在 -12 到 12 之间")
        if model_name not in {model.name for model in self.models()}:
            raise ValueError(f"未配置模型: {model_name}")

        music_api = self._music_api
        if music_api is None:
            raise RuntimeError("点歌插件未加载或正在重载，请稍后再试")
        started_at = time.perf_counter()
        workspace = Path(tempfile.mkdtemp(prefix="cover-", dir=self.cfg.temp_dir))
        source_path: Path | None = None
        converted_path: Path | None = None
        succeeded = False
        try:
            source_path = await music_api.download_audio(
                source,
                song,
                resolve_timeout=min(60, self.cfg.inference_timeout),
                download_timeout=min(180, self.cfg.inference_timeout),
            )
            if source_path is None or not Path(source_path).is_file():
                raise RuntimeError("点歌插件未能下载所选歌曲")
            source_path = Path(source_path)

            input_dir = workspace / "input"
            vocal_dir = workspace / "vocal"
            accompaniment_dir = workspace / "accompaniment"
            input_dir.mkdir()
            vocal_dir.mkdir()
            accompaniment_dir.mkdir()
            input_path = input_dir / "source.wav"
            audio = await asyncio.to_thread(AudioSegment.from_file, source_path)
            if clip is not None:
                requested_ms = min(round(clip.duration_seconds * 1000), len(audio))
                if clip.start_seconds is None:
                    start_ms = min(round(len(audio) * 0.55), len(audio) - requested_ms)
                else:
                    start_ms = round(clip.start_seconds * 1000)
                    if start_ms >= len(audio):
                        start_ms = min(
                            round(len(audio) * 0.55),
                            len(audio) - requested_ms,
                        )
                end_ms = min(
                    len(audio),
                    start_ms + requested_ms,
                )
                if start_ms >= end_ms:
                    raise RuntimeError("识别到的高潮区间超出歌曲时长")
                audio = audio[start_ms:end_ms]
                audio = audio.fade_in(min(100, len(audio))).fade_out(
                    min(800, len(audio))
                )
            audio = audio.set_channels(2).set_frame_rate(44100).set_sample_width(2)
            exported = await asyncio.to_thread(audio.export, input_path, format="wav")
            exported.close()
            del audio
            music_api.release_audio(source_path)
            source_path = None

            async with self._inference_lock:
                separation_result = await self.client.call(
                    "/uvr_convert",
                    self.cfg.inference_config.separator_model,
                    str(input_dir.resolve()),
                    str(vocal_dir.resolve()),
                    None,
                    str(accompaniment_dir.resolve()),
                    10,
                    "wav",
                )
                vocals = list(vocal_dir.glob("*.wav"))
                accompaniments = list(accompaniment_dir.glob("*.wav"))
                if not vocals or not accompaniments:
                    raise RuntimeError(f"人声分离失败: {separation_result}")

                model_result = await self.client.call(
                    "/infer_change_voice",
                    model_name,
                    0.33,
                    0.33,
                )
                index_update = (
                    model_result[3]
                    if isinstance(model_result, tuple) and len(model_result) > 3
                    else None
                )
                index_path = (
                    str(index_update.get("value") or "")
                    if isinstance(index_update, dict)
                    else ""
                )
                conversion_result = await self.client.call(
                    "/infer_convert",
                    0,
                    str(vocals[0].resolve()),
                    key_shift,
                    None,
                    "rmvpe",
                    "",
                    index_path,
                    0.75,
                    3,
                    0,
                    0.25,
                    0.33,
                )

            if not isinstance(conversion_result, tuple) or len(conversion_result) != 2:
                raise RuntimeError(f"RVC 返回了无效结果: {conversion_result}")
            message, audio_result = conversion_result
            converted_path = (
                Path(str(audio_result.get("name")))
                if isinstance(audio_result, dict) and audio_result.get("name")
                else None
            )
            if not converted_path or not converted_path.is_file():
                raise RuntimeError(f"RVC 转换失败: {message}")

            vocal_audio, accompaniment_audio = await asyncio.gather(
                asyncio.to_thread(AudioSegment.from_file, converted_path),
                asyncio.to_thread(AudioSegment.from_file, accompaniments[0]),
            )
            if key_shift:
                shifted_path = workspace / "accompaniment-shifted.wav"
                ratio = 2 ** (key_shift / 12)
                sample_rate = accompaniment_audio.frame_rate
                process = await asyncio.to_thread(
                    subprocess.run,
                    [
                        AudioSegment.converter,
                        "-y",
                        "-i",
                        str(accompaniments[0]),
                        "-filter:a",
                        (
                            f"asetrate={sample_rate * ratio:.6f},"
                            f"aresample={sample_rate},atempo={1 / ratio:.6f}"
                        ),
                        str(shifted_path),
                    ],
                    capture_output=True,
                    check=False,
                )
                if process.returncode != 0 or not shifted_path.is_file():
                    error = process.stderr.decode(errors="replace").strip()
                    raise RuntimeError(f"伴奏变调失败: {error[-500:]}")
                accompaniment_audio = await asyncio.to_thread(
                    AudioSegment.from_file, shifted_path
                )

            vocal_audio += self.cfg.vocal_volume
            accompaniment_audio += self.cfg.accompaniment_volume
            if big_reverb:
                wet_vocal = AudioSegment.silent(
                    duration=len(vocal_audio) + 500,
                    frame_rate=vocal_audio.frame_rate,
                )
                for delay, gain in ((0, 0), (90, -7), (180, -11), (330, -15)):
                    wet_vocal = wet_vocal.overlay(vocal_audio + gain, position=delay)
                vocal_audio = wet_vocal[: len(accompaniment_audio)]

            mixed = accompaniment_audio.overlay(vocal_audio)
            output_path = workspace / "cover.mp3"
            exported = await asyncio.to_thread(
                mixed.export,
                output_path,
                format="mp3",
                bitrate="192k",
            )
            exported.close()
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("混音后没有生成有效音频")

            for directory in (input_dir, vocal_dir, accompaniment_dir):
                await asyncio.to_thread(shutil.rmtree, directory, True)
            (workspace / "accompaniment-shifted.wav").unlink(missing_ok=True)
            succeeded = True
            return CoverResult(
                path=output_path,
                elapsed_seconds=max(0, round(time.perf_counter() - started_at)),
                workspace=workspace,
            )
        finally:
            if source_path is not None:
                music_api.release_audio(source_path)
            if converted_path is not None:
                converted_path.unlink(missing_ok=True)
            if not succeeded:
                await asyncio.to_thread(shutil.rmtree, workspace, True)

    @staticmethod
    def release_result(result: CoverResult) -> None:
        """Delete generated audio and its task workspace.

        Args:
            result: Cover result returned by :meth:`generate_cover`.
        """
        shutil.rmtree(result.workspace, ignore_errors=True)
