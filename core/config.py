from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, get_type_hints

from astrbot.api import logger
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path


class ConfigNode:
    _SCHEMA_CACHE: dict[type, dict[str, type]] = {}
    _FIELDS_CACHE: dict[type, set[str]] = {}

    @classmethod
    def _schema(cls) -> dict[str, type]:
        return cls._SCHEMA_CACHE.setdefault(cls, get_type_hints(cls))

    @classmethod
    def _fields(cls) -> set[str]:
        return cls._FIELDS_CACHE.setdefault(
            cls,
            {key for key in cls._schema() if not key.startswith("_")},
        )

    def __init__(self, data: MutableMapping[str, Any]):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_children", {})
        for key in self._fields():
            if key not in data:
                logger.warning(
                    f"[music_cover:{self.__class__.__name__}] miss key: {key}"
                )

    def __getattr__(self, key: str) -> Any:
        if key not in self._fields():
            raise AttributeError(key)

        value = self._data.get(key)
        field_type = self._schema()[key]
        if isinstance(field_type, type) and issubclass(field_type, ConfigNode):
            children: dict[str, ConfigNode] = self.__dict__["_children"]
            if key not in children:
                if not isinstance(value, MutableMapping):
                    raise TypeError(
                        f"[music_cover:{self.__class__.__name__}] "
                        f"key {key} requires a mapping, got {type(value).__name__}"
                    )
                children[key] = field_type(value)
            return children[key]
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        if key in self._fields():
            self._data[key] = value
            return
        object.__setattr__(self, key, value)

    def raw_data(self) -> Mapping[str, Any]:
        return MappingProxyType(self._data)

    def save_config(self) -> None:
        if not isinstance(self._data, AstrBotConfig):
            raise RuntimeError(
                f"{self.__class__.__name__}.save_config() requires AstrBotConfig"
            )
        self._data.save_config()

class BackendConfig(ConfigNode):
    base_url: str
    models: list[str]
    separator_models: list[str]


class SearchConfig(ConfigNode):
    song_limit: int
    timeout: int


class InferenceConfig(ConfigNode):
    timeout: int
    separator_model: str
    default_model: str
    default_mode: str
    chorus_duration: int
    vocal_volume: float
    accompaniment_volume: float


class PluginConfig(ConfigNode):
    backend_config: BackendConfig
    search_config: SearchConfig
    inference_config: InferenceConfig

    _plugin_name = "astrbot_plugin_music_cover"

    def __init__(self, data: MutableMapping[str, Any]):
        super().__init__(data)
        self.temp_dir = Path(get_astrbot_temp_path()) / self._plugin_name
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def song_limit(self) -> int:
        """Return the bounded search result limit."""
        return max(1, min(20, int(self.search_config.song_limit)))

    @property
    def selection_timeout(self) -> int:
        """Return the bounded interactive selection timeout."""
        return max(10, min(300, int(self.search_config.timeout)))

    @property
    def inference_timeout(self) -> int:
        """Return the bounded backend inference timeout."""
        return max(30, min(1800, int(self.inference_config.timeout)))

    @property
    def chorus_duration(self) -> int:
        """Return the bounded chorus clip duration."""
        return max(15, min(90, int(self.inference_config.chorus_duration)))

    @property
    def default_cover_mode(self) -> str:
        """Return the default interactive cover range."""
        mode = str(self.inference_config.default_mode).strip()
        if mode not in {"高潮", "全曲"}:
            raise ValueError(f"Unsupported default cover mode: {mode}")
        return mode

    @property
    def default_model(self) -> str:
        """Return the configured raw default voice model name."""
        return str(self.inference_config.default_model or "").strip()

    @property
    def vocal_volume(self) -> float:
        """Return the bounded vocal volume adjustment."""
        return max(-3.0, min(3.0, float(self.inference_config.vocal_volume)))

    @property
    def accompaniment_volume(self) -> float:
        """Return the bounded accompaniment volume adjustment."""
        return max(
            -3.0,
            min(3.0, float(self.inference_config.accompaniment_volume)),
        )
