"""Locate a likely chorus from repeated lines in an LRC timeline."""

from __future__ import annotations

import re

_TIMESTAMP_RE = re.compile(r"^\[(\d+):(\d+)(?:[.:](\d{1,3}))?\]")
_OFFSET_RE = re.compile(r"\[offset:\s*([+-]?\d+)\]", re.IGNORECASE)
_METADATA_RE = re.compile(
    r"^(作词|作曲|编曲|制作人|监制|混音|录音|吉他|贝斯|鼓|键盘|弦乐|和声|出品|"
    r"发行|统筹|企划|op|sp|lyrics?|compos|arrang|produc|mix|master)"
)


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text).casefold()


def _parse_lrc(lyrics: str) -> list[tuple[float, str]]:
    """Parse timestamped LRC lines and apply the optional global offset.

    Args:
        lyrics: Raw LRC text.

    Returns:
        Timestamp and lyric pairs sorted by playback position.
    """
    offset_match = _OFFSET_RE.search(lyrics)
    offset = int(offset_match.group(1)) / 1000 if offset_match else 0.0
    rows: list[tuple[float, str]] = []

    for raw_line in lyrics.splitlines():
        remaining = raw_line.strip()
        timestamps = []
        while match := _TIMESTAMP_RE.match(remaining):
            fraction = match.group(3) or ""
            seconds = int(match.group(1)) * 60 + int(match.group(2))
            if fraction:
                seconds += int(fraction) / (10 ** len(fraction))
            timestamps.append(max(0.0, seconds + offset))
            remaining = remaining[match.end() :]
        rows.extend((timestamp, remaining.strip()) for timestamp in timestamps)

    rows.sort(key=lambda row: row[0])
    unique_rows = []
    seen_timestamps = set()
    for timestamp, text in rows:
        key = round(timestamp, 2)
        if key not in seen_timestamps:
            seen_timestamps.add(key)
            unique_rows.append((timestamp, text))
    return unique_rows


def locate_chorus(
    lyrics: str,
    *,
    duration: float,
    total: float | None = None,
) -> tuple[float | None, float]:
    """Locate a chorus or choose a middle-late fallback interval.

    Args:
        lyrics: Timestamped LRC text.
        duration: Requested chorus clip length in seconds.
        total: Optional song duration in seconds.

    Returns:
        Start time and clip duration in seconds. The start is ``None`` when the
        caller must calculate it from the downloaded audio's actual duration.
    """
    if duration <= 0:
        raise ValueError("Clip duration must be positive")
    rows = [
        (timestamp, text)
        for timestamp, text in _parse_lrc(lyrics)
        if (normalized := _normalize(text)) and not _METADATA_RE.match(normalized)
    ]
    last_timestamp = rows[-1][0] if rows else 0.0
    if total and total > 0:
        if rows and (total - last_timestamp > 60 or last_timestamp - total > 3):
            rows = []
        song_end = total
    elif rows:
        song_end = last_timestamp + 6
    else:
        return None, duration

    requested_duration = min(duration, song_end)
    if song_end <= 8:
        return 0.0, round(song_end, 1)
    window = min(30.0, max(15.0, duration))
    scored = []
    if len(rows) >= 8 and song_end >= window:
        counts: dict[str, int] = {}
        for _, text in rows:
            normalized = _normalize(text)
            counts[normalized] = counts.get(normalized, 0) + 1

        items = []
        for index, (start, text) in enumerate(rows):
            end = rows[index + 1][0] if index + 1 < len(rows) else song_end
            items.append(
                (
                    start,
                    start + max(0.6, min(8.0, end - start)),
                    counts[_normalize(text)] - 1,
                )
            )

        for start in sorted(
            {
                round(timestamp, 2)
                for timestamp, _ in rows
                if timestamp + window <= song_end
            }
        ):
            score = 0.0
            repeated_lines = 0
            for line_start, line_end, weight in items:
                overlap = min(line_end, start + window) - max(line_start, start)
                if weight > 0 and overlap > 0:
                    score += overlap * weight
                    repeated_lines += 1
            if repeated_lines >= 3:
                scored.append((score / window, start))

    best_score = max((score for score, _ in scored), default=0.0)
    if best_score >= 0.8:
        detected_start = next(
            start for score, start in scored if score >= best_score * 0.9
        )
        previous_line = max(
            (timestamp for timestamp, _ in rows if timestamp < detected_start),
            default=0.0,
        )
        chorus_start = max(previous_line, detected_start - 1.0, 0.0)
    elif rows:
        latest_start = max(0.0, song_end - requested_duration)
        target_start = min(song_end * 0.55, latest_start)
        line_start = min(
            (timestamp for timestamp, _ in rows if timestamp <= latest_start),
            key=lambda timestamp: abs(timestamp - target_start),
            default=target_start,
        )
        chorus_start = max(0.0, line_start - 1.0)
    else:
        chorus_start = max(
            0.0,
            min(song_end * 0.55, song_end - requested_duration),
        )

    target_end = min(chorus_start + duration, song_end)
    if rows:
        boundaries = [
            timestamp
            for timestamp, _ in rows
            if chorus_start + 8 <= timestamp <= song_end
        ]
        boundaries.append(song_end)
        boundaries.sort()
        later_boundary = next(
            (
                timestamp
                for timestamp in boundaries
                if target_end <= timestamp <= target_end + 6
            ),
            None,
        )
        if later_boundary is not None:
            chorus_end = later_boundary
        else:
            chorus_end = max(
                (timestamp for timestamp in boundaries if timestamp <= target_end),
                default=target_end,
            )
    else:
        chorus_end = target_end
    clip_duration = chorus_end - chorus_start
    if clip_duration < 8:
        chorus_start = max(
            0.0,
            min(song_end * 0.55, song_end - requested_duration),
        )
        clip_duration = min(requested_duration, song_end - chorus_start)
    return round(chorus_start, 1), round(clip_duration, 1)
