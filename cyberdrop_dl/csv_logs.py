from __future__ import annotations

import asyncio
import csv
import dataclasses
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from cyberdrop_dl import constants
from cyberdrop_dl.filepath import sanitize_filename
from cyberdrop_dl.progress import ProgressHook
from cyberdrop_dl.utils import json

if TYPE_CHECKING:
    import datetime
    from collections.abc import Iterable, Iterator

    import yarl

    from cyberdrop_dl.clients.response import AbstractResponse
    from cyberdrop_dl.config import Config
    from cyberdrop_dl.url_objects import AbsoluteHttpURL


logger = logging.getLogger(__name__)

_CSV_DELIMITER = ","


@dataclasses.dataclass(slots=True, kw_only=True)
class CSVFiles:
    unsupported_urls: Path
    download_errors: Path
    scrape_errors: Path
    last_forum_post: Path
    jsonl_file: Path
    progress_events_file: Path | None = None
    scrape_events_file: Path | None = None

    def __iter__(self) -> Iterator[Path]:
        for value in dataclasses.astuple(self):
            if isinstance(value, Path):
                yield value


@dataclasses.dataclass(slots=True)
class CSVLogsManager:
    files: CSVFiles
    task_group: asyncio.TaskGroup = dataclasses.field(init=False, default_factory=asyncio.TaskGroup)
    _file_locks: dict[Path, asyncio.Lock] = dataclasses.field(
        init=False, default_factory=lambda: defaultdict(asyncio.Lock)
    )
    _has_headers: set[Path] = dataclasses.field(init=False, default_factory=set)
    _ready: bool = dataclasses.field(init=False, default=False)
    _responses_folder: Path = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self._responses_folder = self.files.jsonl_file.parent / "cdl_responses"

    @classmethod
    def from_config(cls, config: Config) -> Self:
        files = config.logs.files
        main_log = files.main
        return cls(
            CSVFiles(
                unsupported_urls=files.unsupported,
                download_errors=files.download_errors,
                scrape_errors=files.scrape_errors,
                jsonl_file=files.jsonl_file,
                last_forum_post=files.last_forum_post,
                progress_events_file=main_log.with_suffix(".progress.jsonl") if config.progress_events else None,
                scrape_events_file=main_log.with_suffix(".scrape.jsonl") if config.scrape_events else None,
            )
        )

    def delete_old_logs(self) -> None:
        if self._ready:
            return
        for path in self.files:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            else:
                logger.warning(f"Deleted conflicting old log file: '{path}'")

        self._ready = True

    async def write_jsonl(self, data: Iterable[dict[str, Any]]) -> None:
        async with self._file_locks[self.files.jsonl_file]:
            await asyncio.to_thread(json.dump_jsonl, data, self.files.jsonl_file)

    def write_progress_event(self, event: dict[str, Any]) -> None:
        if (path := self.files.progress_events_file) is None:
            return
        _ = self.task_group.create_task(self._append_jsonl(path, event))

    def write_scrape_event(self, event: dict[str, Any]) -> None:
        if (path := self.files.scrape_events_file) is None:
            return
        _ = self.task_group.create_task(self._append_jsonl(path, event))

    async def _append_jsonl(self, path: Path, event: dict[str, Any]) -> None:
        async with self._file_locks[path]:
            await asyncio.to_thread(_ensure_parent, path)
            await asyncio.to_thread(json.dump_jsonl, (event,), path)

    def make_progress_writer(
        self,
        *,
        url: str,
        filename: str,
        total: int | None,
        interval: float,
        min_bytes: int,
    ) -> ProgressHook:
        cumulative = 0
        started = False
        last_emit_ts = time.monotonic()
        last_emit_bytes = 0

        def emit_start() -> None:
            nonlocal started
            self.write_progress_event(
                {"event": "start", "ts": time.time(), "url": url, "filename": filename, "total": total},
            )
            started = True

        def advance(amount: int = 1) -> None:
            nonlocal cumulative, last_emit_ts, last_emit_bytes
            if not started:
                emit_start()
            cumulative += amount
            now = time.monotonic()
            if (now - last_emit_ts) >= interval and (cumulative - last_emit_bytes) >= min_bytes:
                self.write_progress_event(
                    {"event": "chunk", "ts": time.time(), "url": url, "bytes": cumulative},
                )
                last_emit_ts = now
                last_emit_bytes = cumulative

        def done() -> None:
            if not started:
                emit_start()
            self.write_progress_event(
                {"event": "finish", "ts": time.time(), "url": url, "bytes": cumulative, "ok": True},
            )

        return ProgressHook(advance, _zero_speed, done)

    async def _write_to_csv(self, file: Path, **row: object) -> None:
        async with self._file_locks[file]:
            is_first_write = file not in self._has_headers
            self._has_headers.add(file)

            await asyncio.to_thread(_write_to_csv, file, row, write_headers=is_first_write)

    def write_unsupported(self, url: AbsoluteHttpURL, origin: yarl.URL | Path | None = None) -> None:
        _ = self.task_group.create_task(self._write_to_csv(self.files.unsupported_urls, url=url, origin=origin))

    def write_last_forum_post(self, url: AbsoluteHttpURL) -> None:
        _ = self.task_group.create_task(self._write_to_csv(self.files.last_forum_post, url=url))

    def write_download_error(
        self,
        url: AbsoluteHttpURL,
        referer: AbsoluteHttpURL,
        error_message: str,
        origin: yarl.URL | Path | None = None,
    ) -> None:
        _ = self.task_group.create_task(
            self._write_to_csv(
                self.files.download_errors,
                url=url,
                error=error_message,
                referer=referer,
                origin=origin,
            )
        )

    def write_scrape_error(
        self,
        url: yarl.URL | str,
        error_message: str,
        origin: yarl.URL | Path | None = None,
    ) -> None:
        _ = self.task_group.create_task(
            self._write_to_csv(
                self.files.scrape_errors,
                url=url,
                error=error_message,
                origin=origin,
            )
        )

    def write_response(
        self,
        url: AbsoluteHttpURL,
        response: AbstractResponse[Any],
        exc: Exception | None = None,
    ) -> None:
        _ = self.task_group.create_task(
            asyncio.to_thread(
                _write_resp_to_disk,
                self._responses_folder,
                url,
                response,
                exc,
            )
        )


def _write_resp_to_disk(
    folder: Path,
    url: AbsoluteHttpURL,
    response: AbstractResponse[Any],
    exc: Exception | None = None,
) -> None:
    ext = ".json" if "json" in response.content_type else ".html"
    file = _prepare_resp_file(folder, url, response.created_at, ext)
    try:
        _ = file.write_text(response.create_report(exc), "utf8")
    except OSError as e:
        logger.warning(f"Unable to write response from {url} to disk ({e!r})")
    else:
        logger.debug(f"Saved response from {url} to '{file}'")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _zero_speed() -> float:
    return 0.0


def _prepare_resp_file(folder: Path, url: AbsoluteHttpURL, created_at: datetime.datetime, ext: str = ".html") -> Path:
    max_stem_len = 245 - len(str(folder)) + len(constants.STARTUP_TIME_STR) + 10
    log_date = created_at.strftime(constants.LOGS_DATETIME_FORMAT)
    path_safe_url = sanitize_filename(Path(str(url)).as_posix().replace("/", "-"))
    filename = f"{path_safe_url[:max_stem_len]}_{log_date}{ext}"
    return folder / filename


def _write_to_csv(file: Path, row: dict[str, object], *, write_headers: bool) -> None:
    if write_headers:
        file.parent.mkdir(parents=True, exist_ok=True)

    with file.open("a", encoding="utf8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=tuple(row),
            delimiter=_CSV_DELIMITER,
            quoting=csv.QUOTE_ALL,
        )
        if write_headers:
            writer.writeheader()
        writer.writerow(row)
