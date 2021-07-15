"""Efficiently use stored in Deta instead of file-like object if available.

Find the `file` cache in DetaBase and DetaDrive and return it. 
If not, preprocesses the data and save it in DetaBase and DetaDrive.
"""
import asyncio
import hashlib
import io
from datetime import datetime
from typing import AsyncGenerator, AsyncIterator, Final, Literal, Optional, Union

from fastapi import UploadFile
from pydantic import BaseModel, validator

from ..config import Settings
from ..preprocessor import Aozora, Wakatu
from . import detacon

ASLEEP_TIME: Final[float] = 0.001
DEFAULT_CHUNK_SIZE: Final[int] = 4096


class DetaBaseItem(BaseModel):
    key: str
    ts: str
    original_fname: str
    path_on_drive: str

    @validator("key")
    def sha1_hash_like(cls, v):
        """Hash value based on raw data."""
        if len(v) != 40:
            raise ValueError("must length 40")
        if not v.isalnum():
            raise ValueError("must be alphanumeric")
        return v

    @validator("ts", pre=True, always=True)
    def force_set_ts_now(cls, v):
        """Force `%FT%T` format timestamp. `v` is ignored always."""
        return datetime.now().strftime("%FT%T")

    @validator("path_on_drive")
    def safe_path(cls, v):
        if not v:
            raise ValueError("must truthy")

        forbidden = "\\!?;:*~^#$%&\"'|"
        if v in forbidden:
            raise ValueError(f"contained forbidden character(s), like a `{forbidden}`")

        return v


# REVIEW Will be able to use without first-iteration by changing to closure or class?
async def file_hash_generator(
    init_chunk: Optional[bytes] = None,
) -> AsyncGenerator[str, Optional[bytes]]:
    """Calc SHA1 hash with updating.

    Yield updated `hash.hexdigest()` each iteration. Use `send()` to update it.

    Iterate once and `send(INIT_CHUNK)` at first if `init_chunk` is None:
    ```python
    it = file_hash_generator()
    # once, iterate
    next(it)
    # pass the init chunk
    it.send(init_chunk)
    ...
    ```
    """
    sha1 = hashlib.sha1()

    if init_chunk is None:
        # set `init_chunk` by `send()` after just a once iterate
        init_chunk = yield "set `init_chunk` by `send(INIT_CHUNK)`"
    chunk = init_chunk
    while chunk:
        sha1.update(chunk)
        chunk = yield sha1.hexdigest()
        await asyncio.sleep(ASLEEP_TIME)


def get_from_base(key: str) -> Union[DetaBaseItem, None]:
    if record := detacon.DetaController.base_get(key=key):
        return DetaBaseItem(**record)
    return None


async def read_from_drive(
    path: str, chunk_size: Union[int, Literal[1024, 2048, 4096]] = DEFAULT_CHUNK_SIZE
) -> AsyncIterator[bytes]:
    if file := detacon.DetaController.drive_get(path):
        try:
            while chunk := file.read(chunk_size if chunk_size > 0 else None):
                yield chunk
                await asyncio.sleep(ASLEEP_TIME)
        finally:
            file.close()
    else:
        raise ValueError(f"`{path}` was not found on drive")


# Main
async def read_preprocessed_txt_or_deta(
    filename: str,
    file: UploadFile,
    chunk_size: Union[int, Literal[1024, 2048, 4096]] = DEFAULT_CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """
    Search in Base with hash as key, get path, and get preprocessed data from drive.
    If not found, preprocess and register the data, and return the processed data.
    """
    key = ""
    raw_bytes = b""

    gen = file_hash_generator()
    await gen.__anext__()  # Prepare for `send(init_chunk)`

    while chunk := await file.read(chunk_size):
        if isinstance(chunk, str):
            raise TypeError("not bytes, its str")
        key = await gen.asend(chunk)  # hexdigest()
        raw_bytes += chunk

    if item := get_from_base(key):
        async for chunk in read_from_drive(item.path_on_drive, chunk_size):
            yield chunk
    else:
        # NOTE The following are hookable?

        # Because Aozora-Bunko exports files with Shift-JIS.
        raw = str(raw_bytes, "shift-jis")
        cleansed = Aozora.cleansing(raw)
        wakatita = Wakatu.parse_only_nouns_verbs_adjectives(cleansed)
        bwaka = wakatita.encode("utf-8")

        path_on_drive = Settings().deta_drive_txt_prefix + key
        # TODO Backgroud/Parallel
        # Backgroud/Parallel 1
        item = DetaBaseItem(key=key, ts=None, original_fname=filename, path_on_drive=path_on_drive)
        detacon.DetaController.base_put(item)
        # Backgroud/Parallel 2
        detacon.DetaController.drive_put(name=path_on_drive, data=bwaka)

        with io.BytesIO(bwaka) as bio:
            while chunk := bio.read(chunk_size):
                yield chunk
                await asyncio.sleep(ASLEEP_TIME)
