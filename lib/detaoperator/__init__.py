"""Efficiently use stored in Deta instead of file-like object if available.

Find the `file` cache in DetaBase and DetaDrive and return it. 
If not, preprocesses the data and save it in DetaBase and DetaDrive.
"""
import asyncio
import hashlib
import io
from datetime import datetime
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Final, List, Literal, Optional, Union

from fastapi import UploadFile
from pydantic import BaseModel, validator

from .. import imgps
from ..config import Settings
from ..preprocessor import Aozora, ImgNormalizer, Wakatu
from . import detacon

ASLEEP_SECOND: Final[float] = 0.001
DEFAULT_CHUNK_SIZE: Final[int] = 4096


class DetaBaseItem(BaseModel):
    key: str
    ts: str
    original_fname: str
    preprocessed_path: str
    other_files: Optional[Dict[str, str]] = None
    other_data: Optional[Dict[str, Any]] = None

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

    @validator("preprocessed_path")
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
    await it.__anext__()
    # pass the init chunk
    await it.asend(init_chunk)
    ...
    ```
    """
    sha1 = hashlib.sha1()

    if init_chunk is None:
        # set `init_chunk` by `asend()` after just a once iterate
        init_chunk = yield "set `init_chunk` by `send(INIT_CHUNK)`"
    chunk = init_chunk
    while chunk:
        sha1.update(chunk)
        chunk = yield sha1.hexdigest()  # FIXME Overhead; called by each iteration.
        await asyncio.sleep(ASLEEP_SECOND)


def get_from_base(key: str) -> Union[DetaBaseItem, None]:
    if record := detacon.DetaController.base_get(key=key):
        return DetaBaseItem(**record)
    return None


def query_base(
    query: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Union[List[DetaBaseItem], None]:
    res = detacon.DetaController.base_fetch(query)

    if not res.items:
        return None

    resi: List[Dict[str, Any]] = res.items
    while res.last:  # turn over the pages
        resi += detacon.DetaController.base.fetch(last=res.last).items

    return [DetaBaseItem(**i) for i in resi]


def query_one_base(
    query: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Union[DetaBaseItem, None]:
    res = detacon.DetaController.base_fetch(query, limit=1)

    if not res.items:
        return None

    return DetaBaseItem(**res.items[0])


def ls_drive(prefix: Optional[str] = None) -> Union[List[str], None]:
    res = detacon.DetaController.list_drive(prefix)

    if not isinstance(res, dict):
        return None

    all_files: List[str] = res["names"]
    if res.get("paging") is None:
        return all_files
    elif last := res["paging"].get("last") is not None:
        while last:  # turn over the pages
            res = detacon.DetaController.list_drive(prefix, last=last)
            if not isinstance(res, dict):
                break
            all_files += res["names"]

            if res.get("paging") is None:
                return all_files
            last = res["paging"].get("last")

    return all_files


async def read_from_drive(
    path: str, chunk_size: Union[int, Literal[1024, 2048, 4096]] = DEFAULT_CHUNK_SIZE
) -> AsyncIterator[bytes]:
    if file := detacon.DetaController.drive_get(path):
        try:
            while chunk := file.read(chunk_size if chunk_size > 0 else None):
                yield chunk
                await asyncio.sleep(ASLEEP_SECOND)
        finally:
            file.close()
    else:
        raise ValueError(f"`{path}` was not found on drive")


# Main
async def read_preprocessed_or_from_deta(
    file: UploadFile,
    file_type: Literal["text", "image"],
    chunk_size: Union[int, Literal[1024, 2048, 4096]] = DEFAULT_CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """
    Search in Base with hash as key, get path, and get preprocessed data from drive.
    If not found, preprocess and register the data, and return the processed data.
    """
    key = ""
    raw_bytes = b""

    gen = file_hash_generator()
    await gen.__anext__()  # Prepare for `asend(init_chunk)`

    while chunk := await file.read(chunk_size):
        if isinstance(chunk, str):
            raise TypeError("not bytes, its str")
        key = await gen.asend(chunk)  # hexdigest()
        raw_bytes += chunk

    if item := get_from_base(key):
        async for chunk in read_from_drive(item.preprocessed_path, chunk_size):
            yield chunk
    else:
        other_data = None

        # NOTE The following are hookable?
        if file_type == "text":
            # Because Aozora-Bunko exports files with Shift-JIS.
            raw = raw_bytes.decode("shift-jis")
            parsed = Wakatu.parse_only_nouns_verbs_adjectives(Aozora.cleansing(raw))
            preprocessed_bytes = parsed.encode("shift-jis")

            preprocessed_path = Settings().deta_drive_txt_prefix + key

            other_files = {"raw": "raw" + preprocessed_path}
        elif file_type == "image":
            preprocessed_bytes = ImgNormalizer.resize_and_equalize_hist(raw_bytes)
            preprocessed_path = Settings().deta_drive_img_prefix + key

            (kp, desc), kpimg = imgps.akaze(preprocessed_bytes, draw=True)

            kp = [
                {"response": k.response, "pt": k.pt, "size": k.size, "angle": k.angle} for k in kp
            ]
            features = [
                {"kp": k, "desc": d}
                for k, d in zip(
                    sorted(kp, key=lambda x: x["response"], reverse=True), desc.tolist()
                )
            ]

            other_files = {"raw": "raw" + preprocessed_path, "akaze": "akaze" + preprocessed_path}
            other_data = {"akaze_features": features}
            if kpimg is None:
                raise ValueError("`kpimg` is `None`, akaze drawing failed?")
            detacon.DetaController.drive_put(name=other_files["akaze"], data=kpimg)

        # TODO Backgroud/Parallel
        # Backgroud/Parallel 1
        if other_data:
            detacon.DetaController.base_put(
                DetaBaseItem(
                    key=key,
                    ts=None,
                    original_fname=file.filename,
                    preprocessed_path=preprocessed_path,
                    other_files=other_files,
                    other_data=other_data,
                )
            )
        else:
            detacon.DetaController.base_put(
                DetaBaseItem(
                    key=key,
                    ts=None,
                    original_fname=file.filename,
                    preprocessed_path=preprocessed_path,
                    other_files=other_files,
                )
            )

        # Backgroud/Parallel 2
        detacon.DetaController.drive_put(name=preprocessed_path, data=preprocessed_bytes)
        # Backgroud/Parallel 3
        detacon.DetaController.drive_put(name=other_files["raw"], data=raw_bytes)

        with io.BytesIO(preprocessed_bytes) as bio:
            while chunk := bio.read(chunk_size):
                yield chunk
                await asyncio.sleep(ASLEEP_SECOND)


def remove_from_base(query: Union[Dict[str, Any], List[Dict[str, Any]]]):
    if item := query_one_base(query):
        detacon.DetaController.base_del(item.key)
    return None


def remove_files(files: List[str]) -> Union[List[str], None]:
    def removeprefix(s: str, p: str, /) -> str:
        if s[: len(p)] == p:
            return s[len(p) :]
        return s

    additional = []
    for file in files:
        if (removed := removeprefix(file, "img/")) != file:
            additional += ["rawimg/" + removed, "akazeimg/" + removed]
        elif (removed := removeprefix(file, "akazeimg/")) != file:
            additional += ["rawimg/" + removed, "img/" + removed]
        elif (removed := removeprefix(file, "rawimg/")) != file:
            additional += ["img/" + removed, "akazeimg/" + removed]
        elif (removed := removeprefix(file, "rawtxt/")) != file:
            additional += ["txt/" + removed]
        elif (removed := removeprefix(file, "txt/")) != file:
            additional += ["rawtxt/" + removed]

    res = detacon.DetaController.drive_delmany(files + additional)

    if res is None:
        return files + additional

    [remove_from_base({"preprocessed_path": name}) for name in res["deleted"]]

    if failefiles := res.get("failed"):
        return failefiles

    return None
