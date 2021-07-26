import io
from typing import Any, Dict, List, Optional, Union

from deta import Deta
from deta.base import FetchResponse
from deta.drive import DriveStreamingBody

from .. import config
from .__init__ import DetaBaseItem


class DetaController:
    """This will use just single instance."""

    _dotenv = config.Dotenv()
    _settings = config.Settings()

    _deta = Deta(project_key=_dotenv.deta_project_key, project_id=_dotenv.deta_project_id)
    base = _deta.Base(_settings.deta_base)
    drive = _deta.Drive(_settings.deta_drive)

    @classmethod
    def base_get(cls, key: str) -> Union[Dict[str, Any], None]:
        return cls.base.get(key=key)

    @classmethod
    def drive_get(cls, fpath: str) -> Union[DriveStreamingBody, None]:
        """
        Return it that must close() after used.
        It does not support the `with` phrase.
        """
        return cls.drive.get(name=fpath)

    @classmethod
    def base_put(cls, item: DetaBaseItem) -> Any:
        return cls.base.put(item.dict())

    @classmethod
    def drive_put(
        cls, name: str, data: Union[str, bytes, io.TextIOBase, io.BufferedIOBase, io.RawIOBase]
    ) -> str:
        return cls.drive.put(name=name, data=data)

    @classmethod
    def base_fetch(
        cls,
        query: Union[Dict[str, Any], List[Dict[str, Any]]],
        limit: Optional[int] = None,
        last: Optional[int] = None,
    ) -> FetchResponse:
        return cls.base.fetch(query=query, limit=limit, last=last)

    # FIXME
    @classmethod
    def base_put_features(cls, key: str, data: Dict):
        base = cls._deta.Base(cls._settings.deta_base_features)
        return base.put(key=key, data=data)
