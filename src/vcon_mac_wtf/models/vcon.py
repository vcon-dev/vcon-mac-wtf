"""Pydantic models for vCon documents."""

from typing import Any

from pydantic import BaseModel


class Party(BaseModel, extra="allow"):
    name: str | None = None
    mailto: str | None = None
    tel: str | None = None


class Dialog(BaseModel, extra="allow"):
    type: str
    start: str | None = None
    duration: float | None = None
    parties: list[int] | int | None = None
    mediatype: str | None = None
    body: str | None = None
    url: str | None = None
    encoding: str | None = None


class Analysis(BaseModel, extra="allow"):
    type: str
    dialog: int | list[int] | None = None
    mediatype: str | None = None
    vendor: str | None = None
    product: str | None = None
    schema_: str | None = None  # 'schema' is reserved in Pydantic
    body: Any = None
    encoding: str | None = None


class Vcon(BaseModel, extra="allow"):
    vcon: str = "0.0.1"
    uuid: str | None = None
    created_at: str | None = None
    parties: list[Party] = []
    dialog: list[Dialog] = []
    analysis: list[Analysis] = []
    attachments: list[Any] = []
