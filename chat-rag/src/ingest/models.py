from datetime import datetime

from pydantic import BaseModel, Field


class Message(BaseModel):
    id: int
    date: datetime
    from_: str = Field(alias="from")
    text: str = ""
    chat_name: str = ""
    topic: str = ""

    model_config = {"populate_by_name": True}


class ChatExport(BaseModel):
    name: str
    type: str
    id: int
    topic: str
    messages: list[Message]


class ChunkMetadata(BaseModel):
    conversation_id: str
    chat_name: str
    topic: str
    start_time: datetime
    end_time: datetime
    senders: str
