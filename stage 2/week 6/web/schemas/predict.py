from pydantic import BaseModel

class SubmitRequest(BaseModel):
    title: str
    label: str
