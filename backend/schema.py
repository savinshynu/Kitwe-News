# pydantic is used to structure and validate the inputs and outputs
# Pydantic is similiar to data classes but offer runtime validation, good for collecting data through APIs

from pydantic import BaseModel

class TextRequest(BaseModel):
    """
    The format of request for the server end
    input text
    """
    text: str

class PredictionResponse(BaseModel):
    """
    Format of output response:
    intiger label and probability score.
    """
    label: int
    confidence: float
