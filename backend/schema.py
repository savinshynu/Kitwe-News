# pydantic is used to structure and validate the inputs and outputs
# Pydantic is similiar to data classes but offer runtime validation, good for collecting data through APIs
# typing used for type annotation, here defining static List
# You usually give the input as a json format and get the output as a json format.

from pydantic import BaseModel
from typing import List 

class TextRequest(BaseModel):
    """
    The format of request for the server end
    input text
    """
    texts: List[str] # list of texts

class PredictionResponse(BaseModel):
    """
    Format of output response:
    intiger label and probability score.
    """
    label: List[int]
    confidence: List[float]
