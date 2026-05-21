# defines what needs to be imported from this folder
# every folder needs a __init__.py for python to see it as a package, otherwise seen as a folder
from .model_quantized import model_handler
from .model_distilbert import DistilbertClassifier

#controls what is imported when someone does "import *"
__all__ = ["model_handler"]