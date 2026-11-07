from .DeepDDI import Predictor

class SeverityPredictor(Predictor):

    def __init__(self, root=None):
        super().__init__('severity', **{} if root is None else {'root': root})
