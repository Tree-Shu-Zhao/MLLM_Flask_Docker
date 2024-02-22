try:
    from .model_llava import LLaVA
except:
    pass

try:
    from .model_vicuna import Vicuna
except:
    pass


class ModelController:
    def __init__(self, model_version):
        device_id = 0
        if model_version == "llava":
            self.model = LLaVA("liuhaotian/llava-v1.5-7b", device_id)
        elif model_version == "vicuna":
            self.model = Vicuna("lmsys/vicuna-7b-v1.5")

    def generate(self, text, image=None):
        response = self.model.generate(text, image)
        return {"response": response}

