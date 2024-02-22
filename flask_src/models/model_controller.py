from .model_llava import LLaVA


class ModelController:
    def __init__(self, model_version):
        device_id = 0
        if model_version == "llava":
            self.model = LLaVA("liuhaotian/llava-v1.5-7b", device_id)

    def generate(self, text, image=None):
        response = self.model.generate(text, image)
        return {"response": response}

