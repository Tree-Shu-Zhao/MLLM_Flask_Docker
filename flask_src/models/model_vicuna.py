import torch
from PIL import Image
from typing import Optional
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.utils import get_context_length


class Vicuna:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.repetition_penalty = 0.

        # Model
        self.model, self.tokenizer = load_model(model_path)
        self.generate_stream_func = get_generate_stream_function(self.model, model_path)

        model_type = str(type(self.model)).lower()
        is_t5 = "t5" in model_type
        self.is_codet5p = "codet5p" in model_type

        # Hardcode T5's default repetition penalty to be 1.2
        if is_t5 and self.repetition_penalty == 1.0:
            self.repetition_penalty = 1.2

        # Set context length
        self.context_len = get_context_length(self.model.config)

        self.temperature = 0
        self.max_new_tokens = 512

    def generate(self, query: str, image: Optional[Image.Image]=None):
        conv = get_conversation_template(self.model_path)
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if self.is_codet5p:  # codet5p is a code completion model.
            prompt = query

        gen_params = {
            "model": self.model_path,
            "prompt": prompt,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }
        output_stream = self.generate_stream_func(
            self.model,
            self.tokenizer,
            gen_params,
            "cuda",
            context_len=self.context_len,
            judge_sent_end=True,
        )
        outputs = self.stream_output(output_stream)
        conv.update_last_message(outputs.strip())

        return outputs.strip()


    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        return " ".join(output_text)



