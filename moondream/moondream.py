import torch
from .vision_encoder import VisionEncoder
from .configuration_moondream import MoondreamConfig
from transformers import PreTrainedModel
import re

from .modeling_phi import PhiForCausalLM
from .configuration_moondream import PhiConfig


class Moondream(PreTrainedModel):
    config_class = MoondreamConfig
    # Newer transformers (>=4.50) reads `self.all_tied_weights_keys` from
    # several from_pretrained code paths. Phi doesn't tie any weights, and the
    # outer Moondream class adds none either, so an empty dict is correct.
    # Defining it as a plain class attribute (no setter) is also important: if
    # this were a property, post_init()'s implicit assignment would crash.
    all_tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)
        self.vision_encoder = VisionEncoder()

        if type(config.phi_config) == dict:
            phi_config = PhiConfig(**config.phi_config)
        else:
            phi_config = config.phi_config
        self.text_model = PhiForCausalLM(phi_config)
        # NOTE: do NOT call self.post_init() here. Under transformers >=4.50's
        # meta-device init pipeline, calling post_init() at the outer level
        # re-runs _init_weights AFTER the checkpoint has been loaded, which
        # silently overwrites every Linear/Embedding weight with N(0, 0.02)
        # noise (LayerNorms survive because _init_weights skips them). The
        # inner PhiForCausalLM still calls its own post_init() in __init__,
        # which is enough — that runs BEFORE load and gets overwritten by
        # the checkpoint values, as intended.

    @property
    def device(self):
        return self.text_model.device

    def encode_image(self, image):
        return self.vision_encoder(image)

    def input_embeds(self, prompt, image_embeds, tokenizer):
        def _tokenize(txt):
            return tokenizer(
                txt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)

        text_emb = self.text_model.get_input_embeddings()

        # Add BOS token
        embeds = []
        embeds.append(
            text_emb((torch.tensor([[tokenizer.bos_token_id]], device=self.device)))
        )

        if "<image>" not in prompt:
            embeds.append(text_emb(_tokenize(prompt)))
        else:
            assert prompt.count("<image>") == 1
            before, after = prompt.split("<image>")
            embeds.append(text_emb(_tokenize(f"{before}<image>")))
            embeds.append(image_embeds.to(self.device))
            embeds.append(text_emb(_tokenize(f"</image>{after}")))

        return torch.cat(embeds, dim=1)

    def generate(
        self,
        image_embeds,
        prompt,
        tokenizer,
        eos_text="<END>",
        max_new_tokens=128,
        **kwargs,
    ):
        eos_tokens = tokenizer(eos_text, add_special_tokens=False)[0].ids

        generate_config = {
            "eos_token_id": eos_tokens,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": max_new_tokens,
            **kwargs,
        }

        with torch.no_grad():
            inputs_embeds = self.input_embeds(prompt, image_embeds, tokenizer)
            output_ids = self.text_model.generate(
                inputs_embeds=inputs_embeds, **generate_config
            )

        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def answer_question(
        self,
        image_embeds,
        question,
        tokenizer,
        max_new_tokens,
        chat_history="",
        result_queue=None,
        **kwargs,
    ):
        prompt = f"<image>\n\n{chat_history}Question: {question}\n\nAnswer: "
        answer = self.generate(
            image_embeds,
            prompt,
            eos_text="<END>",
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )[0]
        cleaned_answer = re.sub("<$|<END$", "", answer).strip()

        # Use the result_queue to pass the result if it is provided
        if result_queue:
            result_queue.put(cleaned_answer)
        else:
            return cleaned_answer
