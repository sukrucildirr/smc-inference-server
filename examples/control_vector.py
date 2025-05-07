import asyncio
import json
import math
import numpy as np
import re
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Set


from llamppl import smc_steer
from llamppl import Model, LMContext, CachedCausalLM, TokenCategorical, Token, Transformer
from llamppl import log_softmax
from llamppl import sample_word

# Needs repeng installed (https://github.com/vgel/repeng)
from repeng import ControlVector, ControlModel, DatasetEntry

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
END = "\033[0m"

NUM_PARTICLES = 3
BEAM_FACTOR = 1
ENTROPY_THRESHOLD = 2.5
MAX_TOKENS = 200
ITERATIONS = 1
PROMPT = "Here is a short story about cheese: "

cvec_token_instances = []

def pretty_format(particle, tokenizer):
    """This method highlights control vector areas using cvec_token_instances"""
    context_str = str(particle.context)

    token_ids = tokenizer(context_str)["input_ids"]
    prompt_ids = tokenizer(prompt)["input_ids"]

    yellow_tokens = tokenizer(YELLOW, add_special_tokens=False)["input_ids"]
    end_tokens = tokenizer(END, add_special_tokens=False)["input_ids"]

    for pair in reversed(cvec_token_instances):
        start_idx = pair["start"]- len(prompt_ids)
        end_idx = pair["end"] - len(prompt_ids)
        token_ids = token_ids[:start_idx + 1] + yellow_tokens + token_ids[start_idx + 1:]
        token_ids = token_ids[:end_idx + len(yellow_tokens) + 1] + end_tokens + token_ids[end_idx + len(yellow_tokens) + 1:]

    modified_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return f"{modified_text} (weight: {RED}{particle.weight:.3f}{END})\n"

class NightwingControlVectorModel(Model):
    def __init__(
        self,
        lm: CachedCausalLM,
        prompt: str,
        max_tokens: int = 200,
        base_model = None,
        control_model = None
    ):
        super().__init__()

        print(f"\nInitializing model...")

        self.lm = lm
        self.context = LMContext(lm, prompt)

        self.max_tokens = max_tokens
        self.generated_tokens = []

        self.control_vector = self.train_control_vector(lm)
        self.control_model = control_model
        self.control_model.model = ControlModel(control_model.model, list(range(-5, -12, -1)))
        self.control_model.model.set_control(self.control_vector, 2.2)
        self.is_generating_control_vector = False
        self.cvec_tokens = []
        self.current_cvec = {"start": 0, "end": 0}

    # Trigger control vector between tokens 25 and 50
    async def step(self):
        if len(self.generated_tokens) >= self.max_tokens:
            self.finish()
            return

        if len(self.generated_tokens) == 26:
            cvec_token_instances.append({"start": len(self.context.tokens), "end": len(self.context.tokens) + 25})

        if len(self.generated_tokens) <= 25 or len(self.generated_tokens) >= 50:
            next_dist = self.context.next_token()
            token = await self.sample(next_dist)
            self.generated_tokens.append(token)
        else:
            control_context = Transformer(self.control_model, self.context.tokens)
            next_dist = self.context.next_token()
            token = await self.sample(next_dist, proposal=control_context)
            self.generated_tokens.append(token)

        if token.token_id == self.lm.tokenizer.eos_token_id:
            self.finish()

    # Make dataset to train control vector
    def make_dataset(self, template: str, pos_personas: list[str], neg_personas: list[str], suffixes: list[str]):
        dataset = []
        user_tag, asst_tag = "", ""
        for suffix in suffixes:
            for positive_persona, negative_persona in zip(pos_personas, neg_personas):
                positive_template = template.format(persona=positive_persona)
                negative_template = template.format(persona=negative_persona)
                dataset.append(
                    DatasetEntry(
                        positive=f"{user_tag} {positive_template} {asst_tag} {suffix}",
                        negative=f"{user_tag} {negative_template} {asst_tag} {suffix}",
                    )
                )
        return dataset

    def train_control_vector(self, lm):
        print("Training control vector...")
        user_tag, asst_tag = "", ""
        lm.tokenizer.pad_token_id = 0

        with open("/path/to/repeng/notebooks/data/all_truncated_outputs.json") as f:
            output_suffixes = json.load(f)

        truncated_output_suffixes = [
            lm.tokenizer.convert_tokens_to_string(tokens[:i])
            for tokens in (lm.tokenizer.tokenize(s) for s in output_suffixes)
            for i in range(1, len(tokens))
        ]

        trippy_dataset = self.make_dataset(
            "Act as if you're extremely {persona}.",
            ["high on psychedelic drugs"],
            ["sober from psychedelic drugs"],
            truncated_output_suffixes,
        )

        trippy_vector = ControlVector.train(lm.model, lm.tokenizer, trippy_dataset)

        return trippy_vector
    
async def main():
    model_string = "NousResearch/Hermes-3-Llama-3.2-3B"
    lm = CachedCausalLM.from_pretrained(model_string, backend='hf')
    control_lm = CachedCausalLM.from_pretrained(model_string, backend='hf')

    control_model = ControlModel(lm.model, list(range(-5, -17, -1)))

    device_0 = torch.device("cuda:0")
    device_1 = torch.device("cuda:1")

    lm.device = device_0
    control_lm.device = device_1

    lm.batch_size = 8

    model = NightwingControlVectorModel(
        lm=lm,
        prompt=PROMPT,
        entropy_threshold=ENTROPY_THRESHOLD,
        max_tokens=MAX_TOKENS,
        control_model=control_lm
    )

    print(f"\nSteering with smc_steer with {GREEN}{NUM_PARTICLES}{END} particles and beam factor {GREEN}{BEAM_FACTOR}{END}")

    for i in range(ITERATIONS):
        particles = await smc_steer(model, NUM_PARTICLES, BEAM_FACTOR)
        print(f"\n{GREEN}{PROMPT}{END}")
        sorted_particles = sorted(particles, key=lambda p: p.weight, reverse=True)
        print(cvec_token_instances)
        print(pretty_format(sorted_particles[0], lm.tokenizer))

if __name__ == "__main__":
    asyncio.run(main())