import asyncio
import math
import numpy as np
import re
import torch
import torch.nn.functional as F

from scipy.stats import entropy
from scipy.spatial.distance import cosine
from typing import List, Set

from llamppl import smc_steer
from llamppl import Model, LMContext, CachedCausalLM, TokenCategorical, Token
from llamppl import log_softmax

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
END = "\033[0m"

NUM_PARTICLES = 3
BEAM_FACTOR = 3
ENTROPY_THRESHOLD = 2.5
MAX_TOKENS = 150
ITERATIONS = 1

PROMPT = "Which number is larger, 9.11 or 9.9?"
uncertainty_token = "wait..."

def pretty_format(particle):
    context_str = str(particle.context)
    new_context_str = re.sub(f"({re.escape(uncertainty_token)})", f"{YELLOW}\\1{END}", context_str)
    return f"{new_context_str} (weight: {RED}{particle.weight:.3f}{END})\n"

class NightwingEntropyModel(Model):
    def __init__(
        self,
        lm: CachedCausalLM,
        prompt: str,
        uncertainty_token: str = "Wait...",
        entropy_threshold: float = 3.0,
        max_tokens: int = 100,
        min_tokens_between_uncertainty: int = 50
    ):
        super().__init__()

        print(f"\nInitializing model...")

        self.lm = lm
        self.context = LMContext(lm, prompt)
        self.uncertainty_tokens = self.lm.tokenizer.encode(
            f" {uncertainty_token}", # Encode uncertainty token with a space in front
            add_special_tokens=False
        )
        self.entropy_threshold = entropy_threshold
        self.max_tokens = max_tokens

        # Tracking uncertainty
        self.min_tokens_between_uncertainty = min_tokens_between_uncertainty
        self.generated_tokens = []
        self.last_uncertainty_pos = -self.min_tokens_between_uncertainty
        self.is_generating_uncertainty = False


    # Entropy calculation with LSE and normalization
    def calculate_entropy(self, logprobs: np.ndarray) -> float:
        probs = np.exp(logprobs - np.max(logprobs)) # log-sum-exp
        probs = probs / np.sum(probs) # Normalize

        return float(-np.sum(probs * np.log(probs + 1e-10))) # Add constant to avoid taking log(0)

    async def step(self):
        if len(self.generated_tokens) >= self.max_tokens:
            self.finish()
            return

        # Current position in generated output
        current_pos = len(self.generated_tokens)
        current_entropy = 0.0
        
        # If we aren't currently inserting uncertainty tokens
        # And we're outside of the minimum distance to insert tokens
        if (not self.is_generating_uncertainty and 
            current_pos - self.last_uncertainty_pos >= self.min_tokens_between_uncertainty):
            
            # Calculate entropy values
            logprobs = self.context.next_token_logprobs
            current_entropy = self.calculate_entropy(logprobs)
            
            # If entropy is above the threshold and it's not the first token
            if current_entropy > self.entropy_threshold and current_pos != 0:
                self.is_generating_uncertainty = True
                self.last_uncertainty_pos = current_pos

        # Observe the uncertainty token
        if self.is_generating_uncertainty:
            if len(self.generated_tokens) - self.last_uncertainty_pos < len(self.uncertainty_tokens):

                # Create token
                token_idx = len(self.generated_tokens) - self.last_uncertainty_pos
                token = Token(self.lm, self.uncertainty_tokens[token_idx], self.lm.tokenizer.decode(self.uncertainty_tokens[token_idx]))

                # Observe the token over the next token distribution
                next_dist = self.context.next_token()
                await self.observe(next_dist, token)
                score = math.log(((current_entropy-ENTROPY_THRESHOLD)**2))
                #self.twist(score)

                self.generated_tokens.append(token)
                if token_idx == len(self.uncertainty_tokens) - 1:
                    self.is_generating_uncertainty = False
            return

        # Normal sampling behavior
        next_dist = self.context.next_token()
        token = await self.sample(next_dist)
        self.generated_tokens.append(token)

        if token.token_id == self.lm.tokenizer.eos_token_id:
            self.finish()
        
async def main():
    lm = CachedCausalLM.from_pretrained("NousResearch/Hermes-3-Llama-3.2-3B", backend='hf')
    lm.batch_size = 8

    model = NightwingEntropyModel(
        lm=lm,
        prompt=PROMPT,
        uncertainty_token=uncertainty_token,
        entropy_threshold=ENTROPY_THRESHOLD,
        max_tokens=MAX_TOKENS
    )

    print(f"\nSteering with smc_steer with {GREEN}{NUM_PARTICLES}{END} particles and beam factor {GREEN}{BEAM_FACTOR}{END}")

    for i in range(ITERATIONS):
        particles = await smc_steer(model, NUM_PARTICLES, BEAM_FACTOR)

        print(f"\n{GREEN}{PROMPT}{END}")
        for particle in particles:
            print(pretty_format(particle))

if __name__ == "__main__":
    asyncio.run(main())