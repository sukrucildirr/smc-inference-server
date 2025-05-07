import asyncio
from llamppl import smc_steer
from llamppl import Model, LMContext, CachedCausalLM

from transformers import AutoTokenizer

from scipy.spatial.distance import cosine

GREEN = "\033[92m"
RED = "\033[91m"
END = "\033[0m"

PROMPT = "When you turn 16 in this country you can legally drive a"

banned_tokens = [
    "car",
    "vehicle"
]

class NightwingModel(Model):

    def __init__(self, lm, prompt, forbidden_topics, max_tokens=50):
        super().__init__()

        print(f"\nInitializing model...")

        self.context = LMContext(lm, prompt)
        self.eos_token = lm.tokenizer.eos_token_id
        self.max_tokens = max_tokens
        self.lm = lm
        self.embedding_matrix = self.lm.model.get_input_embeddings().weight.detach().cpu().numpy()

        ## Fields needed to store banned tokens
        self.forbidden_tokens = set()
        if isinstance(forbidden_topics, str):
            forbidden_topics = [forbidden_topics]

        forbidden_topics = list(set(forbidden_topics))
        for topic in forbidden_topics:
            self.forbidden_tokens.update(self.get_related_words(topic))

    async def step(self):
        await self.observe(self.context.mask_dist(self.forbidden_tokens), False)
        token = await self.sample(self.context.next_token())
        self.max_tokens -= 1

        if token.token_id == self.eos_token or self.max_tokens == 0:
            self.finish()

    def get_related_words(self, topic: str, num_related: int = 25):
        """Get 25 related tokens for each topic using embedding matrix and cosine similarity"""
        if topic: print(f"\nFinding {str(num_related)} similar tokens to {GREEN}{topic}{END}")
        topic_tokens = self.lm.tokenizer.encode(topic, add_special_tokens=False)
        related_tokens = set(topic_tokens)
        
        for token_id in topic_tokens:
            token_embedding = self.embedding_matrix[token_id]
            
            similarities = []
            for i, other_embedding in enumerate(self.embedding_matrix):
                if i not in related_tokens:
                    similarity = 1 - cosine(token_embedding, other_embedding)
                    similarities.append((i, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)

            print(f"\nTop 5 similar tokens to {GREEN}{token_id}{END}:\n")
            for token_id, _ in similarities[:5]:
                print(self.lm.tokenizer.decode(token_id))
            
            related_tokens.update(tok_id for tok_id, _ in similarities[:num_related])
        
        return related_tokens

    def immutable_properties(self):
        return set(['banned_tokens'])

async def main():
    lm = CachedCausalLM.from_pretrained("NousResearch/Hermes-3-Llama-3.2-3B", backend='hf')

    model = NightwingModel(lm, PROMPT, banned_tokens)
    model.batch_size = 8
    num_particles = 3

    print(f"\nSteering with SMC with {GREEN}{num_particles}{END} particles\n")
    particles = asyncio.run(smc_steer(model, num_particles, beams))
    print("\nFinished steering")

    print(f"\n{GREEN}{PROMPT}{END}")
    for particle in sorted(particles, key=lambda p: p.weight, reverse=True)[:1]:
        print(f"{particle.context} (weight: {RED}{particle.weight:.3f}{END})\n")

if __name__ == "__main__":
    asyncio.run(main())