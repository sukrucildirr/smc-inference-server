from pydantic import BaseModel

class GenerationRequest(BaseModel):
    prompt: str
    num_tokens: int = 20
    num_particles: int = 3
    beam_factor: int = 1
    temperature: float = 1.0