import os
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

load_dotenv()

class OpenAIFallbackAgent:
    """A simple agent that tries to fetch from cache first and falls back to OpenAI if needed.
    This is a placeholder for your actual LLM agent logic, which would include prompt engineering, API calls, and response parsing.
    """
    def __init__(self):
        self.disable_openai = os.getenv("DISABLE_OPENAI", "0") == "1"
        self.client = None if self.disable_openai else OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # standard pricing for the gpt-4o-mini model is exactly $0.15 per million input tokens and $0.60 per million output tokens
        self.model = "gpt-4o-mini"
        self.INPUT_COST_PER_TOKEN = 0.15 / 1_000_000
        self.OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000

    def generate_formulary_fallback(self, query: str):
        """Compatibility wrapper used by app.py."""
        return self.llm_fallback(query)

    def llm_fallback(self, query: str):
        """Calls OpenAI live, tracks token expenditures, and measures runtime latency."""
        start_time = time.time()

        if self.disable_openai or self.client is None:
            end_time = time.time()
            latency = end_time - start_time
            text = "LLM_DISABLED"
            return {
                "response": text,
                "text": text,
                "tokens_used": 0,
                "cost_usd": 0.0,
                "latency_ms": latency * 1000,
            }
        
        system_prompt = (
            "You are an expert Pharmacy Benefit Manager (PBM) formulary assistant. "
            "Analyze the user's non-covered drug request and provide a generic alternative "
            "along with its standard tier placement. Be concise and precise."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )

        end_time = time.time()
        latency = end_time - start_time

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = (input_tokens * self.INPUT_COST_PER_TOKEN) + (output_tokens * self.OUTPUT_COST_PER_TOKEN)
        llm_response = response.choices[0].message.content
        
        return {
            "response": llm_response,
            "text": llm_response,
            "tokens_used": input_tokens + output_tokens,
            "cost_usd": cost,
            "latency_ms": latency * 1000,
        }