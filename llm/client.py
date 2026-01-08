import httpx

from config import settings

class MistralClient:
    def __init__(self, base_url=None, model_name=None, api_key=None, timeout=None, temp=None, max_tokens=None):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model_name = model_name or settings.llm_model_name
        self.api_key = api_key
        self.timeout = timeout or settings.llm_timeout
        self.temp = temp or settings.llm_temp
        self.max_tokens = max_tokens or settings.llm_max_tokens

        headers = {}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=self.timeout
        )

    def chat(self, messages):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temp,
            "max_tokens": self.max_tokens
        }

        try:
            response = self.client.post("/chat/completions", json=payload)
            response.raise_for_status()

            data = response.json()

            if data.get("choices"):
                return data["choices"][0]["message"]["content"]
                
            raise ValueError(f"Unexpceted response format: {data}")
            
        except httpx.HTTPError as e:
            detail = ""
            
            if hasattr(e, "response") and e.response is not None:
                try:
                    detail = f"\nResponse body: {e.response.text}"
                
                except Exception:
                    pass

            raise Exception(f"LLM request failed: {e}{detail}") from e
        
    def generate(self, prompt):
        messages=[{"role": "user", "content": prompt}]
        return self.chat(messages)
    
    def close(self):
        self.client.close()

    # For context manager
    def __enter__(self):
        return self
    
    # For context manager
    def __exit__(self):
        self.close()