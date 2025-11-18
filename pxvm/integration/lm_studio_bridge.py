#!/usr/bin/env python3
"""
pxvm/integration/lm_studio_bridge.py

A bridge for communicating with a local LLM running in LM Studio.
"""
import requests
import json

class LMStudioBridge:
    def __init__(self, base_url="http://localhost:1234/v1"):
        self.base_url = base_url

    def query(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """
        Sends a prompt to the LM Studio server and returns the response.
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "local-model", # Loaded model in LM Studio
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to LM Studio: {e}")
            # Return a dummy JSON response to allow the generator to proceed
            return json.dumps({
                "feature": "Error connecting to LM Studio",
                "primitives": [
                    {
                        "type": "COMMENT",
                        "comment": "Could not connect to the local LLM. Please ensure LM Studio is running."
                    }
                ]
            })
