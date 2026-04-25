import requests

prompt = """
Explain AthleteIQ in one short paragraph.

AthleteIQ is a proof of concept that uses wearable movement data,
heart-rate data, machine learning, and a dashboard to support sport
science and human performance analysis.
"""

print("Sending prompt to local Ollama model...")

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3:4b",
        "prompt": prompt,
        "stream": False,
    },
    timeout=120,
)

response.raise_for_status()

answer = response.json()["response"]

print()
print("Local AI response:")
print(answer)
