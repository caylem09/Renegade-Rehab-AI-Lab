import re
import requests


def remove_thinking_text(text: str) -> str:
    """
    Some local models may include hidden-thinking style text.
    This removes anything between <think> and </think> if it appears.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def generate_ai_summary(summary_data: str) -> str:
    prompt = f"""
You are an applied sport science assistant.

Write a short coach-facing summary based on this activity-recognition dashboard data.

Rules:
- Keep it under 200 words.
- Do not diagnose injury.
- Do not prescribe medical treatment.
- Mention that this is a proof of concept.
- Explain how this could support sport science, movement analysis, or human performance assessment.
- Use plain English.

Dashboard data:
{summary_data}
"""

    try:
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
        raw_text = response.json().get("response", "")
        return remove_thinking_text(raw_text)

    except Exception as error:
        return (
            "Local AI summary could not be generated. "
            "Make sure Ollama is open and qwen3:4b is installed. "
            f"Error: {error}"
        )
