# OpenAI/Gemini explanation & summarization


import openai
import os
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def explain_text(text: str) -> str:
    if not text.strip():
        return "No text detected in the image."
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant explaining lecture content for visually impaired students. Be concise and clear."},
            {"role": "user", "content": f"Explain this lecture content simply:\n\n{text}"}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content