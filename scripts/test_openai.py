import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Are you connected to this project? And if so can you give me a summary of it."}]
)

print(response.choices[0].message.content)
