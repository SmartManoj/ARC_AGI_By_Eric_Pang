from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ["OPENROUTER_API_KEY"]
)

import json
with open("messages.json", "r") as f:
  messages = json.load(f)

completion = client.chat.completions.create(
  extra_body={},
  model="deepseek/deepseek-chat-v3.1:free",
  messages=messages
)
print(completion.choices[0].message.content)