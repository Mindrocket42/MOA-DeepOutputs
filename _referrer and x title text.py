import httpx
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HTTP_REFERER = os.getenv("HTTP_REFERER")
X_TITLE = os.getenv("X_TITLE")

async def test_api():
       async with httpx.AsyncClient(
           headers={
               "Authorization": f"Bearer {OPENROUTER_API_KEY}",
               "HTTP-Referer": HTTP_REFERER,
               "X-Title": X_TITLE,
               "Content-Type": "application/json"
           }
       ) as client:
           response = await client.post("https://openrouter.ai/api/v1/chat/completions", json={"prompt": "Test"})
           print(f"Response status: {response.status_code}, Response content: {response.text}")


asyncio.run(test_api())