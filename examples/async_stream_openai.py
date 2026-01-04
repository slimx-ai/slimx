import asyncio
from slimx import allm

async def main():
    m = allm("openai:gpt-4.1-nano")
    async for ev in m.astream("Count from 1 to 5, one per line."):
        if ev.type == "token":
            print(ev.text, end="", flush=True)
    print()

asyncio.run(main())
