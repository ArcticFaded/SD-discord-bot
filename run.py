from server import bot
import asyncio
import os

bot_token = os.environ.get("discord_key", None)
async def run():
    try:
        await bot.start(bot_token)
    except KeyboardInterrupt:
        await bot.close()

# Because the discord bot is running inside of its own loop, an optional http server can run alongside it

loop = asyncio.get_event_loop()
task = loop.create_task(run())

loop.run_until_complete(asyncio.wait([task]))
loop.close()
