#!/usr/bin/env python

# WS client example

import asyncio
import websockets
import time

async def hello():
    uri = "ws://localhost:9001"
    count = 0
    async with websockets.connect(uri) as websocket:
        while True:
            time.sleep(0.1)
            # name = input("enter text: ")
            count = count + 1
            mess = str(count)

            # await websocket.send(name)
            await websocket.send(mess)
            # print(f"> {name}")
            # print("Client send: %s" % mess)

            greeting = await websocket.recv()
            # print(f"< {greeting}")
            print("Server return: %s" %  greeting)


asyncio.get_event_loop().run_until_complete(hello())