#!/usr/bin/env python

# WS client example

import asyncio
import websockets

async def hello():
    uri = "ws://localhost:9001"
    async with websockets.connect(uri) as websocket:
        while True:
            name = input("enter text: ")

            await websocket.send(name)
            # print(f"> {name}")
            # print("Client send: %s" % name)

            greeting = await websocket.recv()
            # print(f"< {greeting}")
            print("Server return: %s" %  greeting)


asyncio.get_event_loop().run_until_complete(hello())