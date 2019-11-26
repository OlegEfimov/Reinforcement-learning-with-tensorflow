import asyncio
import websockets

counter = 0
websocket = 0

def mess_selector(arg): 
    switcher = { 
        "client_send_mess": client_send_mess_handler
    } 
    return switcher.get(arg, unknown_handler)

async def client_send_mess_handler():
    global counter
    global websocket
    print("--------client_send_mess_handler")
    counter += 1
    if counter < 10:
        message = "continue"
    else:
        message = "stop"
    print("client_send_mess_handler websocket.send(message)")
    await websocket.send(message)

async def unknown_handler():
    global counter
    global websocket
    print("--------unknown_handler")
    counter += 1
    if counter > 30:
        message = "stop"
    print("unknown_handler websocket.send(message)")
    await websocket.send(message)

async def mess_handler(ws, path):
    global websocket
    websocket = ws
    async for message in websocket:
        print("--------for message in websocket")
        messHandler = mess_selector(message)
        await messHandler()

if __name__ == '__main__':
    start_server = websockets.serve(mess_handler, "localhost", 9001)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


