import asyncio
import websockets

from config_env import ConfigEnv
from agent import Agent


config = ConfigEnv()
agent = Agent(config)

async def unknown_handler():
    print("DDPG server: unknown_handler")

async def notify_client(websocket, message):
    if websocket:
        await asyncio.wait([websocket.send(message)])

async def register(websocket):
    print("register")
    # TBD
    await notify_client(websocket, 'register_done')

async def unregister(websocket):
    print("unregister")
    # TBD
    # await notify_client(websocket,'unregister_done')

async def state_handler(websocket, arg_str):
    print("state_handler(arg_str) arg_str=%s" % arg_str)
    action = agent.handle_new_state(arg_str)
    message = "action:"
    for num in action:
        message += str(num) + ','
    print("ddpg send %s" % str(message[:-1]))
    await websocket.send(message[:-1])

async def save_handler(websocket, arg_str):
    agent.handle_save(arg_str)
    print("save_handler")

async def load_handler(websocket, arg_str):
    agent.handle_load(arg_str)
    print("load_handler")

async def mess_handler(websocket, path):
    await register(websocket)
    try:
        async for message in websocket:
            print("DDPG server receive %s" % str(message))
            cmdHandler, message_data = mess_selector(message)
            await cmdHandler(websocket, message_data)
        print("probably mess_handler should not be here!!!")
    finally:
        await unregister(websocket)

def mess_selector(message):
    print("mess_selector(message) message=%s", message)
    args = message.split(':')
    switcher = { 
        "state": state_handler,
        "save": save_handler,
        "load": load_handler,
    }
    return switcher.get(args[0], unknown_handler), args[1]

if __name__ == '__main__':
    agent.init()

    start_server = websockets.serve(mess_handler, "localhost", 9001)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
