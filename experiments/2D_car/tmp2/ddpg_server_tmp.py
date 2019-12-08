import asyncio
import websockets

from config_env import ConfigEnv
from agent import Agent


config_env = ConfigEnv()
agent = Agent(config_env)

async def unknown_handler():
    print("DDPG server: unknown_handler")

async def notify_client(websocket, message):
    if websocket:
        await asyncio.wait([websocket.send(message)])

async def register(websocket):
    # TBD
    await notify_client(websocket, 'register_done')

async def unregister(websocket):
    # TBD
    await notify_client(websocket,'unregister_done')

async def action_done_handler(websocket, arg_str):
    action = agent.handle_new_state(arg_str)
    message = "step:"
    for num in action:
        message += str(num) + ','
    await websocket.send(message[:-1])

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
    args = message.split(':')
    switcher = { 
        "action_done": action_done_handler,
    }
    return switcher.get(args[0], unknown_handler), args[1]

if __name__ == '__main__':
    # agent.init()

    start_server = websockets.serve(mess_handler, "localhost", 9001)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
