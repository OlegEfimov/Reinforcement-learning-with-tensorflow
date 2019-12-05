import numpy as np
import asyncio
import websockets

from car_env import CarEnv


async def init_handler(websocket, arg_str):
    s = env.init()
    env.render()
    message = "init_done:"
    for num in s:
        message += str(num) + ','
    print("send %s" % str(message[:-1]))
    await websocket.send(message[:-1])

async def reset_handler(websocket, arg_str):
    s = env.reset()
    env.render()
    message = "reset_done:"
    for num in s:
        message += str(num) + ','
    print("send %s" % str(message[:-1]))
    await websocket.send(message[:-1])

async def step_handler(websocket, arg_str):
    arg_data_str = arg_str.split(',')
    arr_str = np.array(arg_data_str)
    arr_float = arr_str.astype(np.float)
    s, r, terminal = env.step(arr_float)
    env.render()

    message = "step_done:"
    for num in s:
        message += str(num) + ','
    message += str(r) + ','
    message += str(terminal)
    print("send %s" % str(message))
    await websocket.send(message)

async def stop_handler(websocket, arg_str):
    # print("--------stop_handler")
    env.stop()
    message = "stop_done:0"
    print("send %s" % str(message))
    await websocket.send(message)

async def unknown_handler():
    print("--------unknown_handler")

def command_selector(message): 
    args = message.split(':')
    switcher = { 
        "init": init_handler,
        "reset": reset_handler,
        "step": step_handler,
        "stop": stop_handler
    } 
    return switcher.get(args[0], unknown_handler), args[1]

async def mess_handler(websocket, path):
    async for message in websocket:
        # print("--------for message in websocket")
        print("receive %s" % str(message))
        cmdHandler, message_data = command_selector(message)
        await cmdHandler(websocket, message_data)

if __name__ == '__main__':
# DDPG initialization

    start_server = websockets.serve(mess_handler, "localhost", 9001)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

