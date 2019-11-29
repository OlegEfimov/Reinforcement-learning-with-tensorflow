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
    # print("send %s" % str(message[:-1]))
    await websocket.send(message[:-1])

async def reset_handler(websocket, arg_str):
    s = env.reset()
    env.render()
    message = "reset_done:"
    for num in s:
        message += str(num) + ','
    # print("send %s" % str(message[:-1]))
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
    # print("send %s" % str(message))
    await websocket.send(message)

async def stop_handler():
    # print("--------stop_handler")
    env.stop()
    message = "stop_done:0"
    # print("send %s" % str(message))
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
        cmdHandler, message_data = command_selector(message)
        await cmdHandler(websocket, message_data)

if __name__ == '__main__':
    np.random.seed(1)
    env = CarEnv()
    env.set_fps(30)
    s = env.reset()
    action = env.sample_action()

    start_server = websockets.serve(mess_handler, "localhost", 9001)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


# ----------------------------------------------------------------------
# async def main_cycle():
#     s = env.reset()
#     action = env.sample_action()
#     recv_data_str = ''
#     # for ep in range(20):
#     #     s = env.reset()
#     #     # for t in range(100):
#     #     while True:
#     #         env.render()
#     #         s, r, done = env.step(env.sample_action())
#     #         if done:
#     #             break
#     uri = "ws://localhost:9001"
#     async with websockets.connect(uri) as websocket:
#         while True:
#             env.render()
#             done_mess = 0
#             if recv_data_str == 'reset':
#                 s = env.reset()
#                 r = 0
#                 done = False
#                 done_mess = 0
#             else:
#                 # print("--------------env.step(action) action = %s" % str(action))
#                 s, r, done = env.step(action)
#                 if done:
#                     s = env.reset()
#                     done_mess = 1
#                     print("---------------env.reset() %s" % str(done))

#                 print("send reward: %s" % str(r))
#                 mess = str(r) + ',' + str(done_mess)
#                 await websocket.send(mess)


#             state_as_string = ''
#             for num in s:
#                 state_as_string += str(num) + ','
#             print("send state: %s" % str(state_as_string[:-1]))
#             await websocket.send(state_as_string[:-1])
#             # recv_data = ''

#             # actionZZZZZZZ = await websocket.recv()
#             # print("actionZZZZZZZ %s" % actionZZZZZZZ)
#             # recv_data_str = str(actionZZZZZZZ)
#             # if recv_data_str != 'reset':
#             #     print(type(actionZZZZZZZ))
#             #     actionTmp = float(actionZZZZZZZ)
#             #     print(type(actionTmp))
#             #     action = np.array([actionTmp])
#             # print("receive action: %s" % recv_data_str)

#             action = await websocket.recv()
#             recv_data_str = str(action)
#             if recv_data_str != 'reset':
#                 action = np.array([float(action)])
#             print("receive action: %s" % recv_data_str)

