import asyncio
import websockets

from car_env import CarEnv

websocket = 0

recv_data_str = ''

env = CarEnv()

def command_selector(arg): 
    switcher = { 
        "init": init_handler,
        "reset": reset_handler,
        "step": step_handler,
        "stop": stop_handler
    } 
    return switcher.get(arg, unknown_handler)

async def init_handler():
    env.init()
    await websocket.send("init_done")

async def reset_handler():
    s = env.reset()
    await websocket.send("reset_done")

async def step_handler():
    s, r, self.terminal = env.step()

async def stop_handler():
    env.stop()

    global websocket
    print("--------client_send_mess_handler")
    if counter < 10:
        message = "continue"
    else:
        message = "stop"
    print("client_send_mess_handler websocket.send(message)")
    await websocket.send(message)

async def unknown_handler():
    global websocket
    print("--------unknown_handler")
    if counter > 30:
        message = "stop"
    print("unknown_handler websocket.send(message)")
    await websocket.send(message)

async def mess_handler(ws, path):
    global websocket
    websocket = ws
    async for message in websocket:
        print("--------for message in websocket")
        messHandler = command_selector(message)
        messHandler(message)

if __name__ == '__main__':
    s = env.reset()
    action = env.sample_action()

    start_server = websockets.serve(mess_handler, "localhost", 9001)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

# ----------------------------------------------------------------------
async def main_cycle():
    s = env.reset()
    action = env.sample_action()
    recv_data_str = ''
    # for ep in range(20):
    #     s = env.reset()
    #     # for t in range(100):
    #     while True:
    #         env.render()
    #         s, r, done = env.step(env.sample_action())
    #         if done:
    #             break
    uri = "ws://localhost:9001"
    async with websockets.connect(uri) as websocket:
        while True:
            env.render()
            done_mess = 0
            if recv_data_str == 'reset':
                s = env.reset()
                r = 0
                done = False
                done_mess = 0
            else:
                # print("--------------env.step(action) action = %s" % str(action))
                s, r, done = env.step(action)
                if done:
                    s = env.reset()
                    done_mess = 1
                    print("---------------env.reset() %s" % str(done))

                print("send reward: %s" % str(r))
                mess = str(r) + ',' + str(done_mess)
                await websocket.send(mess)


            state_as_string = ''
            for num in s:
                state_as_string += str(num) + ','
            print("send state: %s" % str(state_as_string[:-1]))
            await websocket.send(state_as_string[:-1])
            # recv_data = ''

            # actionZZZZZZZ = await websocket.recv()
            # print("actionZZZZZZZ %s" % actionZZZZZZZ)
            # recv_data_str = str(actionZZZZZZZ)
            # if recv_data_str != 'reset':
            #     print(type(actionZZZZZZZ))
            #     actionTmp = float(actionZZZZZZZ)
            #     print(type(actionTmp))
            #     action = np.array([actionTmp])
            # print("receive action: %s" % recv_data_str)

            action = await websocket.recv()
            recv_data_str = str(action)
            if recv_data_str != 'reset':
                action = np.array([float(action)])
            print("receive action: %s" % recv_data_str)

