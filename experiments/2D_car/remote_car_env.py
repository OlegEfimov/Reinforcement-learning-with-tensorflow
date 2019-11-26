import asyncio
import websockets

class RemoteCarEnv(object):
    n_sensor = 5
    action_dim = 1
    state_dim = n_sensor

    def __init__(self):

        asyncio.get_event_loop().run_until_complete(main_handler())

    def step(self, action):
        s = self._get_state()
        r = -1 if self.terminal else 0
        return s, r, self.terminal

    def init(self):

    def reset(self):

    def sample_action(self):


    def _get_state(self):


async def main_handler():
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


# if __name__ == '__main__':
#     asyncio.get_event_loop().run_until_complete(main_cycle())
# https://github.com/aaugustin/websockets/blob/master/example/client.py
# https://pypi.org/project/websocket_client/
# https://stackoverflow.com/questions/3142705/is-there-a-websocket-client-implemented-for-python