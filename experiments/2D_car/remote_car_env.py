import websocket
try:
    import thread
except ImportError:
    import _thread as thread


class RemoteCarEnv(object):
    n_sensor = 5
    action_dim = 1
    state_dim = n_sensor
    ws = 0
    stop = False
    play = False

    def __init__(self):
        self.ws = websocket.WebSocketApp("ws://localhost:9001",
                                  on_message = self.on_message,
                                  on_error = self.on_error,
                                  on_close = self.on_close)
        self.ws.on_open = self.on_open
        self.ws.run_forever()


    def on_message(ws, message):
        global stop
        global play
        if message == "stop":
            print("on_message - stop")
            stop = True
            play = False
        elif message == "continue":
            print("on_message - continue")
            play = True
        print(message)

    def on_error(ws, error):
        print(error)

    def on_close(ws):
        print("### closed ###")

    def on_open(ws):
        def run(*args):
            global stop
            global play
            play = True
            while stop != True:
                if play:
                    play = False
                    ws.send("client_send_mess")
            ws.close()
            print("thread terminating...")

        thread.start_new_thread(run, ())

    def init(self):
        self.ws.send("init")

    def reset(self):
        self.ws.send("reset")

    def step(self, action):
        mess = str(action[0]) + ',' + str(action[1])
        self.ws.send(mess)

    def main_handler():
    recv_data_str = ''

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

            action = await websocket.recv()
            recv_data_str = str(action)
            if recv_data_str != 'reset':
                action = np.array([float(action)])
            print("receive action: %s" % recv_data_str)
