import websocket
try:
    import thread
except ImportError:
    import _thread as thread


class RemoteCarEnv(object):
    n_sensor = 5
    action_dim = 1
    state_dim = n_sensor
    action_bound = [-1, 1]
    ws = 0
    stop = False
    play = False

    init_done = False
    reset_done = False
    sample_action = None
    env_state = None
    env_reward = None
    env_done = None

    def __init__(self):
        self.ws = websocket.WebSocketApp("ws://localhost:9001",
                                  on_message = self.on_message,
                                  on_error = self.on_error,
                                  on_close = self.on_close)
        self.ws.on_open = self.on_open
        # self.ws.run_forever()

    def init(self):
        self.init_done = False
        print("send init")
        self.ws.send("init")

    def init_done_handler(self, arg_str):
        arg_data_str = arg_str.split(',')
        arr_str = np.array(arg_data_str)
        arr_float = arr_str.astype(np.float)
        self.env_state = arr_float
        self.init_done = True

    def reset(self):
        self.reset_done = False
        print("send reset")
        self.ws.send("reset")

    def reset_done_handler(self, arg_str):
        arg_data_str = arg_str.split(',')
        arr_str = np.array(arg_data_str)
        arr_float = arr_str.astype(np.float)
        self.env_state = arr_float
        self.reset_done = True

    def step(self, action):
        self.step_done = False
        message = "step:"
        for num in action:
            message += str(num) + ','
        print("send %s" % str(message[:-1]))
        self.ws.send(message[:-1])

    def step_done_handler(self, arg_str):
        arg_data_str = arg_str.split(',')
        arr_str = np.array(arg_data_str)
        arr_float = arr_str.astype(np.float)
        self.env_state = arr_float[:state_dim]
        self.env_reward = arr_float[state_dim]
        self.env_done = arr_float[-1]
        self.step_done = True

    def mess_selector(message):
        args = message.split(':')
        switcher = { 
            "init_done": init_done_handler,
            "reset_done": reset_done_handler,
            "step_done": step_done_handler,
            "nn_learn": nn_learn_handler,
            "stop": stop_handler
        } 
        return switcher.get(args[0], unknown_state_handler), args[1]

    def on_message(self, message):
        messHandler, message_data = self.mess_selector(message)
        self.messHandler(message_data)

    def on_error(self, error):
        print("### on_error ###")
        print(error)

    def on_close(self):
        print("### on_close ###")
        ws.close()
        print("on_close ... ws.close()")

    def on_open(self):
        print("### on_open ###")
        def run(*args):
            self.ws.run_forever()
            print("thread terminating...")
            ws.close()

        thread.start_new_thread(run, ())

    # def main_handler():
    # recv_data_str = ''

    #     while True:
    #         env.render()
    #         done_mess = 0
    #         if recv_data_str == 'reset':
    #             s = env.reset()
    #             r = 0
    #             done = False
    #             done_mess = 0
    #         else:
    #             # print("--------------env.step(action) action = %s" % str(action))
    #             s, r, done = env.step(action)
    #             if done:
    #                 s = env.reset()
    #                 done_mess = 1
    #                 print("---------------env.reset() %s" % str(done))

    #             print("send reward: %s" % str(r))
    #             mess = str(r) + ',' + str(done_mess)
    #             await websocket.send(mess)


    #         state_as_string = ''
    #         for num in s:
    #             state_as_string += str(num) + ','
    #         print("send state: %s" % str(state_as_string[:-1]))
    #         await websocket.send(state_as_string[:-1])

    #         action = await websocket.recv()
    #         recv_data_str = str(action)
    #         if recv_data_str != 'reset':
    #             action = np.array([float(action)])
    #         print("receive action: %s" % recv_data_str)
