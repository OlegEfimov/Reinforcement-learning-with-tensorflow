import websocket
import threading
import numpy as np
import ast
# try:
#     import thread
# except ImportError:
#     import _thread as thread
# def tmp_init_done_handler(arg_str):
#     print("RemoteCarEnv - tmp_init_done_handler")
#     arg_data_str = arg_str.split(',')
#     np.array(arg_data_str)


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
    step_done = False
    sample_action = None
    env_state = None
    env_reward = None
    env_done = None
    init_done_handler0 = None
    reset_done_handler0 = None
    step_done_handler0 = None
    # stop_handler0 = None



    def __init__(self):
        self.ws = websocket.WebSocketApp("ws://localhost:9001",
                                  on_message = self.on_message,
                                  on_error = self.on_error,
                                  on_close = self.on_close)
        self.ws.on_open = self.on_open

        self.init_done_handler0 = self.init_done_handler
        self.reset_done_handler0 = self.reset_done_handler
        self.step_done_handler0 = self.step_done_handler
        # self.stop_handler0 = self.stop_handler


        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
        # self.ws.run_forever

    def init(self):
        print("RemoteCarEnv - init")
        self.init_done = False
        print("send init")
        self.ws.send("init:0")

    def init_done_handler(self, arg_str):
        print("RemoteCarEnv - init_done_handler")
        arg_data_str = arg_str.split(',')
        arr_str = np.array(arg_data_str)
        arr_float = arr_str.astype(np.float)
        self.env_state = arr_float
        self.init_done = True

    def reset(self):
        print("RemoteCarEnv - reset")
        self.reset_done = False
        print("send reset")
        self.ws.send("reset:0")

    def reset_done_handler(self, arg_str):
        print("RemoteCarEnv - reset_done_handler")
        arg_data_str = arg_str.split(',')
        arr_str = np.array(arg_data_str)
        arr_float = arr_str.astype(np.float)
        self.env_state = arr_float
        self.reset_done = True

    def step(self, action):
        print("RemoteCarEnv - step")
        self.step_done = False
        message = "step:"
        for num in action:
            message += str(num) + ','
        print("send %s" % str(message[:-1]))
        self.ws.send(message[:-1])

    def step_done_handler(self, arg_str):
        print("RemoteCarEnv - step_done_handler")
        arg_data_str = arg_str.split(',')

        state_str0 = arg_data_str[:self.state_dim]
        reward_str0 = arg_data_str[self.state_dim]
        step_done_str0 = arg_data_str[-1]

        arr_state_str = np.array(state_str0)
        arr_state_float = arr_state_str.astype(np.float)
        self.env_state = arr_state_float
        reward_float = float(reward_str0)
        self.env_reward = reward_float
        self.env_done = ast.literal_eval(step_done_str0)
        self.step_done = True


        # arr_str = np.array(arg_data_str)
        # arr_float = arr_str.astype(np.float)
        # self.env_state = arr_float[:state_dim]
        # self.env_reward = arr_float[state_dim]
        # self.env_done = arr_float[-1]
        # self.step_done = True

    def unknown_state_handler(self, arg_str):
        print("RemoteCarEnv - unknown_state_handler")

    def mess_selector(self, message):
        print("RemoteCarEnv - mess_selector")
        args = message.split(':')
        switcher = { 
            # "init_done": tmp_init_done_handler
            "init_done": self.init_done_handler0,
            "reset_done": self.reset_done_handler0,
            "step_done": self.step_done_handler0
            # "stop": self.stop_handler0
        }
        print("RemoteCarEnv - mess_selector 2")
        tmp555 = switcher.get(args[0], self.unknown_state_handler)
        print("RemoteCarEnv - mess_selector 3")
        switcher.get(args[0], self.unknown_state_handler)(args[1])
        print("RemoteCarEnv - mess_selector 4")

    def on_message(self, message):
        print("RemoteCarEnv - on_message")
        # messHandler, message_data = self.mess_selector(message)
        # print("RemoteCarEnv - on_message ---2")
        # messHandler(self, message_data)
        self.mess_selector(message)
        print("RemoteCarEnv - on_message 2")

    def on_error(self, error):
        print("RemoteCarEnv - on_error")
        print(error)

    def on_close(self):
        print("RemoteCarEnv - on_close")
        ws.close()
        print("on_close ... ws.close()")

    def on_open(self):
        print("RemoteCarEnv - on_open")
        # def run(*args):
        #     self.ws.run_forever()
        #     print("thread terminating...")
        #     ws.close()

        # thread.start_new_thread(run, ())


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
