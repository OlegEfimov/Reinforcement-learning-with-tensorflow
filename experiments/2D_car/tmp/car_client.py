import websocket
import threading
import numpy as np
import ast


class RemoteNnClient(object):
    n_sensor = 5
    action_dim = 1
    state_dim = n_sensor
    action_bound = [-1, 1]
    ws = 0

    action_ready = False
    action = None

    def __init__(self):
        self.ws = websocket.WebSocketApp("ws://localhost:9001",
                                  on_message = self.on_message,
                                  on_error = self.on_error,
                                  on_close = self.on_close)
        self.ws.on_open = self.on_open

        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

    def action_handler(self, arg_str):
        arr_args = arg_str.split(',')
        arr_action_str = np.array(arr_args)
        arr_action_float = arr_action_str.astype(np.float)
        self.action = arr_action_float
        self.action_ready = True

    def unknown_state_handler(self, arg_str):
        print("RemoteCarEnv - unknown_state_handler")

    def mess_selector(self, message):
        print("RemoteCarEnv mess_selector message=%s" % str(message))
        args = message.split(':')
        switcher = { 
            "action": self.action_handler,
        }
        switcher.get(args[0], self.unknown_state_handler)(args[1])

    def on_message(self, message):
        self.mess_selector(message)

    def on_error(self, error):
        print("RemoteCarEnv - on_error")
        print(error)

    def on_close(self):
        print("RemoteCarEnv - on_close")
        self.ws.close()
        print("on_close ... ws.close()")

    def on_open(self):
        print("RemoteCarEnv - on_open")

    def send(self, message):
        print("RemoteCarEnv - send start")
        self.ws.send(message)
        print("RemoteCarEnv - send finished")
