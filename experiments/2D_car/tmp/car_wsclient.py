import websocket
import threading
import numpy as np

class WsClient(object):
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
        print("WsClient - unknown_state_handler")

    def on_message(self, message):
        print("recv %s" % str(message))
        args = message.split(':')
        switcher = { 
            "action": self.action_handler,
        }
        switcher.get(args[0], self.unknown_state_handler)(args[1])

    def on_error(self, error):
        print("WsClient - on_error")
        print(error)

    def on_close(self):
        print("WsClient - on_close")
        self.ws.close()
        print("WsClient ... ws.close()")

    def on_open(self):
        print("WsClient - on_open")

    def send(self, message):
        self.ws.send(message)
