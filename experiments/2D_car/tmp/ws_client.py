import websocket
try:
    import thread
except ImportError:
    import _thread as thread
stop = False
play = False
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


if __name__ == "__main__":
    # websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://localhost:9001",
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close)
    ws.on_open = on_open
    ws.run_forever()

#https://www.sewio.net/wp-content/uploads/productlist/RTLSTDOA/integration/Example_WebSocketPython.pdf
