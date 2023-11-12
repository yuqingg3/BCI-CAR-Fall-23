# client.py

import socket
import time
import numpy as np

class Controller():
    def __init__(self, server_address, server_port):
        self.server_address = server_address
        self.server_port = server_port
        self.moving = False
        self.turning = False
        self.moving_time = 0
        self.MAX_MOVE_TIME = 10
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_address, self.server_port))
        print("Connected to server")

    def close(self):
        self.sock.close()
        print("Connection closed")

    def sendAction(self, action):
        actionInt = np.argmax(action)
        actionLabel = "S"

        """
        if actionInt == 1
        if actionInt == 2
        if actionInt == 3
        if actionInt == 4 then corresponding code
        """
    

        # Similar logic for other actions...

        if (self.moving and actionLabel == "S"):
            if (self.moving_time > self.MAX_MOVE_TIME):
                self.sendStopAction()
                self.moving = False
                self.moving_time = 0
            else:
                self.moving_time += 1
            return

        print("Sending %s" % actionLabel)
        self.sock.sendall(actionLabel.encode())
        print("Receiving message from server ...")
        res = self.sock.recv(1024).decode()
        if res:
            print("Server says: %s" % res)
        else:
            print("Server doesn't respond")

        if self.turning:
            time.sleep(0.5)
            self.sendStopAction()
            self.turning = False

        if self.moving:
            time.sleep(0.1)

    def sendStopAction(self):
        self.sock.sendall(b'S')
        print("Receiving message from server ...")
        res = self.sock.recv(1024).decode()
        if res:
            print("Server says: %s" % res)
        else:
            print("Server doesn't respond")

    def getSocket(self):
        return self.sock

if __name__ == "__main__":
    sample_server = 12345
    sample_host = "localhost"
    controller = Controller(sample_host, sample_server)
    controller.connect()

    # Example: Send an action
    controller.sendAction(np.array([0, 1, 0, 0, 0, 0]))

    controller.close()
