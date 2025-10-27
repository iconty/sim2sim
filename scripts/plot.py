import zmq
import json

context = zmq.Context()
serverSocket = context.socket(zmq.PUB)

port = 9872
serverSocket.bind("tcp://*:"+str(port))
def plot(data, name):
    d = {name : data.tolist()}
    serverSocket.send_string(json.dumps(d))