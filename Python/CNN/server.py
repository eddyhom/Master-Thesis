import socket
import tensorflow as tf
import predictServer as predict

MODEL_DIR = "D:\\Users\\eddy_\\Documents\\Master-Thesis\\Python\\CNN\\Models\\model_2emotions_34.hdf5"
model = tf.keras.models.load_model(MODEL_DIR)

PORT = 1026

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), PORT))
s.listen(5)
clt, adr = s.accept()


with clt:
    print(f"Connection to {adr} established")
    while True:
        print("IN WHILE TRUE")
        data = clt.recv(2048)
        if len(data) <= 0:
            print(len(data))
            break
        else:
            print("IN ELSE", len(data))
            with open('emotion.png', 'wb') as f:
                f.write(data)
            f.close()
            break

    print("Send data to predict")
    emotion = predict.predict(model)
    print("Emotion predicted was: ", emotion)

    clt.send(bytes(emotion, "utf-8"))
    clt.send(bytes("THANK YOU FOR CONNECTING !", "utf-8"))

clt.close()


'''while True:

    print(f"Connection to {adr} established")
    clt.send(bytes("Socket programming in python", "utf-8"))
    clt.close()'''