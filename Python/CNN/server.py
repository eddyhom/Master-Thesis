import socket
import tensorflow as tf
import cv2
import predictServer as predict

MODEL_DIR = "D:\\Users\\eddy_\\Documents\\Master-Thesis\\Python\\CNN\\Models\\model_2emotions_30.hdf5"
CASCADE_DIR = "D:\\Users\\eddy_\\Documents\\Master-Thesis\\Python\\Examples\\TrainedCNN\\haarcascade_frontalface_alt.xml"

#### INITIALIZE CASCADE AND CNN MODEL
face_cascade = cv2.CascadeClassifier(CASCADE_DIR)
model = tf.keras.models.load_model(MODEL_DIR)

#### TEST MODEL BY PREDICTING A TEST IMAGE
#### (First prediction takes longer, dont want this to happen in real time)
initialize_image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
emo = predict.predict(model, initialize_image)
#print("Initialized Model Predictor, returned: ", emo)

SERVER_IP = '192.168.1.33'
PORT = 1026
HEADER_SIZE = 10
c2 = 0

#### INITIALIZE SOCKET COMMUNICATION
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((SERVER_IP, PORT))
s.listen(5)
clt, adr = s.accept()

with clt:
    print(f"Connection to {adr} established")
    finish = False
    while True:  # Go until Client closes connection
        full_msg = b''
        new_msg = True
        while True:  # Go Until full message is received
            msg = clt.recv(1024)
            if len(msg) == 0:
                finish = True
                break
            if new_msg:
                # print(f"new message length: {msg[:HEADER_SIZE]}")
                msglen = int(msg[:HEADER_SIZE])
                new_msg = False
            full_msg += msg

            if len(full_msg) - HEADER_SIZE == msglen:
                # print("full msg rcvd")

                d = full_msg[HEADER_SIZE:]
                pic_path = 'GazeboPics\\emotion' + str(c2) + '.png'

                with open(pic_path, 'wb') as f:
                    f.write(d)
                f.close()

                print("Send data to predict")
                emotion = predict.isFace(face_cascade, model, pic_path)

                print("Image was: ", emotion)

                clt.send(bytes(str(emotion), "utf-8"))

                new_msg = True
                full_msg = b''
                c2 += 1
        if finish:
            break

    clt.send(bytes("THANK YOU FOR CONNECTING !", "utf-8"))

print(c2)
clt.close()
s.close()
