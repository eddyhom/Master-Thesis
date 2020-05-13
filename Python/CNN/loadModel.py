import tensorflow as tf

DATA_DIR = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/Training"
DEST_DIR = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/Python/Examples/model.hdf5"
IMG_SIZE = 160
BATCH_SIZE = 32

model = tf.keras.models.load_model(DEST_DIR)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                                                zoom_range=0.2, horizontal_flip=True,
                                                                validation_split=0.2)  # set validation split

train_generator = train_datagen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='binary', subset='training')  # set as training data

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,  # same directory as training data
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='binary', subset='validation')  # set as validation data

loss0, accuracy0 = model.evaluate_generator(validation_generator, steps=50, verbose=1)
print("INITIAL ACCURACY: {:.2f}".format(accuracy0))
print("INITIAL LOSS: {:.2f}".format(loss0))