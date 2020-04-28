import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

#CHANGE DATA_DIR TO YOUR OWN DIRECTORY INCLUDING THE TRAINING DATA
DATA_DIR = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/Training"
DEST_DIR = "model_1.2.hdf5"
IMG_SIZE = 70
BATCH_SIZE = 32
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
LEARNING_RATE = 0.0001
INITIAL_EPOCHS = 50
VALIDATION_STEPS = 20
RATE_DECAY = 400

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                                                zoom_range=0.2, horizontal_flip=True,
                                                                validation_split=0.2)  # set validation split

train_generator = train_datagen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='binary',
    subset='training')  # set as training data

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,  # same directory as training data
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='binary', subset='validation')  # set as validation data

conv2D_layer = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE)
maxPool_layer = tf.keras.layers.MaxPooling2D((2, 2))
conv2D_layer2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
maxPool_layer2 = tf.keras.layers.MaxPooling2D((2, 2))
conv2D_layer3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
flat_layer = tf.keras.layers.Flatten()
prediction_layer1 = tf.keras.layers.Dense(16, activation='relu')
final_layer = tf.keras.layers.Dense(1)

model = tf.keras.Sequential([conv2D_layer, maxPool_layer, conv2D_layer2, maxPool_layer2,
                             conv2D_layer3, flat_layer, prediction_layer1, final_layer])

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001, decay_steps=190*RATE_DECAY,
  decay_rate=1, staircase=False)

optimizer_TimeDecay = tf.keras.optimizers.Adam(lr_schedule)

early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
cb_list = [early_stopper]

model.compile(optimizer=optimizer_TimeDecay, #tf.keras.optimizers.RMSprop(lr=LEARNING_RATE),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary() #SHOWS THE ARCHITECTURE OF THE CNN

history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              epochs=INITIAL_EPOCHS, callbacks=cb_list, verbose=1)

model.save(DEST_DIR, overwrite=True)

test_loss, test_acc = model.evaluate(validation_generator, verbose=2)

print("INITIAL ACCURACY: {:.2f}".format(test_acc))
print("INITIAL LOSS: {:.2f}".format(test_loss))
