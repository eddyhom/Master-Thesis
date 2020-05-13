import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

DATA_DIR = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/DataBases/Emotios Labelled/RAFdatabase/Training"
DEST_DIR = "/home/eddyhom/Documents/MasterThesis/Master-Thesis/Python/Examples/model_0.1.hdf5"
IMG_SIZE = 70
BATCH_SIZE = 32
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
LEARNING_RATE = 0.0001
INITIAL_EPOCHS = 50
VALIDATION_STEPS = 20
RATE_DECAY = 250

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                validation_split=0.2)  # set validation split

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training')  # set as training data

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,  # same directory as training data
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation')  # set as validation data

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001, decay_steps=190*RATE_DECAY,
  decay_rate=1, staircase=False)

optimizer_TimeDecay = tf.keras.optimizers.Adam(lr_schedule)
early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
cb_list = [early_stopper]

model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])

model.compile(optimizer=optimizer_TimeDecay,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

'''
loss0, accuracy0 = model.evaluate_generator(validation_generator, steps=64, verbose=1, callbacks=cb_list)
print("INITIAL ACCURACY: {:.2f}".format(accuracy0))
print("INITIAL LOSS: {:.2f}".format(loss0))
'''

history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              epochs=INITIAL_EPOCHS, callbacks=cb_list, verbose=1)

model.save(DEST_DIR, overwrite=True)


