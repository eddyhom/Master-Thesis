import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# CHANGE DATA_DIR TO YOUR OWN DIRECTORY INCLUDING THE TRAINING DATA
DATA_DIR = "C:\\Users\\eddy_\\Desktop\\Master-Thesis\\DB\\TrainingGray"
DATAVAL_DIR = "C:\\Users\\eddy_\\Desktop\\Master-Thesis\\DB\\ValidationGray"
MODEL_VERSION = "model_2.11"
DEST_DIR = "Models/" + MODEL_VERSION + ".hdf5"
IMG_SIZE = 70
BATCH_SIZE = 32
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)
LEARNING_RATE = 0.001
INITIAL_EPOCHS = 80
VALIDATION_STEPS = 20
RATE_DECAY = 247
VAL_FREQ = 10

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                                                zoom_range=0.2, horizontal_flip=True,
                                                                validation_split=None)  # set validation split

train_generator = train_datagen.flow_from_directory(
    DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale',
    batch_size=BATCH_SIZE, class_mode='binary',
    shuffle=True)  # set as training data

validation_generator = train_datagen.flow_from_directory(
    DATAVAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale',  # same directory as training data
    batch_size=BATCH_SIZE, class_mode='binary',
    shuffle=True)  # set as validation data

# --Layer 1
conv2D_layer = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE)
# dropout_layer = tf.keras.layers.Dropout(0.1)#0.2
# normal_layer = tf.keras.layers.LayerNormalization()
maxPool_layer = tf.keras.layers.MaxPooling2D((2, 2))

# --Layer 2
conv2D_layer2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
# dropout_layer2 = tf.keras.layers.Dropout(0.1)#0.1
# normal_layer2 = tf.keras.layers.LayerNormalization()
maxPool_layer2 = tf.keras.layers.MaxPooling2D((2, 2))

# --Layer 3
conv2D_layer3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
# dropout_layer3 = tf.keras.layers.Dropout(0.2)
# normal_layer3 = tf.keras.layers.LayerNormalization()
maxPool_layer3 = tf.keras.layers.MaxPooling2D((2, 2))

# --Layer 4
conv2D_layer4 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
# dropout_layer4 = tf.keras.layers.Dropout(0.2)
# normal_layer4 = tf.keras.layers.LayerNormalization()

# --Layer 5
flat_layer = tf.keras.layers.Flatten()
prediction_layer1 = tf.keras.layers.Dense(32, activation='relu')

# --Layer 6
final_layer = tf.keras.layers.Dense(3, activation='softmax')

model = tf.keras.Sequential([conv2D_layer, maxPool_layer,
                             conv2D_layer2, maxPool_layer2,
                             conv2D_layer3, maxPool_layer3,
                             conv2D_layer4,
                             flat_layer, prediction_layer1, final_layer])

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    LEARNING_RATE, decay_steps=RATE_DECAY,
    decay_rate=0.7, staircase=False)

optimizer_TimeDecay = tf.keras.optimizers.Adam(lr_schedule)

early_stopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
cb_list = [early_stopper]

model.compile(optimizer=optimizer_TimeDecay,  # tf.keras.optimizers.RMSprop(lr=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',  # tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()  # SHOWS THE ARCHITECTURE OF THE CNN

history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              epochs=INITIAL_EPOCHS, callbacks=cb_list, verbose=1,
                              validation_freq=VAL_FREQ)

model.save(DEST_DIR, overwrite=True)

'''
test_loss, test_acc = model.evaluate(validation_generator, steps=62, verbose=1)
'''

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
f = np.arange(VAL_FREQ, (len(val_acc) + 1) * VAL_FREQ, VAL_FREQ)

plt.plot(acc, label="Training Accuracy")
plt.plot(f, val_acc, label="Validation Accuracy")
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Training and Validation Accuracy')

PLOT_NAME = "Plots/" + MODEL_VERSION + ".png"
plt.savefig(PLOT_NAME, transparent=True)

plt.show()

'''
arr = model.predict(validation_generator)

#model.save(DEST_DIR)

counter = 0
for prediction in arr:
    print("Prediction nr "+str(counter)+": ", end="")
    print("Negative: "+str(prediction[0])+" Neutral: "+str(prediction[1]) + " Positive: " + str(prediction[2]))
    counter += 1
'''
# print("INITIAL ACCURACY: {:.2f}".format(test_acc))
# print("INITIAL LOSS: {:.2f}".format(test_loss))
