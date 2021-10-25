import tensorflow as tf
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.datasets import cifar10
import keras
from keras.engine import training
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average, Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model, load_model
from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL


import PIL.Image
import pathlib

# %%

data_dir = pathlib.Path("/home/tristan/Desktop/CodingProjects/trg880/capstone/data/Training")
test_dir = pathlib.Path("/home/tristan/Desktop/CodingProjects/trg880/capstone/data/Test")

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

apples = list(data_dir.glob('Apple Braeburn/*'))
im = PIL.Image.open(str(apples[0]))
im.show()

im = PIL.Image.open(str(apples[1]))
plt.show()
im.show()

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

test_count = len(list(test_dir.glob('*/*.jpg')))
print(test_count)

list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

list_ds_test = tf.data.Dataset.list_files(str(test_dir / '*/*'), shuffle=False)
list_ds_test = list_ds_test.shuffle(test_count, reshuffle_each_iteration=False)
#%%
for f in list_ds.take(131):
    print(f.numpy())

for f in list_ds_test.take(131):
    print(f.numpy())

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

test_class_names = np.array(sorted([item.name for item in test_dir.glob('*') if item.name != "LICENSE.txt"]))
print(test_class_names)

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)
test_ds = list_ds_test.take(test_count)
train_ds

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

print(tf.data.experimental.cardinality(test_ds).numpy())

batch_size = 32
img_height = 100
img_width = 100


def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


AUTOTUNE = tf.data.AUTOTUNE
print(train_ds)
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(map_func=process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds1 = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)

# %%

image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.title(class_names[label])
    plt.axis("off")

plt.show()

# %%
num_classes = 131
conv_pool_cnn_model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (1, 1), activation='relu'),
    tf.keras.layers.Conv2D(131, (1, 1)),
    tf.keras.layers.GlobalAveragePooling2D(),
    Activation(activation='softmax')
], name='conv_pool_cnn_model')

# %%
conv_pool_cnn_model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# %%
conv_pool_cnn_model.summary()
# %%

filepath = 'capstone/weights/' + conv_pool_cnn_model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                             save_best_only=True, mode='auto', save_freq=1)
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0)
# %%
epochs = 10
history = conv_pool_cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint, tensor_board],
    verbose=1
)

weight_files = glob.glob(os.path.join(os.getcwd(), 'capstone/weights/*'))
weight_file = max(weight_files, key=os.path.getctime)
# %%

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.title('CONV POOL CNN')
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %%


CONV_POOL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'capstone/weights', 'conv_pool_cnn_model.hdf5')


def evaluate_error(model: training.Model) -> np.float64:
    pred = model.predict(test_ds, batch_size=32)
    pred = np.argmax(pred, axis=1)
    pred = np.expand_dims(pred, axis=1)  # make same shape as y_test
    error = np.sum(np.not_equal(pred, test_ds)) / test_ds.shape[0]
    return error


# try:
#     conv_pool_cnn_weight_file
# except NameError:
conv_pool_cnn_model.load_weights(CONV_POOL_CNN_WEIGHT_FILE)
# conv_pool_cnn_model.evaluate(test_ds)

# evaluate_error(conv_pool_cnn_model)

# %%

frorveg_path = pathlib.Path("/home/tristan/Desktop/CodingProjects/trg880/capstone/data/Test")
frorveg_choice = class_names
print(frorveg_choice)
idxClass = (class_names == 'Passion Fruit').nonzero()
apples = list(data_dir.glob('Passion Fruit/*'))

# %%
img = tf.keras.utils.load_img(
    apples[0], target_size=(img_height, img_width, 3)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = conv_pool_cnn_model.predict(img_array)

score = predictions[0]
print(score)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
)
# %%
del conv_pool_cnn_model
# %% all cnn

all_cnn = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', strides=2),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', strides=2),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same'),
    Conv2D(131, kernel_size=(1, 1)),
    GlobalAveragePooling2D(),
    Activation(activation='softmax')
], name='all_cnn')
# %%

all_cnn.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# %%

filepath = 'capstone/weights/' + all_cnn.name + '.{epoch:02d}-{loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                             save_best_only=True, mode='auto', save_freq=1)
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0)
# %%
epochs = 10
history = all_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint, tensor_board],
    verbose=1
)

# %%
ALL_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'capstone/weights', 'all_cnn_model_weights.hdf5')
all_cnn.load_weights(ALL_CNN_WEIGHT_FILE)
# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.title('ALL CNN')
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %%
frorveg_path = pathlib.Path("/home/tristan/Desktop/CodingProjects/trg880/capstone/data/Test")
frorveg_choice = class_names
print(frorveg_choice)
idxClass = (class_names == 'Passion Fruit').nonzero()
apples = list(data_dir.glob('Passion Fruit/*'))

# %%
img = tf.keras.utils.load_img(
    apples[0], target_size=(img_height, img_width, 3)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = all_cnn.predict(img_array)

score = predictions[0]
print(score)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# %%

nin_cnn = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    # mlpconv block 1
    Conv2D(32, (5, 5), activation='relu', padding='valid'),
    Conv2D(32, (1, 1), activation='relu'),
    Conv2D(32, (1, 1), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    # mlpconv block2
    Conv2D(64, (3, 3), activation='relu', padding='valid'),
    Conv2D(64, (1, 1), activation='relu'),
    Conv2D(64, (1, 1), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    # mlpconv block3
    Conv2D(128, (3, 3), activation='relu', padding='valid'),
    Conv2D(32, (1, 1), activation='relu'),
    Conv2D(131, (1, 1)),

    GlobalAveragePooling2D(),
    Activation(activation='softmax')
], name='nin_cnn')
# %%
nin_cnn.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# %%
NIN_CNN_WEIGHT_FILE = os.path.join(os.getcwd(), 'capstone/weights', 'nin_cnn_model_weights.hdf5')
nin_cnn.load_weights(NIN_CNN_WEIGHT_FILE)
# %%

filepath = 'capstone/weights/' + nin_cnn.name + '.{epoch:02d}-{loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                             save_best_only=True, mode='auto', save_freq=1)
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0)

epochs = 10
history = nin_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint, tensor_board],
    verbose=1
)

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.title('NIN CNN')
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %%
frorveg_path = pathlib.Path("/home/tristan/Desktop/CodingProjects/trg880/capstone/data/Test")
frorveg_choice = class_names
print(frorveg_choice)
idxClass = (class_names == 'Banana Lady Finger').nonzero()
apples = list(data_dir.glob('Banana Lady Finger/*'))

# %%
img = tf.keras.utils.load_img(
    apples[0], target_size=(img_height, img_width, 3)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = nin_cnn.predict(img_array)

score = predictions[0]
print(score)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# %%
models = [conv_pool_cnn_model, all_cnn, nin_cnn]
# def ensemble(models: List[training.Model], model_input: Tensor) -> training.Model:
#     outputs = [models.outputs[0] for model in models]
#
#
#     return model

# %%
pred = [model.predict(img_array)[0] for model in models]
y = Average()(pred)
avg = y.numpy().tolist()
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(avg)], 100 * np.max(avg))
)
# %%
nin_ev = nin_cnn.evaluate(test_ds)
# %%
all_ev = all_cnn.evaluate(test_ds)
# %%
conv_ev = conv_pool_cnn_model.evaluate(test_ds)
# %%
metrics = np.array([nin_ev, conv_ev, all_ev])
np.average(metrics, axis=0)

# %%
input_shape = (img_height, img_width, 3)
print(input_shape)
model_input = Input(shape=input_shape)


# %%
def conv_pool_cnn(model_input: Tensor) -> training.Model:
    x = tf.keras.layers.Rescaling(1. / 255)(model_input)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(131, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)

    model = Model(model_input, x, name='conv_pool_cnn')

    return model


# %%

conv_pool_cnn_model = conv_pool_cnn(model_input)

# %%

NUM_EPOCHS = 20


# %%

def compile_and_train(model: training.Model, num_epochs: int) -> Tuple[History, str]:
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['acc'])
    filepath = 'capstone/weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True,
                                 save_best_only=True, mode='auto', save_freq=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0)
    history = model.fit(x=train_ds, validation_data=val_ds, batch_size=32,
                        epochs=num_epochs, verbose=1, callbacks=[checkpoint, tensor_board])
    weight_files = glob.glob(os.path.join(os.getcwd(), 'capstone/weights/*'))
    weight_file = max(weight_files, key=os.path.getctime)  # most recent file
    return history, weight_file


# %%
history, conv_pool_cnn_weight_file = compile_and_train(conv_pool_cnn_model, NUM_EPOCHS)

# %%

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(NUM_EPOCHS)

plt.title('NIN CNN')
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %%
conv_pool_cnn_model.save('20ep_conv_pool_cnn')


# %%
def all_cnn(model_input: Tensor) -> training.Model:
    x = tf.keras.layers.Rescaling(1. / 255)(model_input)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(131, (1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)

    model = Model(model_input, x, name='all_cnn')

    return model


# %%
all_cnn_model = all_cnn(model_input)

# %%

history, all_cnn_weight_file = compile_and_train(all_cnn_model, NUM_EPOCHS)

# %%
all_cnn_model.save('20ep_all_cnn')
# %%

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(NUM_EPOCHS)

plt.title('NIN CNN')
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# %%

def nin_cnn(model_input: Tensor) -> training.Model:
    x = tf.keras.layers.Rescaling(1. / 255)(model_input)
    # mlpconv block 1
    x = Conv2D(32, (5, 5), activation='relu', padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    # mlpconv block2
    x = Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)

    # mlpconv block3
    x = Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(131, (1, 1))(x)

    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)

    model = Model(model_input, x, name='nin_cnn')

    return model


# %%
nin_cnn_model = nin_cnn(model_input)

# %%

history, nin_cnn_weight_file = compile_and_train(nin_cnn_model, NUM_EPOCHS)

# %%
nin_cnn_model.save('20ep_nin_cnn')
# %%

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(NUM_EPOCHS)

plt.title('NIN CNN')
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %%
nin_cnn_model = load_model('20ep_nin_cnn')
all_cnn_model = load_model('20ep_all_cnn')
conv_pool_cnn_model = load_model('20ep_conv_pool_cnn')
# %%

nin_cnn_weight_file = nin_cnn_model.weights
all_cnn_weight_file = all_cnn_model.weights
conv_pool_cnn_weight_file = conv_pool_cnn_model.weights

# %%
models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model]
outputs = [model.weights[0] for model in models]
y = Average()(outputs)
model = Model(model_input, y, name='ensemble')
# %%

def ensemble(models: List[training.Model], model_input: Tensor) -> training.Model:
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')

    return model


#%%

pred = [model.predict(img_array)[0] for model in models]
y = Average()(pred)
avg = y.numpy().tolist()
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(avg)], 100 * np.max(avg))
)

# %%
nin_ev = nin_cnn_model.evaluate(test_ds)
# %%
all_ev = all_cnn_model.evaluate(test_ds)
# %%
conv_ev = conv_pool_cnn_model.evaluate(test_ds)
# %%
metrics = np.array([nin_ev, conv_ev, all_ev])
np.average(metrics, axis=0)


# %%
frorveg_path = pathlib.Path("/home/tristan/Desktop/CodingProjects/trg880/capstone/data/Test")
frorveg_choice = class_names
print(frorveg_choice)
idxClass = (class_names == 'Banana Lady Finger').nonzero()
apples = list(data_dir.glob('Banana Lady Finger/*'))

# %%
img = tf.keras.utils.load_img(
    apples[0], target_size=(img_height, img_width, 3)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = nin_cnn_model.predict(img_array)

score = predictions[0]
print(score)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
)


#%%

