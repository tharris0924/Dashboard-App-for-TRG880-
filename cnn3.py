# %%
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
# import tensorflow_datasets as tfds
import pathlib
import pickle
import matplotlib.pyplot as plt


def save_obj(obj: object, name: str):
    """
    Save a data-type of interest as a .pkl file

    Args:
        obj (any): variable name of interest
        name (str): string-name for .pkl file
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name: str):
    """
    Load .pkl file from current working directory

    Args:
        name (str): name for .pkl file of interest

    Returns:
        [any]: unpacked .pkl file either in the form of a pd.DataFrame or list
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# %%


data_dir = pathlib.Path("/home/tristan/Desktop/CodingProjects/trg880/capstone/data/Training")
test_dir = pathlib.Path("/home/tristan/Desktop/CodingProjects/trg880/capstone/data/Test")

# %%

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# %%

apples = list(data_dir.glob('Apple Braeburn/*'))
im = PIL.Image.open(str(apples[0]))
im.show()

# %%

im = PIL.Image.open(str(apples[1]))
plt.show()
im.show()

# %%

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

test_count = len(list(test_dir.glob('*/*.jpg')))
print(test_count)

# %%

list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

list_ds_test = tf.data.Dataset.list_files(str(test_dir / '*/*'), shuffle=False)
list_ds_test = list_ds_test.shuffle(test_count, reshuffle_each_iteration=False)

# %%

for f in list_ds.take(131):
    print(f.numpy())

for f in list_ds_test.take(131):
    print(f.numpy())

# %%

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

test_class_names = np.array(sorted([item.name for item in test_dir.glob('*') if item.name != "LICENSE.txt"]))
print(test_class_names)

# %%
save_obj(class_names,'class_names')

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

test_ds = list_ds_test.take(test_count)

train_ds
# %%

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

print(tf.data.experimental.cardinality(test_ds).numpy())

# %%
batch_size = 32
img_height = 100
img_width = 100


# %%


def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)


# %%

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


# %%

def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# %%

AUTOTUNE = tf.data.AUTOTUNE
print(train_ds)
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(map_func=process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds1 = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# %%
for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())


# %%

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

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

# %%

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# %%

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# %%
model.save('cnn_ep10_selftrain')
# %%
# tf.data.experimental.save(train_ds, path=os.getcwd())
# %%
# tf.data.experimental.save(test_ds, path=os.getcwd())
# %%
# tf.data.experimental.save(test_ds1, path=os.getcwd())


# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

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
from tensorflow.keras.models import load_model

model = load_model('cnn_ep10_selftrain')
# %%
model.evaluate(test_ds)

# %%

# Predict the label of the test_images
pred = model.predict(test_ds)
pred = np.argmax(pred, axis=1)

# %%

# Map the label

labels = class_names
pred_labels = [labels[k] for k in pred]

# Display the result
print(f'The first 5 predictions: {pred[:5]}')

# %%


frorveg_path = pathlib.Path("/home/tristan/Desktop/CodingProjects/trg880/capstone/data/Test")
frorveg_choice = class_names
print(frorveg_choice)
idxClass = (class_names == 'Apple Braeburn').nonzero()
apples = list(data_dir.glob('Apple Braeburn/*'))

# %%
img = tf.keras.utils.load_img(
    apples[0], target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
)
