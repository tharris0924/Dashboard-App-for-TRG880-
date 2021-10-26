import io
import pathlib
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import load_model

import pickle


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


st.title('Fruit360 Classification Task')
model_dir = "/home/tristan/Desktop/CodingProjects/trg880/"
model_dict = {
    'SVM (Experimental)': "SVM_finalized_model.sav",
    "Convoluted Pool CNN": "20ep_conv_pool_cnn",
    "ALL CNN": "20ep_all_cnn",
    "NIN CNN": "20ep_nin_cnn",
    "Ensemble CNN": "all"
}
option = st.selectbox(
    'What model would you like to use?',
    model_dict)

st.write('You selected:', option)

st.cache()


def load_data():
    # data_dir = pathlib.Path("/home/tristan/Desktop/CodingProjects/trg880/capstone/data/Training")
    class_names = load_obj('class_names')
    # class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))

    return class_names


def load_ensemble():
    nin_cnn_model = load_model('20ep_nin_cnn')
    all_cnn_model = load_model('20ep_all_cnn')
    conv_pool_cnn_model = load_model('20ep_conv_pool_cnn')
    models = [conv_pool_cnn_model, all_cnn_model, nin_cnn_model]
    return models


class_names = load_data()
if model_dict[option] != "SVM_finalized_model.sav":
    if model_dict[option] != "all":
        cnn = load_model(model_dict[option])
    else:
        ensembles = load_ensemble()

# img = st.file_uploader("Choose a fruit or veg", type="jpg")
st.set_option('deprecation.showfileUploaderEncoding', False)
buffer = st.file_uploader("Choose a fruit or veg", type=["png", "jpg"])


def pred_svm(input_image):
    buffer = input_image
    # img = Image.open(io.BytesIO(buffer.getvalue()))
    img = imread(buffer)
    img_resized = resize(img, (100, 100, 3))
    # img = img.convert('RGB')
    # img = img.resize((100, 100), Image.NEAREST)
    # # img_ = imread(img)
    l = [img_resized.flatten()]
    loaded_model = pickle.load(open(model_dict[option], 'rb'))
    # dt = pathlib.Path("/home/tristan/Desktop/CodingProjects/trg880/capstone/data/Training")
    # ds = tf.keras.utils.image_dataset_from_directory(
    #     dt)
    fruit = class_names
    probability = loaded_model.predict_proba(l)
    st.write(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(fruit[np.argmax(probability)], 100 * np.max(probability))
    )


def pred(input_image):
    img_height = 100
    img_width = 100
    buffer = input_image
    img = Image.open(io.BytesIO(buffer.getvalue()))
    img = img.convert('RGB')
    img = img.resize((img_height, img_width), Image.NEAREST)
    img_array = tf.keras.utils.img_to_array(img)
    # imgs = load_img(input_image, target_size=(img_height, img_width, 3))
    # img_array = tf.keras.utils.img_to_array(imgs)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = cnn.predict(img_array)
    score = predictions[0]
    print(np.argmax(score))
    st.write((
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    ))


def mostFrequent(arr, n):
    # Sort the array
    arr.sort()

    # find the max frequency using
    # linear traversal
    max_count = 1;
    res = arr[0];
    curr_count = 1

    for i in range(1, n):
        if (arr[i] == arr[i - 1]):
            curr_count += 1

        else:
            if (curr_count > max_count):
                max_count = curr_count
                res = arr[i - 1]

            curr_count = 1

    # If last element is most frequent
    if (curr_count > max_count):
        max_count = curr_count
        res = arr[n - 1]

    return res


def pred_ensembles(input_image):
    img_height = 100
    img_width = 100
    buffer = input_image
    img = Image.open(io.BytesIO(buffer.getvalue()))
    img = img.convert('RGB')
    img = img.resize((img_height, img_width), Image.NEAREST)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # predictions
    results = np.zeros((3, 131))
    for j in range(3):
        results += ensembles[j].predict(img_array)[0]
    score1 = np.argmax(results[0])
    score2 = np.argmax(results[1])
    score3 = np.argmax(results[2])
    scores = [score1, score2, score3]

    print(scores)
    n = len(scores)
    item = mostFrequent(scores, n)
    item = int(item)
    obj = class_names[item]
    # print(results)
    st.write(
        "Based on an ensemble apporach, the image belongs to {}"
            .format(obj)
    )


if buffer is not None:
    st.image(buffer)
    if model_dict[option] != "all" and model_dict[option] != "SVM_finalized_model.sav":
        pred(buffer)
    elif model_dict[option] == "SVM_finalized_model.sav":
        pred_svm(buffer)
    else:
        pred_ensembles(buffer)
