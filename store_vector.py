import os
import numpy as np
import cv2
from sklearn.preprocessing import normalize
from keras.api.preprocessing import image
from keras.api.applications.efficientnet import EfficientNetB7, preprocess_input
from keras.api.models import Model
from dao import save_embeddings_and_comments_to_db

def get_extract_model():
    base_model = EfficientNetB7(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

def preprocess_image(img):
    img = cv2.resize(img, (600, 600))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def compute_embeddings(images, model):
    embeddings = []
    for img in images:
        img = preprocess_image(img)
        img_embedding = model.predict(img)
        embeddings.append(img_embedding.flatten())
    return normalize(np.array(embeddings), axis=1, norm='l2')

def load_images_from_folder(folder):
    images = []
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    image_paths.append(img_path)
    return images, image_paths

data_folder = 'flickr30k_images/flickr30k_images'
file_excel = 'results.csv'

model = get_extract_model()
train_embeddings = np.load('train_embeddings.npy')
image_paths = np.load('train_image_paths.npy')
image_paths = [str(path) for path in image_paths]

save_embeddings_and_comments_to_db(image_paths, train_embeddings, file_excel)