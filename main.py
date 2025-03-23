import os
import numpy as np
import cv2
import faiss
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

from sklearn.preprocessing import normalize


# Load EfficientNetB7 Model (Feature Extractor)
def get_extract_model():
    base_model = EfficientNetB7(weights='imagenet', include_top=False, pooling='avg')
    return Model(inputs=base_model.input, outputs=base_model.output)


# Fine-tune EfficientNetB7 Model
def get_fine_tuned_model(num_classes):
    base_model = EfficientNetB7(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(x)  # Dropout để tránh overfitting
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model

def random_level(assign_probas, rng):
    f = rng.uniform()  # Giá trị random [0,1]
    for level in range(len(assign_probas)):
        if f < assign_probas[level]:
            return level
        f -= assign_probas[level]
    return len(assign_probas) - 1


def set_default_probas(M: int, m_L: float):
    nn = 0
    cum_nneighbor_per_level = []
    level = 0
    assign_probas = []

    while True:
        proba = np.exp(-level / m_L) * (1 - np.exp(-1 / m_L))
        if proba < 1e-9: break
        assign_probas.append(proba)
        nn += M * 2 if level == 0 else M
        cum_nneighbor_per_level.append(nn)
        level += 1

    return assign_probas, cum_nneighbor_per_level

# Image Preprocessing
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image '{img_path}' not found.")
    img = cv2.resize(img, (244, 244))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# Load Precomputed Embeddings
def load_embeddings():
    train_embeddings = np.load("train_embeddings.npy")
    train_image_paths = np.load("train_image_paths.npy")
    return normalize(train_embeddings, axis=1, norm='l2'), train_image_paths


# Extract Image Embedding
def get_image_embedding(img_path, model):

    img = preprocess_image(img_path)
    embedding = model.predict(img).flatten()
    return normalize(embedding.reshape(1, -1), axis=1, norm='l2')


# Image Search using Faiss IVF-PQ
def search_similar_images_hnsw(embedding, train_embeddings, k=5, M=32, m_L=1.5):
    d = train_embeddings.shape[1]  # Kích thước vector

    assign_probas, _ = set_default_probas(M, m_L)

    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 50

    rng = np.random.default_rng()
    for vector in train_embeddings:
        index.add(vector.reshape(1, -1))

    distances, indices = index.search(embedding.reshape(1, -1), k)

    return distances, indices


# Display Results
def display_results(query_img_path, train_image_paths, indices, distances, k=5):
    fig, axes = plt.subplots(1, k + 1, figsize=(15, 5))
    query_img = cv2.imread(query_img_path)
    axes[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Query Image")
    axes[0].axis("off")

    for i in range(k):
        img_path = train_image_paths[indices[0][i]]
        img = cv2.imread(img_path)
        axes[i + 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(f"Top {i + 1}")
        axes[i + 1].axis("off")

    plt.show()


# Main
extract_model = get_extract_model()
train_embeddings, train_image_paths = load_embeddings()
img_path = "huong-dan-chup-anh-the-thao-13.jpg"
embedding = get_image_embedding(img_path, extract_model)
distances, indices = search_similar_images_hnsw(embedding, train_embeddings)
display_results(img_path, train_image_paths, indices, distances)

