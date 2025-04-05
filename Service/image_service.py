import numpy as np
import cv2
import faiss
import pickle
import matplotlib.pyplot as plt
from keras.api.preprocessing.image import img_to_array
from keras.api.applications.efficientnet import EfficientNetB7, preprocess_input
from keras.api.models import Model
from sklearn.preprocessing import normalize
from sqlalchemy.orm import sessionmaker
from model import Image, engine
import redis


Session = sessionmaker(bind=engine)
session = Session()

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
extract_model = None
def get_extract_model():
    global extract_model
    if extract_model is None:
        base_model = EfficientNetB7(weights="imagenet", include_top=False, pooling="avg")
        extract_model = Model(inputs=base_model.input, outputs=base_model.output)
    return extract_model


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image '{img_path}' not found.")
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def load_embeddings():
    images = session.query(Image).all()
    if not images:
        raise ValueError("No images found in database!")

    embeddings = []
    image_paths = []
    for image in images:
        if image.embedding is not None:
            embeddings.append(image.embedding)
            image_paths.append(image.image_path)

    embeddings = np.vstack(embeddings)
    embeddings = normalize(embeddings, axis=1, norm='l2')

    redis_client.set('embeddings', pickle.dumps(embeddings))
    redis_client.set('image_paths', pickle.dumps(image_paths))
    print("Data saved to Redis.")
    return embeddings, image_paths


def get_image_embedding(img_path, model):
    img = preprocess_image(img_path)
    embedding = model.predict(img).flatten()
    return normalize(embedding.reshape(1, -1), axis=1, norm='l2')


def search_similar_images(embedding, train_embeddings, k=5):
    d = train_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(train_embeddings)
    distances, indices = index.search(embedding, k)
    return distances, indices


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