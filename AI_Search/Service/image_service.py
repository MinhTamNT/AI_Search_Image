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
from AI_Search.model import Image, engine
import  redis


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


def random_level(assign_probas: list, rng):
    f = rng.uniform()
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
        nn += M*2 if level == 0 else M
        cum_nneighbor_per_level.append(nn)
        level += 1
    return assign_probas, cum_nneighbor_per_level

def load_embeddings():
    embeddings_data = redis_client.get('embeddings')
    image_paths_data = redis_client.get('image_paths')

    if embeddings_data and image_paths_data:
        print("Loading data from Redis.")
        embeddings = pickle.loads(embeddings_data)
        image_paths = pickle.loads(image_paths_data)
        return embeddings, image_paths

    print("Redis empty. Loading data from database...")
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

    redis_client.setex('embeddings',86400 ,pickle.dumps(embeddings))
    redis_client.setex('image_paths', 86400, pickle.dumps(image_paths))
    print("Data saved to Redis.")

    return embeddings, image_paths



def get_image_embedding(img_path, model):
    img = preprocess_image(img_path)
    embedding = model.predict(img).flatten()
    return normalize(embedding.reshape(1, -1), axis=1, norm='l2')


def search_similar_images(embedding, train_embeddings, k=5):
    d = train_embeddings.shape[1]
    M = 32
    m_L = 1.5
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 200


    assign_probas, _ = set_default_probas(M, m_L)
    rng = np.random.default_rng()

    for vector in train_embeddings:
        level = random_level(assign_probas, rng)
        index.add(np.expand_dims(vector.astype(np.float32), axis=0))

    # Tìm kiếm tương tự
    embedding = embedding.astype(np.float32).reshape(1, -1)
    distances, indices = index.search(embedding, k)
    return distances, indices

