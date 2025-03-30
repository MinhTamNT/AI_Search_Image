import numpy as np
import cv2
import faiss
import redis
import pickle
import matplotlib.pyplot as plt
from keras.api.preprocessing.image import img_to_array
from keras.api.applications.efficientnet import EfficientNetB7, preprocess_input
from keras.api.models import Model
from sklearn.preprocessing import normalize
from sqlalchemy.orm import sessionmaker
from model import Image, engine
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

def random_level(assign_probas, rng):
    f = rng.uniform()
    for level in range(len(assign_probas)):
        if f < assign_probas[level]:
            return level
        f -= assign_probas[level]
    return len(assign_probas) - 1

def set_default_probas(M: int, m_L: float):
    assign_probas = []
    level = 0
    while True:
        proba = np.exp(-level / m_L) * (1 - np.exp(-1 / m_L))
        if proba < 1e-9: break
        assign_probas.append(proba)
        level += 1
    return assign_probas


def load_embeddings():
    cached_embeddings = redis_client.get('embeddings')
    cached_image_paths = redis_client.get('image_paths')

    if cached_embeddings and cached_image_paths:
        return pickle.loads(cached_embeddings), pickle.loads(cached_image_paths)

    images = session.query(Image).all()

    if not images:
        raise ValueError("không tìm thấy dữ liệu trong database!")

    embeddings = np.vstack([image.embedding for image in images])
    image_paths = np.array([image.image_path for image in images])

    embeddings = normalize(embeddings, axis=1, norm='l2')
    redis_client.set('embeddings', pickle.dumps(embeddings))
    redis_client.set('image_paths', pickle.dumps(image_paths))

    print("✅ Dữ liệu đã được lưu vào Redis.")
    return embeddings, image_paths

def get_image_embedding(img_path, model):
    img = preprocess_image(img_path)
    embedding = model.predict(img).flatten()
    return normalize(embedding.reshape(1, -1), axis=1, norm='l2')

def search_similar_images_hnsw(embedding, train_embeddings, k=5, M=16, efSearch=100, m_L=1.2):
    d = train_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = efSearch
    assign_probas = set_default_probas(M, m_L)
    rng = np.random.default_rng()
    for vector in train_embeddings:
        level = random_level(assign_probas, rng)
        index.add(vector.reshape(1, -1))
    distances, indices = index.search(embedding.reshape(1, -1), k)
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

def main ():
    model = get_extract_model()
    train_embeddings, train_image_paths = load_embeddings()
    img_path = "huong-dan-chup-anh-the-thao-13.jpg"
    embedding = get_image_embedding(img_path, model)
    distances, indices = search_similar_images_hnsw(embedding, train_embeddings)
    display_results(img_path, train_image_paths, indices, distances)

if __name__ == "__main__":
    main()
