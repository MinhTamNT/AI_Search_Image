import os
import numpy as np
import pickle
import hashlib
import redis
import cv2
from dotenv import load_dotenv
from sklearn.preprocessing import normalize
from sqlalchemy.orm import sessionmaker
from keras.api.preprocessing.image import img_to_array
from keras.api.applications.efficientnet import EfficientNetB7, preprocess_input
from keras.api.models import Model
from AI_Search.model import Image, Tag, engine
import faiss
import json
load_dotenv()

# Redis setup
redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=False
)


Session = sessionmaker(bind=engine)
session = Session()


def clear_redis_cache():

    if redis_client is None:
        print("Redis chưa được cấu hình.")
        return

    keys = redis_client.keys("embeddings:*")
    if not keys:
        print("Không tìm thấy key nào để xóa.")
        return

    for key in keys:
        redis_client.delete(key)
        print(f"Đã xóa cache Redis: {key}")


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

def get_image_embedding(img_path, model):
    img = preprocess_image(img_path)
    embedding = model.predict(img).flatten()
    return normalize(embedding.reshape(1, -1), axis=1, norm='l2')

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


def generate_cache_key(filter_tags):
    if not filter_tags:
        return "embeddings:all"
    sorted_tags = sorted(filter_tags)
    tags_string = ",".join(sorted_tags)
    hash_key = hashlib.md5(tags_string.encode()).hexdigest()
    return f"embeddings:{hash_key}"

def load_embeddings(filter_tags=None):

    if filter_tags and isinstance(filter_tags, list) and filter_tags != []:
        if isinstance(filter_tags[0], str):
            try:
                decoded = json.loads(filter_tags[0])
                if isinstance(decoded, list):
                    filter_tags = decoded
            except json.JSONDecodeError:
                pass  # giữ nguyên nếu không giải mã được
        elif isinstance(filter_tags[0], Image):
            # Handle the case when filter_tags contains Image objects directly
            image_ids = [image.id for image in filter_tags]  # Example: Using image IDs to filter
            images = session.query(Image).filter(Image.id.in_(image_ids)).all()
            embeddings = [image.embedding for image in images if image.embedding]
            image_paths = [image.image_url for image in images]
            embeddings = np.vstack(embeddings)
            embeddings = normalize(embeddings, axis=1, norm="l2")
            return embeddings, image_paths

    print("Decoded filter_tags:", filter_tags)

    if filter_tags is None or filter_tags == []:
        filter_tags = []

    cache_key = generate_cache_key(filter_tags)
    cached_data = redis_client.get(cache_key)

    if cached_data:
        print("Load from Redis.")
        embeddings, image_paths = pickle.loads(cached_data)
        return embeddings, image_paths

    print("Load from DB...")
    if not filter_tags:
        images = session.query(Image).all()
    else:
        images = (
            session.query(Image)
            .join(Image.tags)
            .filter(Tag.tag_name.in_(filter_tags))
            .all()
        )

    embeddings = []
    image_paths = []

    for image in images:
        if image.embedding is not None:
            embeddings.append(image.embedding)
            image_paths.append(image.image_url)

    if not embeddings:
        raise ValueError("No matching images found with the given tags!")

    embeddings = np.vstack(embeddings)
    embeddings = normalize(embeddings, axis=1, norm="l2")

    redis_client.setex(cache_key, 3600, pickle.dumps((embeddings, image_paths)))

    print("Saved to Redis.")
    return embeddings, image_paths



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

    embedding = embedding.astype(np.float32).reshape(1, -1)
    distances, indices = index.search(embedding, k)
    return distances, indices