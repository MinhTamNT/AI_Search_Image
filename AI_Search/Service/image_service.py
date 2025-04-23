import os
import numpy as np
import pickle
import hashlib
import redis
import cv2
import json
import faiss
import logging
from functools import lru_cache
from typing import List, Tuple, Optional, Union
from dotenv import load_dotenv
from sklearn.preprocessing import normalize
from sqlalchemy.orm import sessionmaker, scoped_session
from keras.api.preprocessing.image import img_to_array
from keras.api.applications.efficientnet import EfficientNetB7, preprocess_input
from keras.api.models import Model
from AI_Search.model import Image, Tag, engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

redis_pool = redis.ConnectionPool(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=False
)
redis_client = redis.Redis(connection_pool=redis_pool)

Session = scoped_session(sessionmaker(bind=engine))

# Constants
IMAGE_SIZE = (224, 224)
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", 3600))
FAISS_M = 32
FAISS_EF_CONSTRUCTION = 200
FAISS_EF_SEARCH = 200
FAISS_M_L = 1.5


def clear_redis_cache() -> None:
    try:
        keys = redis_client.keys("embeddings:*")
        if not keys:
            logger.info("No cache keys found to delete.")
            return

        # Use pipeline for batch deletion
        pipe = redis_client.pipeline()
        for key in keys:
            pipe.delete(key)
        results = pipe.execute()
        logger.info(f"Deleted {sum(results)} Redis cache keys")
    except redis.RedisError as e:
        logger.error(f"Redis error during cache clearing: {e}")


@lru_cache(maxsize=1)
def get_extract_model() -> Model:
    logger.info("Loading EfficientNetB7 model")
    try:
        base_model = EfficientNetB7(weights="imagenet", include_top=False, pooling="avg")
        return Model(inputs=base_model.input, outputs=base_model.output)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def preprocess_image(img_path: str) -> np.ndarray:
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image '{img_path}' not found or couldn't be read.")

        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return preprocess_input(img)
    except Exception as e:
        if not isinstance(e, FileNotFoundError):
            logger.error(f"Error preprocessing image {img_path}: {e}")
        raise


def get_image_embedding(img_path: str, model: Model) -> np.ndarray:
    try:
        img = preprocess_image(img_path)
        embedding = model.predict(img, verbose=0).flatten()  # Disable verbose output
        return normalize(embedding.reshape(1, -1), axis=1, norm='l2')
    except Exception as e:
        logger.error(f"Error extracting embedding from {img_path}: {e}")
        raise


def random_level(assign_probas: List[float], rng: np.random.Generator) -> int:
    f = rng.uniform()
    for level in range(len(assign_probas)):
        if f < assign_probas[level]:
            return level
        f -= assign_probas[level]
    return len(assign_probas) - 1


@lru_cache(maxsize=8)  # Cache results since parameters rarely change
def set_default_probas(M: int, m_L: float) -> Tuple[List[float], List[int]]:
    nn = 0
    cum_nneighbor_per_level = []
    level = 0
    assign_probas = []
    exp_m_L_inv = np.exp(-1 / m_L)  # Precalculate for efficiency

    while True:
        proba = np.exp(-level / m_L) * (1 - exp_m_L_inv)
        if proba < 1e-9:
            break
        assign_probas.append(proba)
        nn += M * 2 if level == 0 else M
        cum_nneighbor_per_level.append(nn)
        level += 1
    return assign_probas, cum_nneighbor_per_level


def generate_cache_key(filter_tags: Optional[List[str]]) -> str:
    if not filter_tags:
        return "embeddings:all"
    # Sort for consistency and join with commas
    sorted_tags = sorted(filter_tags)
    tags_string = ",".join(sorted_tags)
    # Use MD5 for a compact hash representation
    hash_key = hashlib.md5(tags_string.encode()).hexdigest()
    return f"embeddings:{hash_key}"


def load_embeddings(filter_tags: Optional[Union[List[str], List[Image]]] = None) -> Tuple[np.ndarray, List[str]]:
    if filter_tags and isinstance(filter_tags, list) and filter_tags != []:
        if isinstance(filter_tags[0], str):
            try:
                decoded = json.loads(filter_tags[0])
                if isinstance(decoded, list):
                    filter_tags = decoded
                    logger.debug(f"Decoded filter_tags from JSON: {filter_tags}")
            except json.JSONDecodeError:
                pass

        elif isinstance(filter_tags[0], Image):
            try:
                session = Session()
                try:
                    image_ids = [image.id for image in filter_tags]
                    images = session.query(Image).filter(Image.id.in_(image_ids)).all()
                    embeddings = [image.embedding for image in images if image.embedding]
                    image_paths = [image.image_url for image in images]

                    if not embeddings:
                        raise ValueError("No valid embeddings found for the specified images")

                    embeddings_matrix = np.vstack(embeddings)
                    embeddings_matrix = normalize(embeddings_matrix, axis=1, norm="l2")
                    return embeddings_matrix, image_paths
                finally:
                    session.close()
            except Exception as e:
                logger.error(f"Error processing Image objects: {e}")
                raise

    if filter_tags is None or filter_tags == []:
        filter_tags = []

    cache_key = generate_cache_key(filter_tags)
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            logger.info(f"Loading embeddings from Redis cache: {cache_key}")
            embeddings, image_paths = pickle.loads(cached_data)
            return embeddings, image_paths
    except redis.RedisError as e:
        logger.warning(f"Redis error when loading cache: {e}")

    logger.info(f"Loading embeddings from database with tags: {filter_tags}")

    session = Session()
    try:
        if not filter_tags:
            images = session.query(Image).all()
        else:
            images = (
                session.query(Image)
                .join(Image.tags)
                .filter(Tag.tag_name.in_(filter_tags))
                .distinct()
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

        embeddings_matrix = np.vstack(embeddings)
        embeddings_matrix = normalize(embeddings_matrix, axis=1, norm="l2")

        try:
            redis_client.setex(
                cache_key,
                CACHE_EXPIRY,
                pickle.dumps((embeddings_matrix, image_paths))
            )
            logger.info(f"Saved embeddings to Redis cache: {cache_key}")
        except redis.RedisError as e:
            logger.warning(f"Redis error when saving cache: {e}")

        return embeddings_matrix, image_paths
    finally:
        session.close()


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexHNSWFlat:
    # Create HNSW index with parameters
    d = embeddings.shape[1]  # Dimensionality
    index = faiss.IndexHNSWFlat(d, FAISS_M)
    index.hnsw.efConstruction = FAISS_EF_CONSTRUCTION
    index.hnsw.efSearch = FAISS_EF_SEARCH

    # Get probability distribution for levels
    assign_probas, _ = set_default_probas(FAISS_M, FAISS_M_L)
    rng = np.random.default_rng()

    # Add vectors one by one with proper level assignment
    for vector in embeddings:
        level = random_level(assign_probas, rng)
        vector_float32 = vector.astype(np.float32).reshape(1, -1)
        index.add(vector_float32)

    return index


@lru_cache(maxsize=8)
def get_faiss_index(embeddings_key: str) -> Tuple[faiss.IndexHNSWFlat, np.ndarray, List[str]]:
    # Try to get from cache first
    index_key = f"faiss_index:{embeddings_key.split(':')[1]}"
    try:
        cached_index = redis_client.get(index_key)
        if cached_index:
            logger.info(f"Loading FAISS index from cache: {index_key}")
            cached_data = pickle.loads(cached_index)
            return cached_data['index'], cached_data['embeddings'], cached_data['image_paths']
    except (redis.RedisError, pickle.PickleError) as e:
        logger.warning(f"Error loading cached index: {e}")

    # If not in cache, load embeddings and build index
    embeddings, image_paths = load_embeddings(None if embeddings_key == "embeddings:all" else
                                              embeddings_key.split(':')[1].split(','))
    index = build_faiss_index(embeddings)

    # Cache the built index
    try:
        cached_data = {
            'index': index,
            'embeddings': embeddings,
            'image_paths': image_paths
        }
        redis_client.setex(index_key, CACHE_EXPIRY, pickle.dumps(cached_data))
        logger.info(f"Cached FAISS index: {index_key}")
    except (redis.RedisError, pickle.PickleError) as e:
        logger.warning(f"Error caching index: {e}")

    return index, embeddings, image_paths


def search_similar_images(embedding: np.ndarray, train_embeddings: np.ndarray, k: int = 5) -> Tuple[
    np.ndarray, np.ndarray]:
    """Search for similar images using FAISS HNSW index.

    This function uses random_level and set_default_probas to construct the HNSW graph
    with proper level assignment for each vector.

    Args:
        embedding: Query embedding vector
        train_embeddings: Matrix of embeddings to search against
        k: Number of similar images to return

    Returns:
        Tuple[np.ndarray, np.ndarray]: Distances and indices of similar images
    """
    try:
        # Generate a cache key based on the embeddings content
        cache_key = hashlib.md5(train_embeddings.tobytes()).hexdigest()
        index_key = f"embeddings:{cache_key}"

        # Try to get index from cache or build a new one
        try:
            index, _, _ = get_faiss_index(index_key)
        except Exception as e:
            logger.warning(f"Could not retrieve cached index: {e}")
            logger.info("Building new FAISS index with random_level assignment")
            index = build_faiss_index(train_embeddings)

        # Prepare query embedding
        embedding_float32 = embedding.astype(np.float32).reshape(1, -1)

        # Search for similar vectors
        distances, indices = index.search(embedding_float32, min(k, len(train_embeddings)))
        return distances, indices
    except Exception as e:
        logger.error(f"Error searching similar images: {e}")
        raise