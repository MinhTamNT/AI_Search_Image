import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy.orm import sessionmaker
from AI_Search.model import Image, Tag, engine
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

Session = sessionmaker(bind=engine)

def get_all_image_tags(image_objs):
    """
    Lấy list tag cho từng ảnh (theo thứ tự image_objs)
    """
    all_tags = []
    for img in image_objs:
        if img.tags:
            all_tags.append([tag.tag_name for tag in img.tags])
        else:
            all_tags.append([])
    return all_tags

def plot_metrics(results):
    ks = sorted(results.keys())
    precisions = [np.mean(results[k]['precision']) for k in ks]
    maps = [np.mean(results[k]['ap']) for k in ks]
    mrrs = [np.mean(results[k]['rr']) for k in ks]

    x = np.arange(len(ks))  # vị trí trên trục x
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, precisions, width, label='Precision')
    rects2 = ax.bar(x, maps, width, label='mAP')
    rects3 = ax.bar(x + width, mrrs, width, label='MRR')

    ax.set_xlabel('Top-K')
    ax.set_ylabel('Scores')
    ax.set_title('Retrieval Evaluation Metrics by Tag')
    ax.set_xticks(x)
    ax.set_xticklabels([f'@{k}' for k in ks])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    plt.tight_layout()
    plt.show()

def evaluate_retrieval_by_tag_db(ks=[1, 5, 10]):
    session = Session()
    try:
        images = session.query(Image).all()
        embeddings = []
        image_paths = []
        for img in images:
            if img.embedding is not None:
                embeddings.append(np.array(img.embedding))
                image_paths.append(img.image_path if img.image_path else img.image_url)
        embeddings = np.vstack(embeddings)
        n = len(image_paths)
        dists = cosine_distances(embeddings, embeddings)
        all_tags = get_all_image_tags(images)
        results = {k: {'precision': [], 'ap': [], 'rr': []} for k in ks}

        for i in tqdm(range(n), desc="Evaluating"):
            indices = np.argsort(dists[i])
            indices = [idx for idx in indices if idx != i]
            query_tags = set(all_tags[i])
            for k in ks:
                top_k_indices = indices[:k]
                found = False
                ap = 0.0
                rr = 0.0
                for rank, idx in enumerate(top_k_indices, 1):
                    candidate_tags = set(all_tags[idx])
                    if query_tags & candidate_tags:  # Có giao tag
                        if not found:
                            ap = 1.0 / rank
                            rr = 1.0 / rank
                            found = True
                results[k]['precision'].append(int(found))
                results[k]['ap'].append(ap)
                results[k]['rr'].append(rr)

        for k in ks:
            print(f"Precision@{k}: {np.mean(results[k]['precision']):.4f}")
            print(f"mAP@{k}: {np.mean(results[k]['ap']):.4f}")
            print(f"MRR@{k}: {np.mean(results[k]['rr']):.4f}")
            print("-" * 30)

        plot_metrics(results)
        return results
    finally:
        session.close()

if __name__ == "__main__":
    evaluate_retrieval_by_tag_db(ks=[1, 5, 10])
