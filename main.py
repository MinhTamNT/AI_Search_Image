
from model import Image, engine
from flask import request, jsonify
import threading
from . import  app
from Service.image_service import get_extract_model , load_embeddings , get_image_embedding ,search_similar_images

@app.route('/search', methods=['POST'])
def search_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = f"./flickr30k_images/flickr30k_images/{file.filename}"
    file.save(file_path)

    model = get_extract_model()
    train_embeddings, train_image_paths = load_embeddings()
    embedding = get_image_embedding(file_path, model)
    distances, indices = search_similar_images(embedding, train_embeddings)

    threshold = 1.5
    filtered_results = [
        {"image_path": train_image_paths[idx], "distance": float(dist)}
        for dist, idx in zip(distances[0], indices[0]) if dist < threshold
    ]

    if not filtered_results:
        response = jsonify({"message": "No similar images found. The new image will be added to the dataset."})
        response.status_code = 200



    return jsonify({"results": filtered_results}), 200


if __name__ == '__main__':
    app.run(debug=True)
