from flask import request, jsonify
import threading
import os
import base64
from Service.image_service import get_extract_model, load_embeddings, get_image_embedding, search_similar_images
from dao.dao import update_dataset
from AI_Search import app



def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


@app.route('/search', methods=['POST'])
def search_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        file_path = f"AI_Search/Upload/{file.filename}"
        file.save(file_path)

        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        start = (page - 1) * per_page
        end = start + per_page

        filter_tags = request.form.getlist('tags')
        print("Filter tags:", filter_tags)

        model = get_extract_model()

        try:
            train_embeddings, train_image_paths = load_embeddings(filter_tags)
        except ValueError as e:
            return jsonify({
                "message": str(e),
                "results": [],
                "page": page,
                "per_page": per_page,
                "total_results": 0,
                "total_pages": 0
            }), 200

        embedding = get_image_embedding(file_path, model)

        top_k = min(50, len(train_embeddings))
        if top_k == 0:
            return jsonify({
                "message": "không tìm thấy ảnh tương tự",
                "results": [],
                "page": page,
                "per_page": per_page,
                "total_results": 0,
                "total_pages": 0
            }), 200

        distances, indices = search_similar_images(embedding, train_embeddings, k=top_k)

        threshold = 1.5
        filtered_results = [
            {
                "image_path": image_to_base64(train_image_paths[idx]),
                "distance": float(dist)
            }
            for dist, idx in zip(distances[0], indices[0]) if dist < threshold
        ]

        if not filtered_results:
            def async_save():
                update_dataset(file_path, embedding)

            threading.Thread(target=async_save).start()
            return jsonify({
                "message": "không tìm thấy ảnh tương tự",
                "results": [],
                "page": page,
                "per_page": per_page,
                "total_results": 0,
                "total_pages": 0
            }), 200

        # Apply pagination
        total_results = len(filtered_results)
        total_pages = (total_results + per_page - 1) // per_page
        paginated_results = filtered_results[start:end]

        return jsonify({
            "results": paginated_results,
            "page": page,
            "per_page": per_page,
            "total_results": total_results,
            "total_pages": total_pages
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)


@app.route('/')
def home():
    return "Hello, Flask is working!"


if __name__ == '__main__':
    with app.app_context():
        app.run(debug=True)
