from flask import request, jsonify
from flasgger import swag_from
from Service.image_service import get_extract_model, load_embeddings, get_image_embedding, search_similar_images
import threading
from dao.dao import update_dataset
from AI_Search import app
import base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.route('/search', methods=['POST'])
def search_image():
    """
    Search for similar images
    ---
    tags:
      - Image Search
    summary: Search similar images using image embeddings
    operationId: searchImage
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Upload an image to find similar ones
      - name: page
        in: query
        type: integer
        required: false
        description: Page number for pagination
      - name: per_page
        in: query
        type: integer
        required: false
        description: Number of results per page
    responses:
      200:
        description: A list of similar images
        schema:
          type: object
          properties:
            results:
              type: array
              items:
                type: object
                properties:
                  image_path:
                    type: string
                    example: /path/to/image.jpg
                  distance:
                    type: number
                    format: float
                    example: 0.45
            message:
              type: string
              example: không tìm thấy ảnh tương tự
      400:
        description: Invalid input (no file)
        schema:
          type: object
          properties:
            error:
              type: string
              example: No file part
    """

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = f"AI_Search/flickr30k_images/flickr30k_images/{file.filename}"
    file.save(file_path)

    # Pagination info
    PageIndex = request.args.get('page', 1, type=int)
    PageSize = request.args.get('per_page', 10, type=int)
    start = (PageIndex - 1) * PageSize
    end = start + PageSize

    model = get_extract_model()
    train_embeddings, train_image_paths = load_embeddings()
    embedding = get_image_embedding(file_path, model)

    top_k = PageSize
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
            "results": []
        }), 200

    # Pagination
    total_results = len(filtered_results)
    total_pages = (total_results + PageSize - 1) // PageSize
    paginated_results = filtered_results[start:end]

    return jsonify({
        "results": paginated_results,
        "page": PageIndex,
        "per_page": PageSize,
        "total_results": total_results,
        "total_pages": total_pages
    }), 200


if __name__ == '__main__':
    with app.app_context():
        app.run()