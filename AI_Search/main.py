from flask import request, jsonify
from flasgger import swag_from
from Service.image_service import get_extract_model, load_embeddings, get_image_embedding, search_similar_images
import threading
from dao.dao import update_dataset
from AI_Search import  app

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

    file_path = f"/flickr30k_images/flickr30k_images/{file.filename}"
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
        def async_save():
            update_dataset(file_path, embedding)

        threading.Thread(target=async_save).start()

        return jsonify({
            "message": "không tìm thấy ảnh tương tự",
            "results": []
        }), 200

    return jsonify({"results": filtered_results}), 200

if __name__ == '__main__':
    with app.app_context():
        app.run()