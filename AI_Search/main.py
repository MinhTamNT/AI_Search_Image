from flask import request, jsonify
import threading
import os
from Service.image_service import get_extract_model, load_embeddings, get_image_embedding, search_similar_images
from dao.dao import update_dataset , get_tags_with_pagination
from AI_Search import app
from dotenv import load_dotenv
import uuid
import datetime
load_dotenv()
@app.route('/tags', methods=['GET'])
def get_tags():
    """
    Retrieve all tags with pagination and optional name filtering.
    ---
    parameters:
      - name: page
        in: query
        type: integer
        required: false
        default: 1
        description: Page number for pagination
      - name: per_page
        in: query
        type: integer
        required: false
        default: 10
        description: Number of tags per page
      - name: search_name
        in: query
        type: string
        required: false
        description: Filter tags by name (case-insensitive)
    responses:
      200:
        description: A list of tags with pagination
        schema:
          type: object
          properties:
            results:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: integer
                    description: Tag ID
                  tag_name:
                    type: string
                    description: Tag name
            page:
              type: integer
              description: Current page number
            per_page:
              type: integer
              description: Number of tags per page
            total_results:
              type: integer
              description: Total number of tags
            total_pages:
              type: integer
              description: Total number of pages
      500:
        description: Internal server error
    """
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        search_name = request.args.get('search_name', None, type=str)

        total_results, total_pages, tags = get_tags_with_pagination(page, per_page, search_name)

        results = [{"id": tag.id, "tag_name": tag.tag_name} for tag in tags]

        return jsonify({
            "results": results,
            "page": page,
            "per_page": per_page,
            "total_results": total_results,
            "total_pages": total_pages
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/search', methods=['POST'])
def search_image():
    """
    Search for similar images using image embeddings or filter by tags.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: false
        description: Image file to search
      - name: tags
        in: formData
        type: array
        items:
          type: string
        required: false
        description: Filter tags
      - name: page
        in: query
        type: integer
        required: false
        default: 1
        description: Page number for pagination
      - name: per_page
        in: query
        type: integer
        required: false
        default: 10
        description: Number of results per page
    responses:
      200:
        description: A list of results based on the search criteria
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
                    description: Image path
                  distance:
                    type: number
                    description: Distance to the input image (if applicable)
            page:
              type: integer
            per_page:
              type: integer
            total_results:
              type: integer
            total_pages:
              type: integer
      400:
        description: Invalid input
    """
    try:
        # Get pagination params
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        start = (page - 1) * per_page
        end = start + per_page

        # Get input tags & file
        filter_tags = request.form.getlist('tags')
        file = request.files.get('file', None)
        if not filter_tags and not file:
            return jsonify({"error": "No tags or file provided for search"}), 400

        # CASE 1: Only tags
        if not file:
            try:
                embeddings, image_paths = load_embeddings(filter_tags)
            except ValueError as e:
                return jsonify({
                    "message": str(e),
                    "results": [],
                    "page": page,
                    "per_page": per_page,
                    "total_results": 0,
                    "total_pages": 0
                }), 200

            total_results = len(image_paths)
            total_pages = (total_results + per_page - 1) // per_page
            paginated_results = [{"image_path": path} for path in image_paths[start:end]]

            return jsonify({
                "results": paginated_results,
                "page": page,
                "per_page": per_page,
                "total_results": total_results,
                "total_pages": total_pages
            }), 200

        # Xử lý file upload và giữ nguyên định dạng
        if file and file.filename != '':
            # Đảm bảo thư mục upload tồn tại
            upload_folder = os.getenv('URL_UPLOAD')
            os.makedirs(upload_folder, exist_ok=True)
            
            original_filename = file.filename
            file_mime = file.mimetype
            
            # Chuyển đổi mimetype sang phần mở rộng file
            mime_to_extension = {
                'image/jpeg': '.jpg',
                'image/jpg': '.jpg',
                'image/png': '.png',
                'image/gif': '.gif',
                'image/bmp': '.bmp',
                'image/webp': '.webp'
            }
            
            if file_mime not in mime_to_extension:
                allowed_formats = ', '.join(mime_to_extension.values())
                return jsonify({"error": f"File format not supported. Allowed formats: {allowed_formats}"}), 400
                
            file_ext = mime_to_extension[file_mime]
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}{file_ext}"
            
            file_path = os.path.join(upload_folder, unique_filename)
            
            file.save(file_path)
        else:
            return jsonify({"error": "No file provided for search"}), 400

        model = get_extract_model()
        embedding = get_image_embedding(file_path, model)

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

        # Search for similar images
        top_k = min(50, len(train_embeddings))
        if top_k == 0:
            return jsonify({
                "message": "No similar images found",
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
                "image_path": train_image_paths[idx],
                "distance": float(dist)
            }
            for dist, idx in zip(distances[0], indices[0]) if dist < threshold
        ]

        print("Filtered Results:", filtered_results)

        if not filtered_results:
            def async_save():
                update_dataset(file_path, embedding)

            threading.Thread(target=async_save).start()
            return jsonify({
                "message": "No similar images found",
                "results": [],
                "page": page,
                "per_page": per_page,
                "total_results": 0,
                "total_pages": 0
            }), 200

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

    # finally:
    #     if 'file_path' in locals() and os.path.exists(file_path):
    #         os.remove(file_path)


@app.route('/')
def home():
    return "Hello, Flask is working!"


if __name__ == '__main__':
    with app.app_context():
        app.run(debug=True)
