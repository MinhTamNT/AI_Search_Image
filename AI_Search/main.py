from flask import request, jsonify
import threading
import os
from Service.image_service import get_extract_model, load_embeddings, get_image_embedding, search_similar_images
from dao.dao import update_dataset , get_tags_with_pagination
from AI_Search import app

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
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        start = (page - 1) * per_page
        end = start + per_page

        # Get tags and file
        filter_tags = request.form.getlist('tags')
        file = request.files.get('file', None)

        # If no tags or file are provided, return an error
        if not filter_tags and not file:
            return jsonify({"error": "No tags or file provided for search"}), 400

        # If only tags are provided, search by tags
        if not file:
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

            # Return all images matching the tags
            total_results = len(train_image_paths)
            total_pages = (total_results + per_page - 1) // per_page
            paginated_results = [{"image_path": path} for path in train_image_paths[start:end]]

            return jsonify({
                "results": paginated_results,
                "page": page,
                "per_page": per_page,
                "total_results": total_results,
                "total_pages": total_pages
            }), 200

        # If an image is provided, process it
        file_path = f"AI_Search/Upload/{file.filename}"
        file.save(file_path)

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
