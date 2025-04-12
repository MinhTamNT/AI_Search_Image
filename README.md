# Image Search and Embedding Storage

This project provides a web service for searching similar images and storing image embeddings along with comments. It uses a pre-trained EfficientNetB7 model to extract image embeddings and stores them in a database. The service also supports asynchronous updates to the dataset.

## Features

- Search for similar images based on embeddings.
- Filter search results by tags.
- Store new images and their embeddings in the database.
- Add comments to images.
- Asynchronous dataset updates to avoid delays in user responses.
- Caching with Redis for faster performance.

## Technologies Used

- ![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
- ![Flask](https://img.shields.io/badge/Flask-1.1.2-green)
- ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-1.3.23-red)
- ![Keras](https://img.shields.io/badge/Keras-2.4.3-orange)
- ![OpenCV](https://img.shields.io/badge/OpenCV-4.5.1-yellow)
- ![FAISS](https://img.shields.io/badge/FAISS-1.7.0-lightgrey)
- ![Redis](https://img.shields.io/badge/Redis-6.0.9-red)
- ![Pandas](https://img.shields.io/badge/Pandas-1.2.1-blue)

## Requirements

- Python 3.10 or higher
- Flask
- SQLAlchemy
- Keras
- OpenCV
- FAISS
- Redis
- Pandas

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/MinhTamNT/image-search-embedding.git
    cd image-search-embedding
    ```

2. **Create a virtual environment and activate it**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up the database**:
    ```sh
    python setup_database.py
    ```

5. **Configure environment variables**:
    - Create a `.env` file in the root directory and add the following:
      ```
      REDIS_HOST=localhost
      REDIS_PORT=6379
      ```

## Usage

1. **Start the Flask application**:
    ```sh
    python main.py
    ```

2. **Search for similar images**:
    - Using an image file:
      ```sh
      curl -X POST -F "file=@path_to_your_image.jpg" http://127.0.0.1:5000/search
      ```
    - Using tags:
      ```sh
      curl -X POST -F "tags=tag1" -F "tags=tag2" http://127.0.0.1:5000/search
      ```

3. **Retrieve tags with pagination**:
    ```sh
    curl -X GET "http://127.0.0.1:5000/tags?page=1&per_page=10"
    ```

## Project Structure

- `main.py`: The main Flask application file.
- `Service/image_service.py`: Contains functions for image processing and embedding extraction.
- `dao/dao.py`: Contains functions for database operations.
- `model.py`: Defines the database models.
- `setup_database.py`: Script for setting up the database.
- `store_vector.py`: Script for computing and storing image embeddings.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.