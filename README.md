# Image Search and Embedding Storage

This project provides a web service for searching similar images and storing image embeddings along with comments. It uses a pre-trained EfficientNetB7 model to extract image embeddings and stores them in a database. The service also supports asynchronous updates to the dataset.

## Features

- Search for similar images based on embeddings.
- Store new images and their embeddings in the database.
- Add comments to images.
- Asynchronous dataset updates to avoid delays in user responses.

## Technologies Used

- ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
- ![Flask](https://img.shields.io/badge/Flask-1.1.2-green)
- ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-1.3.23-red)
- ![Keras](https://img.shields.io/badge/Keras-2.4.3-orange)
- ![OpenCV](https://img.shields.io/badge/OpenCV-4.5.1-yellow)
- ![FAISS](https://img.shields.io/badge/FAISS-1.7.0-lightgrey)
- ![Redis](https://img.shields.io/badge/Redis-6.0.9-red)
- ![Pandas](https://img.shields.io/badge/Pandas-1.2.1-blue)

## Requirements

- Python 3.8+
- Flask
- SQLAlchemy
- Keras
- OpenCV
- FAISS
- Redis
- Pandas

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/MinhTamNT/image-search-embedding.git
    cd image-search-embedding
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up the database:
    ```sh
    python setup_database.py
    ```

## Usage

1. Start the Flask application:
    ```sh
    python main.py
    ```

2. Use the `/search` endpoint to search for similar images:
    ```sh
    curl -X POST -F "file=@path_to_your_image.jpg" http://127.0.0.1:5000/search
    ```

## Project Structure

- `main.py`: The main Flask application file.
- `Service/image_service.py`: Contains functions for image processing and embedding extraction.
- `dao/dao.py`: Contains functions for database operations.
- `model.py`: Defines the database models.
- `store_vector.py`: Script for computing and storing image embeddings.
- `setup_database.py`: Script for setting up the database.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
