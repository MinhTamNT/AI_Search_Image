from flask import Flask
from flask_cors import CORS
from flasgger import Swagger

app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False
app.config['SWAGGER'] = {
    'title': 'Image Search API',
    'description': 'API to search for similar images',
    'version': '1.0',
    'uiversion': 3
}

app.config['SWAGGER_UI_CONFIG'] = {
    'url': '/swagger.json'
}
CORS(app, resources={r"/*": {"origins": "*"}})
swagger = Swagger(app)
