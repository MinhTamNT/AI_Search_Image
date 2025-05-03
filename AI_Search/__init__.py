from flask import Flask
from flask_cors import CORS
from flasgger import Swagger
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AI_Search')))

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
