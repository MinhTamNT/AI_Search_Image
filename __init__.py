from flask import Flask
from flask_cors import CORS
app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False
CORS(app)