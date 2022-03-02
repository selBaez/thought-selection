from flask import Flask

from chatbot_ui.app.routes import create_endpoints
from chatbot_ui.config import Config

# Create application # TODO make into function
application = Flask(__name__)
application.config.from_object(Config)

# Create endpoints
application = create_endpoints(application)
