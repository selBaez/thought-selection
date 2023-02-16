from flask import Flask

from src.dialogue_system.chatbots import Chatbot
from src.user_interface.app.routes import create_endpoints
from src.user_interface.config import Config

# Create application # TODO make into function
application = Flask(__name__)
application.config.from_object(Config)

chatbot = Chatbot()

# Create endpoints
application = create_endpoints(application, chatbot)
