from flask import Flask

from src.dialogue_system.chatbot import Chatbot
from src.user_interface.app.routes import create_endpoints
from src.user_interface.app.utils.config import Config

# Create application # Future work: make into function
application = Flask(__name__)
application.config.from_object(Config)

chatbot = Chatbot()

# Create endpoints
application = create_endpoints(application, chatbot)
