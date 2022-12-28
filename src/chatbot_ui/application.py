from flask import Flask

from src.chatbot.chatbots import Chatbot
from src.chatbot_ui.app.routes import create_endpoints
from src.chatbot_ui.config import Config

# Create application # TODO make into function
application = Flask(__name__)
application.config.from_object(Config)

chatbot = Chatbot()

# Create endpoints
application = create_endpoints(application, chatbot)
