from flask import Flask

from chatbot_ui.app.routes import create_endpoints
from chatbot_ui.config import Config

from chatbot.chatbots import Chatbot

# Create application # TODO make into function
application = Flask(__name__)
application.config.from_object(Config)

chatbot = Chatbot()

# Create endpoints
application = create_endpoints(application, chatbot)
