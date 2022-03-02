import logging
import sys

from flask import Flask

from src.chatbot_ui.app.routes import create_endpoints
from src.chatbot_ui.config import Config

# # Logging information
# root = logging.getLogger()
# root.setLevel(logging.INFO)
#
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# root.addHandler(handler)
#
# logger = logging.getLogger(__name__)

# Create application
application = Flask(__name__)
application.config.from_object(Config)

# Create endpoints
application = create_endpoints(application)

if __name__ == '__main__':
    # Run
    application.run(debug=True)
