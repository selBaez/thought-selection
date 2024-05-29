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


if __name__ == '__main__':
    from src.user_interface.application import application
    # Run
    application.run(debug=True)
