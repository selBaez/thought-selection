import os

from flask import render_template, flash, redirect, request, url_for

from src.chatbot.chatbots import Chatbot
from src.chatbot_ui.app.forms import TurnForm, ChatForm
from src.chatbot_ui.app.utils.capsule_utils import digest_form, begin_form, capsule_to_form

ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
RESOURCES_PATH = ABSOLUTE_PATH + "/../../../resources/"
THOUGHTS_FILE = RESOURCES_PATH + "thoughts.json"

CHATBOT = None
CAPSULES_SUBMITTED = []
CAPSULES_SUGGESTED = []
SAY_HISTORY = []


def create_endpoints(app):
    @app.route('/', methods=['GET', 'POST'])
    @app.route('/index', methods=['GET', 'POST'])
    def index():

        # handle the POST request
        if request.method == 'POST':
            form_in = ChatForm()

            # Create chatbot
            global CHATBOT
            CHATBOT = Chatbot(form_in.chat_id.data, form_in.speaker.data, "RL", THOUGHTS_FILE)
            reply = {'say': CHATBOT.greet}
            form_out = TurnForm()

            return redirect(url_for('capsule', title='Submit Capsule', form=form_out, reply=reply, capsules=[]))

        # handle the GET request
        elif request.method == 'GET':
            form_out = ChatForm()

            return render_template('index.html', title='Start chat', form=form_out)

    @app.route('/capsule', methods=['GET', 'POST'])
    def capsule():
        global CAPSULES_SUBMITTED
        global CAPSULES_SUGGESTED
        global SAY_HISTORY
        # handle the POST request
        if request.method == 'POST':
            # if form has data, assign it to the capsule and send to chatbot
            form_in = TurnForm()
            form_out, reply, capsule, capsule_user = digest_form(form_in, CHATBOT)
            CAPSULES_SUBMITTED.append(capsule)
            SAY_HISTORY.append(reply)
            CAPSULES_SUGGESTED.append(capsule_user)
            print("I AM HERE, IN THE POST REQUEST")

            return redirect(
                url_for('capsule', title='Submit Capsule', form=form_out, reply=reply, capsules=CAPSULES_SUBMITTED))

        # handle the GET request
        elif request.method == 'GET':
            # if form does not have data, try to prefill somethings from capsule_user or use template for first time
            form_in = TurnForm()
            if CHATBOT._turns == 0:
                # First time, template
                form_out, reply, capsule_user = begin_form(form_in, CHATBOT)
                CAPSULES_SUGGESTED.append(capsule_user)
                print("I AM HERE, IN THE FIRST GET REQUEST")

            else:
                # Next times, use suggested templates
                form_out = capsule_to_form(CAPSULES_SUGGESTED[-1], form_in)
                reply = SAY_HISTORY[-1]
                print("I AM HERE, IN THE OTHER GET REQUEST")

            if form_out.validate_on_submit():
                flash('Fields missing for user {}, remember_me={}'.format(
                    form_out.subject_label.data, form_out.subject_from_label.data))
                return redirect('/index')

            return render_template('capsule.html', title='Submit Capsule', form=form_out, reply=reply,
                                   capsules=CAPSULES_SUBMITTED)

    return app
