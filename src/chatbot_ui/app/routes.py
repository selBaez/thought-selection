import os

from flask import render_template, flash, redirect, request, url_for

from chatbot.chatbots import Chatbot
from chatbot_ui.app.forms import TurnForm, ChatForm
from chatbot_ui.app.utils.capsule_utils import digest_form, begin_form, capsule_to_form

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
        global CHATBOT

        # handle the POST request
        if request.method == 'POST':
            # Create chatbot
            form_in = ChatForm()
            CHATBOT = Chatbot(form_in.chat_id.data, form_in.speaker.data, "RL", form_in.reward.data, THOUGHTS_FILE)
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
        global CHATBOT

        # handle the POST request
        if request.method == 'POST':
            # if form has data, assign it to the capsule and send to chatbot
            form_in = TurnForm()
            form_out, reply, capsule, capsule_user = digest_form(form_in, CHATBOT)

            # add reply and reward
            capsule["brain_state"] = CHATBOT.replier.brain_states[-1]
            CAPSULES_SUBMITTED.append(capsule)
            SAY_HISTORY.append(reply)
            CAPSULES_SUGGESTED.append(capsule_user)

            return redirect(url_for('capsule', title='Submit Capsule',
                                    form=form_out, reply=reply, capsules=CAPSULES_SUBMITTED))

        # handle the GET request
        elif request.method == 'GET':
            # if form does not have data, use templates
            form_in = TurnForm()
            if CHATBOT._turns == 0:
                # First time, template
                form_out, reply, capsule_user = begin_form(form_in, CHATBOT)
                CAPSULES_SUGGESTED.append(capsule_user)

            else:
                # Next times, use suggested templates
                form_out = capsule_to_form(CAPSULES_SUGGESTED[-1], form_in)
                reply = SAY_HISTORY[-1]

            if form_out.validate_on_submit():
                flash('Fields missing for user {}, remember_me={}'.format(
                    form_out.subject_label.data, form_out.subject_from_label.data))
                return redirect('/index')

            return render_template('capsule.html', title='Submit Capsule',
                                   form=form_out, reply=reply, capsules=CAPSULES_SUBMITTED)

    return app
