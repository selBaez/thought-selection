from flask import render_template, redirect, request, url_for

from src.user_interface.app.utils.form_utils import statement_capsule_to_form, form_to_context_capsule, digest_form, \
    begin_form
from src.user_interface.app.forms import TurnForm, ChatForm, SaveForm


def create_endpoints(app, chatbot):
    @app.route('/', methods=['GET', 'POST'])
    @app.route('/index', methods=['GET', 'POST'])
    def index():
        # handle the POST request
        if request.method == 'POST':
            form_in = ChatForm()

            # Create dialogue_system
            chatbot.begin_session(form_in.chat_id.data, form_in.speaker.data, form_in.reward.data)

            # Situate chat
            capsule_for_context = form_to_context_capsule(form_in)
            chatbot.situate_chat(capsule_for_context)

            # Generate greeting for starting chat
            reply = {'say': chatbot.greet}

            # Go to capsule page
            form_out = TurnForm()

            return redirect(url_for('capsule', title='Submit Capsule', form=form_out, reply=reply, capsules=[]))

        # handle the GET request
        elif request.method == 'GET':
            # Empty form
            form_out = ChatForm()
            return render_template('index.html', title='Start chat', form=form_out)

    @app.route('/capsule', methods=['GET', 'POST'])
    def capsule():
        # handle the POST request (user input)
        if request.method == 'POST':
            form_in = TurnForm()

            # assign form data to the capsule and send to dialogue_system
            form_out, reply, capsule_in, capsule_user = digest_form(form_in, chatbot)

            return redirect(url_for('capsule', title='Submit Capsule',
                                    form=form_out, reply=reply, capsules=chatbot.chat_history["capsules_submitted"]))

        # handle the GET request (prefilled)
        elif request.method == 'GET':
            form_in = TurnForm()

            # form does not have data, use templates
            if chatbot.turns == 0:
                # First time, template is hard coded
                form_out, reply, capsule_user = begin_form(form_in, chatbot)

            else:
                # Use last suggested template
                form_out = statement_capsule_to_form(chatbot.chat_history["capsules_suggested"][-1], form_in)
                reply = chatbot.chat_history["say_history"][-1]

            if form_out.validate_on_submit():
                return redirect('/index')

            return render_template('capsule.html', title='Submit Capsule',
                                   form=form_out, reply=reply, capsules=chatbot.chat_history['capsules_submitted'])

    @app.route('/save', methods=['GET', 'POST'])
    def save():
        # handle the POST request
        if request.method == 'POST':
            # Save capsules, thoughts, RDF path
            form_in = SaveForm()

            chatbot.close_session()

            form_out = ChatForm()

            return redirect(url_for('index', title='Start chat', form=form_out))

        # handle the GET request
        elif request.method == 'GET':
            # Show details form
            form_out = SaveForm()

            form_out.session_folder.data = chatbot.scenario_folder
            form_out.database_address.data = chatbot.address
            form_out.rdf_folder.data = chatbot.brain.log_dir
            form_out.thoughts_file.data = chatbot.thoughts_file
            form_out.capsules_file.data = chatbot.capsules_file

            reply = {'say': chatbot.farewell}

            return render_template('save.html', title='Save session', form=form_out, reply=reply)

    return app
