from datetime import datetime

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, SelectField, DateField, URLField
from wtforms.validators import DataRequired

from chatbot.utils.global_variables import CONTEXT_ID, PLACE_ID, PLACE_NAME, LOCATION


class TurnForm(FlaskForm):
    # Triple information
    subject_label = StringField('Subject label', validators=[DataRequired()])
    subject_types = StringField('Subject types, divided by ","', validators=[DataRequired()])
    subject_uri = URLField('Subject URI', default='http://cltl.nl/leolani/world/', validators=[DataRequired()])

    predicate_label = StringField('Predicate label', validators=[DataRequired()])
    predicate_uri = URLField('Predicate URI', default='http://cltl.nl/leolani/n2mu/', validators=[DataRequired()])

    object_label = StringField('Object label', validators=[DataRequired()])
    object_types = StringField('Object types, divided by ","', validators=[DataRequired()])
    object_uri = URLField('Object URI', default='http://cltl.nl/leolani/world/', validators=[DataRequired()])

    # Perspective information
    perspective_certainty = SelectField('Certainty', choices=['CERTAIN', 'PROBABLE', 'POSSIBLE', 'UNDERSPECIFIED'],
                                        default='UNDERSPECIFIED', validators=[DataRequired()])
    perspective_polarity = SelectField('Polarity', choices=['POSITIVE', 'NEGATIVE'],
                                       default='POSITIVE', validators=[DataRequired()])
    perspective_sentiment = SelectField('Sentiment', choices=['POSITIVE', 'NEUTRAL', 'NEGATIVE', 'UNDERSPECIFIED'],
                                        default='UNDERSPECIFIED', validators=[DataRequired()])

    # Chat information
    turn_id = IntegerField('Turn ID', validators=[DataRequired()])

    # Utterance info
    utterance = StringField('Utterance', validators=[DataRequired()])
    utterance_type = SelectField('Utterance type', choices=["STATEMENT", "QUESTION"], validators=[DataRequired()])
    # position = StringField('Position', default="", validators=[DataRequired()])

    submit = SubmitField('Submit capsule')


class ChatForm(FlaskForm):
    # Experimental condition
    chat_id = IntegerField('Chat ID', validators=[DataRequired()])
    speaker = StringField('Speaker', validators=[DataRequired()])

    rewards = ['Average degree', 'Sparseness',
               'Total triples',
               'Ratio statements to triples', 'Ratio perspectives to triples', 'Ratio conflicts to triples',
               'Ratio perspectives to statements', 'Ratio conflicts to statements',
               ]
    reward = SelectField('Reward function', choices=rewards, validators=[DataRequired()])

    # Context information
    context_id = IntegerField('Context ID', default=CONTEXT_ID, validators=[DataRequired()])
    context_date = DateField("Date", format="%Y-%m-%d", default=datetime.today, validators=[DataRequired()])

    # Place info
    place_id = IntegerField('Place ID', default=PLACE_ID, validators=[DataRequired()])
    place_label = StringField('Place label', default=PLACE_NAME, validators=[DataRequired()])
    country = StringField('Country', default=LOCATION["country"], validators=[DataRequired()])
    region = StringField('Region', default=LOCATION["region"], validators=[DataRequired()])
    city = StringField('City', default=LOCATION["city"], validators=[DataRequired()])

    submit = SubmitField('Start chat')


class SaveForm(FlaskForm):
    session_folder = StringField('Session folder', validators=[DataRequired()])
    database_address = StringField('Address for the triple store', default="http://localhost:7200/repositories/sandbox",
                                   validators=[DataRequired()])
    rdf_folder = StringField('RDF logs folder', validators=[DataRequired()])
    thoughts_file = StringField('Thoughts file', validators=[DataRequired()])
    capsules_file = StringField('Capsules file', validators=[DataRequired()])

    submit = SubmitField('Save session')
