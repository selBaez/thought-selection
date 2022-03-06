from datetime import datetime

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, FloatField, SelectField, DateTimeField
from wtforms.validators import DataRequired


class TurnForm(FlaskForm):
    # Triple information
    subject_label = StringField('Subject label', validators=[DataRequired()])
    subject_types = StringField('Subject types', validators=[DataRequired()])
    # subject_from_label = BooleanField('Create URI from label')

    predicate_label = StringField('Predicate label', validators=[DataRequired()])
    # predicate_from_label = BooleanField('Create URI from label')

    object_label = StringField('Object label', validators=[DataRequired()])
    object_types = StringField('Object types', validators=[DataRequired()])
    # object_from_label = BooleanField('Create URI from label')

    # Perspective information
    perspective_certainty = FloatField('Certainty', validators=[DataRequired()])
    perspective_polarity = FloatField('Polarity', validators=[DataRequired()])
    perspective_sentiment = FloatField('Sentiment', validators=[DataRequired()])

    # Chat information
    chat_id = IntegerField('Chat ID', validators=[DataRequired()])
    turn_id = IntegerField('Turn ID', validators=[DataRequired()])
    author = StringField('Author', validators=[DataRequired()])

    # Utterance info
    utterance_types = ["STATEMENT", "QUESTION"]
    utterance = StringField('Utterance', validators=[DataRequired()])
    utterance_type = SelectField('Utterance type', choices=utterance_types, validators=[DataRequired()])
    # position = StringField('Position', default="", validators=[DataRequired()])

    # Context information
    context_id = IntegerField('Context ID', default=56, validators=[DataRequired()])
    context_date = DateTimeField("Date", format="%Y-%m-%d", default=datetime.today, validators=[DataRequired()])

    # Place info
    place_id = IntegerField('Place ID', default=98, validators=[DataRequired()])
    place_label = StringField('Place label', default='office', validators=[DataRequired()])
    country = StringField('Country', default='NL', validators=[DataRequired()])
    region = StringField('Region', default='North Holland', validators=[DataRequired()])
    city = StringField('City', default='Amsterdam', validators=[DataRequired()])

    # Multimodal information
    # objects = StringField('Objects in the room', default='', validators=[DataRequired()])
    # people = StringField('People in the room', default='', validators=[DataRequired()])

    submit = SubmitField('Submit capsule')


class ChatForm(FlaskForm):
    chat_id = IntegerField('Chat ID', validators=[DataRequired()])
    speaker = StringField('Speaker', validators=[DataRequired()])

    rewards = ['Total triples',
               # 'Total classes', 'Total predicates',
               # 'Total statements', 'Total perspectives', 'Total conflicts',
               # 'Total sources',
               'Ratio statements to triples', 'Ratio perspectives to triples', 'Ratio conflicts to triples',
               'Ratio perspectives to statements', 'Ratio conflicts to statements'
               ]
    reward = SelectField('Reward function', choices=rewards, validators=[DataRequired()])

    submit = SubmitField('Start chat')


class SaveForm(FlaskForm):
    session_folder = StringField('Session folder', validators=[DataRequired()])
    database_address = StringField('Address for the triple store', default="http://localhost:7200/repositories/sandbox",
                                   validators=[DataRequired()])
    rdf_folder = StringField('RDF logs folder', validators=[DataRequired()])
    thoughts_file = StringField('Thoughts file', validators=[DataRequired()])
    capsules_file = StringField('Capsules file', validators=[DataRequired()])

    submit = SubmitField('Save session')
