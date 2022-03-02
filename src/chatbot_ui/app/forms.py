from datetime import datetime

from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField, SubmitField, IntegerField, FloatField, SelectField, DateTimeField
from wtforms.validators import DataRequired


class TurnForm(FlaskForm):
    # Triple information
    subject_label = StringField('Subject label', validators=[DataRequired()])
    subject_types = StringField('Subject types', validators=[DataRequired()])
    subject_from_label = BooleanField('Create URI from label')

    predicate_label = StringField('Predicate label', validators=[DataRequired()])
    predicate_from_label = BooleanField('Create URI from label')

    object_label = StringField('Object label', validators=[DataRequired()])
    object_types = StringField('Object types', validators=[DataRequired()])
    object_from_label = BooleanField('Create URI from label')

    # Perspective information
    perspective_certainty = FloatField('Certainty', validators=[DataRequired()])
    perspective_polarity = FloatField('Polarity', validators=[DataRequired()])
    perspective_sentiment = FloatField('Sentiment', validators=[DataRequired()])

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
    objects = StringField('Objects in the room', default='', validators=[DataRequired()])
    people = StringField('People in the room', default='', validators=[DataRequired()])

    submit = SubmitField('Submit capsule')


class ChatForm(FlaskForm):
    rewards = ['Total explicit triples', 'Total classes', 'Total predicates', 'Total semantic statements',
               'Total perspectives', 'Total sources', 'Total conflicts']

    chat_id = IntegerField('Chat ID', validators=[DataRequired()])
    speaker = StringField('Speaker', validators=[DataRequired()])
    reward = SelectField('Reward function', choices=rewards, validators=[DataRequired()])

    submit = SubmitField('Star chat')
