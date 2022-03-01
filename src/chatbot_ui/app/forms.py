from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField, SubmitField, IntegerField,FloatField
from wtforms.validators import DataRequired


class TurnForm(FlaskForm):
    subject_label = StringField('Subject label', validators=[DataRequired()])
    subject_types = StringField('Subject types', validators=[DataRequired()])
    subject_from_label = BooleanField('Create URI from label')

    predicate_label = StringField('Predicate label', validators=[DataRequired()])
    predicate_from_label = BooleanField('Create URI from label')

    object_label = StringField('Object label', validators=[DataRequired()])
    object_types = StringField('Object types', validators=[DataRequired()])
    object_from_label = BooleanField('Create URI from label')

    perspective_certainty = FloatField('Certainty', validators=[DataRequired()])
    perspective_polarity = FloatField('Polarity', validators=[DataRequired()])
    perspective_sentiment = FloatField('Sentiment', validators=[DataRequired()])

    submit = SubmitField('Submit capsule')


class ChatForm(FlaskForm):
    chat_id = IntegerField('Chat ID', validators=[DataRequired()])
    speaker = StringField('Speaker', validators=[DataRequired()])

    submit = SubmitField('Star chat')
