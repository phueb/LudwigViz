import base64
from io import BytesIO
from wtforms.validators import ValidationError
from wtforms import Form, StringField



class LudwigVizEmptySubmission(Exception):
    def __init__(self, key, status_code=500):
        Exception.__init__(self)
        self.message = 'LudwigViz: Did not find "{}" in session and no default provided.'.format(key)
        if status_code is not None:
            self.status_code = status_code


def get_log_dicts_values(log_dicts, key):
    values = []
    for log_dict in log_dicts:
        desc_dict_value = log_dict[key]
        values.append(desc_dict_value)
    return values


def figs_to_imgs(*figs):
    imgs = []
    for fig in figs:
        print('Encoding fig...')
        figfile = BytesIO()
        fig.savefig(figfile, format='png')
        figfile.seek(0)
        img = base64.encodebytes(figfile.getvalue()).decode()
        imgs.append(img)
    return imgs


def make_form(model,
              request,
              default_str,
              valid_type):
    if not request.args:
        valid_set = []  # no need to do expensive validation if no request
        message = 'Please enter {}(s)'.format(valid_type)
        print('Making form with empty validator')
    elif valid_type == 'term':
        valid_set = model.hub.train_terms.types
        message = 'Found non-term.'
    elif valid_type == 'probe':
        valid_set = model.hub.probe_store.types
        message = 'Found non-probe.'
    elif valid_type == 'cat':
        valid_set = model.hub.probe_store.cats
        message = 'Found non-category'
    elif valid_type == 'int':  # for specifying hidden unit ids
        valid_set = [str(i) for i in range(model.embed_size)]
        message = 'Found non-integer'
    else:
        raise AttributeError('LudwigViz: Invalid arg to "valid_type".')

    def validator(form, field):
        if default_str in field.data:
            raise ValidationError(message)
        if not field.data:
            raise ValidationError('Input required')
        elif any(map(lambda x: x not in valid_set, field.data.split())):
            raise ValidationError(message)
        else:
            print('Form validated: "{}"'.format(field.data))

    class TermsForm(Form):
        field = StringField(validators=[validator])
        valid_type = ''

    result = TermsForm(request.args, field=default_str)
    return result
