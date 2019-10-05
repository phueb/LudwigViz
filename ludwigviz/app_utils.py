import base64
from io import BytesIO
import numpy as np
from wtforms.validators import ValidationError
from wtforms import Form, StringField

from ludwigviz import hostname


class LudwigVizEmptySubmission(Exception):
    def __init__(self, key, status_code=500):
        Exception.__init__(self)
        self.message = 'rnnlab: Did not find "{}" in session and no default provided.'.format(key)
        if status_code is not None:
            self.status_code = status_code


class LudwigVizAppError(Exception):
    def __init__(self, message, status_code=500):
        Exception.__init__(self)
        self.message = 'rnnlab: {}'.format(message)
        if status_code is not None:
            self.status_code = status_code


def make_template_dict(session):
    try:
        hub_mode = session['hub_mode']
    except KeyError:
        hub_mode = GlobalConfigs.HUB_MODES[0]
    try:
        timepoint = session['timepoint']
    except KeyError:
        timepoint = AppConfigs.COMMON_TIMEPOINT
    template_dict = {'hub_mode': hub_mode,
                     'hostname': hostname,
                     'timepoint': timepoint}
    return template_dict


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


def make_requested(request, session, key, default=None, verbose=True):
    # get get_fn
    get_fn = request.args.getlist if key.endswith('s') else request.args.get  # TODO test
    # request
    if get_fn(key + '-new'):
        if get_fn(key):
            result = session[key] = get_fn(key)
            print('requested new "{}": {}'.format(key, result)) if verbose else None
        else:  # might be None or [] if only submit button clicked
            print('no new "{}" found'.format(key))
            raise LudwigVizEmptySubmission(key)
    elif get_fn(key + '-default'):
        result = session[key] = default
        print('requested default "{}": {}'.format(key, result)) if verbose else None
    # fallback
    else:
        try:
            result = session[key] = session[key]
            print('fallback to session "{}": {}'.format(key, result)) if verbose else None
        except KeyError:
            if default:  # might be None or [] if only submit button clicked
                result = session[key] = default
                print('fallback to default "{}": {}'.format(key, result)) if verbose else None
            else:
                print('no fallback found for "{}"'.format(key))
                raise LudwigVizEmptySubmission(key)
    return result


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
        raise AttributeError('rnnlab: Invalid arg to "valid_type".')

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


def generate_terms(model,
                   terms,
                   task_id,
                   num_samples=50,
                   sort_column=None,
                   exclude_special_symbols=False):
    print('Generating terms with task_id {}...'.format(task_id))
    bptt_steps = model.configs_dict['bptt_steps']
    output_list = []
    if exclude_special_symbols:
        excluded_term_ids = [model.hub.train_terms.term_id_dict[symbol] for symbol in GlobalConfigs.SPECIAL_SYMBOLS]
    else:
        excluded_term_ids = []
    for i in range(AppConfigs.GENERATE_NUM_PHRASES):
        num_terms_in_phrase = len(terms)
        term_ids = [model.hub.train_terms.term_id_dict[term] for term in terms]
        while not len(term_ids) == AppConfigs.GENERATE_NUM_WORDS + len(terms):
            # get softmax probs
            x = np.asarray(term_ids)[:, np.newaxis][-bptt_steps:].T
            x2 = np.tile(np.eye(GlobalConfigs.NUM_TASKS)[task_id], [1, x.shape[1], 1])
            feed_dict = {model.graph.x: x, model.graph.x2: x2}
            softmax_probs = np.squeeze(model.sess.run(model.graph.softmax_probs, feed_dict=feed_dict))
            # calc new term_id and add
            samples = np.zeros([model.hub.train_terms.num_types], np.int)
            total_samples = 0
            while total_samples < num_samples:
                softmax_probs[0] -= sum(softmax_probs[:]) - 1.0  # need to compensate for float arithmetic
                new_sample = np.random.multinomial(1, softmax_probs)
                term_id_ = np.argmax(new_sample)
                if term_id_ not in excluded_term_ids:
                    samples += new_sample
                    total_samples += 1
            term_id = np.argmax(samples)
            term_ids.append(term_id)
        # convert phrase to string and add to output_list
        phrase_str = ' '.join([model.hub.train_terms.types[term_id] for term_id in term_ids[num_terms_in_phrase:]])
        output_list.append(phrase_str)
    # sort
    if sort_column is not None:
        output_list.sort(key=lambda x: x[sort_column])
    return output_list


def make_model_btn_name_info_dict(model_name):
    configs_dict = load_configs_dict(model_name)
    result = MODEL_BTN_NAME_INFO_DICT.copy()  # TODO where is this going to be made available?
    for btn_name, info in result.items():
        # insert any filtering logic here - like removing button names for tasks not trained on
        if btn_name and configs_dict['model_name']:  # arbitrary check - customize this
            continue
        else:
            continue
    return result


def load_configs_dict(model_name):
    configs_path = GlobalConfigs.RUNS_DIR / model_name / 'Configs' / 'configs_dict.npy'
    configs_dict = np.load(configs_path).item()
    return configs_dict


