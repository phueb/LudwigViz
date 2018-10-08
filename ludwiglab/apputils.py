import base64
import csv
from io import BytesIO
from operator import itemgetter
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from wtforms.validators import ValidationError
from wtforms import Form, StringField
from pathlib import Path
from itertools import chain


def make_log_dicts(logger, config_names):
    log_entry_dicts = logger.load_log()
    # df
    column_names = ['model_name'] + config_names + ['timepoint']
    column_names += ['num_saves'] if 'num_saves' not in config_names else []
    df = pd.DataFrame(data={column_name: [d[column_name] for d in log_entry_dicts]
                            for column_name in column_names})[column_names]
    # make log_dicts
    log_dicts = []
    for config_values, group_df in df.groupby(config_names):
        if not isinstance(config_values, tuple):
            config_values = [config_values]
        model_names = group_df['model_name'].tolist()
        log_dict = {'model_names': model_names,
                    'flavor': model_names[0].split('_')[1],
                    'model_desc': '\n'.join('{}={}'.format(config_name, config_value)
                                            for config_name, config_value in zip(config_names, config_values)),
                    'data_rows': [row.tolist() for ow_id, row in group_df.iterrows()]}
        log_dicts.append(log_dict)
    results = log_dicts[::-1]
    return results


def make_common_timepoint(logger, model_names_list, common_timepoint=config.Interface.common_timepoint):
    timepoints_list_list = []
    for model_names in model_names_list:
        timepoints_list = [logger.get_timepoints(model_name) for model_name in model_names]
        timepoints_list_list.append(timepoints_list)
    sets = [set(list(chain(*l))) for l in timepoints_list_list]
    # default
    if common_timepoint is not None and common_timepoint in set.intersection(*sets):
        print('Using default common timepoint: {}'.format(common_timepoint))
        return common_timepoint
    else:
        print('Did not find default common timepoint.')
    # common
    result = max(set.intersection(*sets))
    print('Found last common timepoint: {}'.format(result))
    return result


def make_log_df(summary_flavors, summary_hostnames):
    log_file_paths = [config.Dirs.log / f.name for f in config.Dirs.log.glob('log*.*')
                      if f.stem.split('_')[-1] in summary_hostnames]
    if not log_file_paths:
        result = pd.DataFrame()  # in case no matching log files found
    else:
        result = pd.DataFrame(pd.concat((pd.read_csv(f) for f in log_file_paths)))
        if not result.empty:
            result['flavor'] = result['model_name'].apply(lambda model_name: model_name.split('_')[1])
            result = result[result['timepoint'] == result['num_saves']]
            result = result[result['flavor'].isin(summary_flavors)]
    return result


class RnnlabEmptySubmission(Exception):
    def __init__(self, key, status_code=500):
        Exception.__init__(self)
        self.message = 'rnnlab: Did not find "{}" in session and no default provided.'.format(key)
        if status_code is not None:
            self.status_code = status_code


class RnnlabAppError(Exception):
    def __init__(self, message, status_code=500):
        Exception.__init__(self)
        self.message = 'rnnlab: {}'.format(message)
        if status_code is not None:
            self.status_code = status_code


def make_template_dict(session):
    hostname = GlobalConfigs.HOSTNAME
    try:
        hub_mode = session['hub_mode']
    except KeyError:
        hub_mode = GlobalConfigs.HUB_MODES[0]
    try:
        timepoint = session['timepoint']
    except KeyError:
        timepoint = AppConfigs.COMMON_TIMEPOINT
    template_dict = {'version': __version__,
                     'hub_mode': hub_mode,
                     'hostname': hostname,
                     'timepoint': timepoint}
    return template_dict


def get_requested_log_dicts(logger, session, request):
    config_names = make_requested(request, session, 'config_names', default=logger.manipulated_config_names)
    log_dicts = make_log_dicts(logger, config_names)
    log_dict_ids = [int(i) for i in request.args.getlist('log_dict_id')]
    requested_log_dicts = [log_dicts[i] for i in log_dict_ids]
    return requested_log_dicts


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
            raise RnnlabEmptySubmission(key)
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
                raise RnnlabEmptySubmission(key)
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
    result = AppConfigs.MODEL_BTN_NAME_INFO_DICT.copy()
    if configs_dict['task_layer_id'] + 1 > configs_dict['num_layers']:
        task_names = sorted(GlobalConfigs.TASK_NAME_QUESTION_DICT.keys())
        for key in ['{}_stats'.format(task_name) for task_name in task_names]:
            try:
                del result[key]
            except KeyError:  # unimplemented task
                pass
    return result


def write_acts_tsv_file(model):
    get_multi_probe_acts_df = model.get_multi_probe_prototype_acts_df()
    acts_dir = GlobalConfigs.RUNS_DIR / model.model_name / 'Word_Vectors'
    probe_cat_list = [model.hub.probe_store.probe_cat_dict[probe] for probe in model.hub.probe_store.types]
    probe_list_df = pd.DataFrame(data={'probe': model.hub.probe_store.types, 'cat': probe_cat_list})
    probe_list_df.to_csv(acts_dir / 'probe_list.tsv', columns=['probe', 'cat'],
                         header=True, index=False, sep='\t', quoting=csv.QUOTE_NONE)
    get_multi_probe_acts_df.to_csv(acts_dir / 'word_vectors.tsv',
                                   header=False, index=False, sep='\t', quoting=csv.QUOTE_NONE)
    print('Exported word vectors to {}'.format(acts_dir))


def write_probe_neighbors_files(model, num_neighbors=10):
    neighbors_dir = GlobalConfigs.RUNS_DIR / model.model_name / 'Neighbors'
    for cat in model.hub.probe_store.cats:
        # init writer
        file_name = '{}_neighbors_{}.csv'.format(cat, model.mb_name)
        f = (neighbors_dir / file_name).open()
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['probe', 'rank', 'neighbor', 'sim', 'taxo', 'thema'])
        thema_pairing = ''
        for probe in model.hub.probe_store.cat_probe_list_dict[cat]:
            # get neighbors
            term_id = model.hub.train_terms.term_id_dict[probe]
            term_sims = model.term_simmat[term_id]
            neighbor_tuples = [(model.hub.train_terms.types[term_id], term_sim)
                               for term_id, term_sim in enumerate(term_sims)
                               if model.hub.train_terms.types[term_id] != probe]
            neighbor_tuples = sorted(neighbor_tuples, key=itemgetter(1), reverse=True)[:num_neighbors]
            # write to file
            for n, (neighbor, sim) in enumerate(neighbor_tuples):
                taxo_pairing = ''
                writer.writerow([probe, n, neighbor, sim, taxo_pairing, thema_pairing])
    print('Exported neighbors files to {}'.format(neighbors_dir))


def write_pca_loadings_files(model, is_terms=FigsConfigs.SVD_IS_TERMS, freq_thr=FigsConfigs.PCA_FREQ_THR):
    if is_terms:
        acts_mat = model.term_acts_mat
        terms = model.hub.train_terms.types
    else:
        acts_mat = model.get_multi_probe_prototype_acts_df().values
        terms = model.hub.probe_store.types
    pca_model = PCA(n_components=FigsConfigs.NUM_PCS)
    pcs = pca_model.fit_transform(acts_mat)
    # init writer
    file_name = 'pca_loadings_{}_'.format(model.mb_name)
    file_name += 'terms_' if is_terms else 'probes_'
    file_name += 'min_freq_{}'.format(freq_thr)
    file_name += '.csv'
    f = (GlobalConfigs.RUNS_DIR / model.model_name / 'PCA' / file_name).open()
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['component', 'word', 'loading', 'train_freq'])
    # write
    for pc_id in range(FigsConfigs.NUM_PCS):
        # make rows
        pc = pcs[:, pc_id]
        sorted_pc, sorted_terms = list(zip(*sorted(zip(pc, terms), key=itemgetter(0))))
        rows = [[pc_id, term, loading, model.hub.train_terms.term_freq_dict[term]]
                for term, loading in zip(sorted_terms, sorted_pc)
                if model.hub.train_terms.term_freq_dict[term] > freq_thr]
        # write rows
        for row in rows:
            writer.writerow(row)
    print('Exported neighbors files.')


def load_configs_dict(model_name):
    configs_path = GlobalConfigs.RUNS_DIR / model_name / 'Configs' / 'configs_dict.npy'
    configs_dict = np.load(configs_path).item()
    return configs_dict


def save_probes_fs_mat(model_groups, model_descs):
    # make group_trajs_mats
    group_trajs_mats = []
    row_labels = []
    x = model_groups[0][0].get_data_step_axis()
    for models, model_desc in zip(model_groups, model_descs):
        traj1 = []
        traj2 = []
        traj3 = []
        for model in models:
            traj1.append(model.get_traj('probes_fs_o'))
            traj2.append(model.get_traj('probes_fs_s'))
            traj3.append(model.get_traj('probes_fs_n'))
            row_labels.append(model_desc.replace(' ',  '_'))
        traj_mat1 = np.asarray([traj[:len(x)] for traj in traj1])
        traj_mat2 = np.asarray([traj[:len(x)] for traj in traj2])
        traj_mat3 = np.asarray([traj[:len(x)] for traj in traj3])
        group_trajs_mat = np.hstack((traj_mat1, traj_mat2, traj_mat3))
        group_trajs_mats.append(group_trajs_mat)
        print('Num "{}": {}'.format(model_desc, len(group_trajs_mat)))
    # df
    num_timepoints = len(x)
    col_labels = []
    for context_type in ['ordered', 'shuffled', 'none']:
        col_labels += ['{}_{}'.format(context_type, timepoint) for timepoint in range(num_timepoints)]
    df_mat = np.vstack(group_trajs_mats)
    df = pd.DataFrame(df_mat, columns=col_labels, index=row_labels)
    # save
    path = Path.home() / 'probes_fs_mat_{}.csv'.format(GlobalConfigs.HOSTNAME)
    df.to_csv(path)
    print('Saved probes_fs_mat.csv to {}'.format(path))
