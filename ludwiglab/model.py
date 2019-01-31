import tensorflow as tf
import time
from cached_property import cached_property
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
import pyprind

from configs import GlobalConfigs, AppConfigs
from directgraph import DirectGraph
from hub import Hub
from options import default_configs_dict


class Model:  # TODO strip out all user-specific stuff and keep the rest in a base class to inherit from
    """
    Stores data associated with trained model
    """

    def __init__(self, model_name, timepoint, hub=None):  # TODO turn configs_Dcit into class with properties to simplify development
        self.model_name = model_name
        self.timepoint = timepoint
        self.mb_name = self.ckpt_mb_names[timepoint]
        self.configs_dict = self.load_configs_dict()
        self.flavor = self.configs_dict['flavor']
        self.bptt_steps = self.configs_dict['bptt_steps']
        self.num_docs = self.configs_dict['num_parts']
        self.num_iterations = self.configs_dict['num_iterations']
        self.mb_size = self.configs_dict['mb_size']
        self._hub = hub
        print('Loaded model {} @ {} timepoint {}'.format(self.model_name, self.mb_name, self.timepoint))

    def load_configs_dict(self):
        configs_path = GlobalConfigs.RUNS_DIR / self.model_name / 'Configs' / 'configs_dict.npy'
        configs_dict = dict(np.load(configs_path).item())
        return configs_dict

    def make_graph(self, mb_name=None):
        start = time.time()
        if mb_name is None:
            mb_name = self.mb_name
        # graph
        tf.reset_default_graph()  # need this because previous graph persists across requests
        if 'graph' not in self.__dict__:
            graph = DirectGraph(self.configs_dict, self.task_names, self.hub, device=AppConfigs.APP_DEVICE)
        else:
            graph = self.graph
        # restore weights
        path = GlobalConfigs.RUNS_DIR / self.model_name / 'Checkpoints' / 'checkpoint_mb_{}.ckpt'.format(mb_name)
        self.ckpt_saver.restore(self.sess, str(path))
        print('Restored weights saved at mb {} in {:.2f} secs'.format(mb_name, time.time() - start))
        return graph

    @cached_property
    def hub(self):
        result = self._hub or Hub(**self.configs_dict)
        return result

    @cached_property
    def sess(self):
        result = tf.Session()
        return result

    @cached_property
    def ckpt_saver(self):
        result = tf.train.Saver()
        return result

    @cached_property
    def graph(self):
        result = self.make_graph()
        return result

    @cached_property
    def doc_id(self):  # approximate doc id based on mb_name
        mb_ratio = self.timepoint / self.configs_dict['num_saves']
        doc_id = int(self.num_docs * mb_ratio)
        return doc_id

    @cached_property
    def task_names(self):  # TODO test
        task_names = [task_name for task_name in GlobalConfigs.TASK_NAME_QUESTION_DICT.keys()
                      if self.configs_dict[task_name] != default_configs_dict[task_name]]
        return task_names

    @property
    def ckpt_mb_names(self):
        # get from checkpoints (which are created before dfs)
        path = GlobalConfigs.RUNS_DIR / self.model_name / 'Checkpoints'
        saved_mb_names = [re.search('\d+', p.name).group() for p in path.rglob('*.meta')  # TODO test
                          if 'task' not in p.name]
        saved_mb_names = sorted(saved_mb_names)
        return saved_mb_names

    @property
    def timepoints(self):
        timepoints = [n for n, saved_mb_name in enumerate(self.ckpt_mb_names)]
        return timepoints

    @cached_property
    def term_simmat(self):
        path = GlobalConfigs.RUNS_DIR / self.model_name / 'Token_Simmat' / 'term_simmat_{}.npy'.format(self.mb_name)
        result = np.load(path)
        if np.any(np.isnan(result)):
            print('rnnlab WARNING: Found NaNs in term_simmat.')
        return result

    @cached_property
    def probe_simmat_o(self):
        result = cosine_similarity(self.get_multi_probe_prototype_acts_df().values)
        return result

    @cached_property
    def term_acts_mat(self):
        path = GlobalConfigs.RUNS_DIR / self.model_name / 'Token_Simmat' / 'term_acts_mat_{}.npy'.format(self.mb_name)
        result = np.load(path)
        return result

    # ////////////////////////////////////////////////////////////////////////// probes

    @cached_property
    def median_probe_cg_list(self):
        result = [self.hub.calc_median_term_cg(probe) for probe in self.hub.probe_store.types]
        return result

    @cached_property
    def avg_probe_loc_list(self):
        result = [self.hub.calc_avg_reordered_loc(probe) for probe in self.hub.probe_store.types]
        return result

    @cached_property
    def avg_probe_loc_asymmetry_list(self):
        result = [self.hub.calc_loc_asymmetry(probe) for probe in self.hub.probe_store.types]
        return result

    @cached_property
    def probe_freq_list(self):
        result = [len(self.hub.term_reordered_locs_dict[probe]) for probe in self.hub.probe_store.types]
        return result

    @cached_property
    def probe_cat_freq_list(self):
        result = [np.sum([len(self.hub.term_reordered_locs_dict[p])
                          for p in self.hub.probe_store.cat_probe_list_dict[
                              self.hub.probe_store.probe_cat_dict[probe]]])
                  for probe in self.hub.probe_store.types]
        return result

    # ////////////////////////////////////////////////////////////////////////// dfs

    @cached_property
    def globals_traj_df(self):
        path = GlobalConfigs.RUNS_DIR / self.model_name / 'Data_Frame' / 'globals_traj_df.h5'
        result = pd.read_hdf(path)
        return result

    @cached_property
    def acts_df(self):
        path = GlobalConfigs.RUNS_DIR / self.model_name / 'Data_Frame' / 'acts_df_{}.h5'.format(self.mb_name)
        result = pd.read_hdf(path, key='acts_{}_df'.format(self.hub.mode))
        return result

    @cached_property
    def fs_traj_df(self):
        path = GlobalConfigs.RUNS_DIR / self.model_name / 'Data_Frame' / 'fs_traj_df.h5'
        result = pd.read_hdf(path, key='fs_{}_traj_df'.format(self.hub.mode))
        return result

    @cached_property
    def ba_traj_df(self):
        path = GlobalConfigs.RUNS_DIR / self.model_name / 'Data_Frame' / 'ba_traj_df.h5'
        result = pd.read_hdf(path, key='ba_{}_traj_df'.format(self.hub.mode))
        return result

    @cached_property
    def pp_traj_df(self):
        path = GlobalConfigs.RUNS_DIR / self.model_name / 'Data_Frame' / 'pp_traj_df.h5'
        result = pd.read_hdf(path, key='pp_{}_traj_df'.format(self.hub.mode))
        return result

    @cached_property
    def cat_task_traj_df(self):
        path = GlobalConfigs.RUNS_DIR / self.model_name / 'Data_Frame' / 'cat_task_traj_df.h5'
        with pd.HDFStore(path, mode='r') as store:
            task_traj_df = store.select('cat_task_{}_traj_df'.format(self.hub.mode))  # TODO test
        return task_traj_df

    # ////////////////////////////////////////////////////////////////////////// evals

    @cached_property
    def avg_probe_cat_pred_goodness_list(self, max_windows=10, clip=100):
        """
        Return list of measure of how well a probe category is predicted relative to all probe categories
        """
        other_term_ids = [self.hub.train_terms.term_id_dict[probe]
                          for probe in self.hub.probe_store.types]
        result = []
        for probe in self.hub.probe_store.types:
            windows = self.hub.get_term_id_windows(probe, roll_left=True, num_samples=max_windows)
            x = np.vstack(windows)
            feed_dict = {self.graph.x: x}
            softmax_probs = self.sess.run(self.graph.softmax_probs, feed_dict=feed_dict)
            cat = self.hub.probe_store.probe_cat_dict[probe]
            sem_term_ids = [self.hub.train_terms.term_id_dict[term]
                            for term in self.hub.probe_store.cat_probe_list_dict[cat]]
            sem_term_ids_filtered = sem_term_ids
            other_term_ids_filtered = list(set(other_term_ids) - set(sem_term_ids))
            sem_probs = softmax_probs[:, sem_term_ids_filtered]
            other_probs = softmax_probs[:, other_term_ids_filtered]
            avg_probe_cat_pred_goodness = np.clip(sem_probs.mean().mean() / other_probs.mean().mean(), -clip, clip)
            result.append(avg_probe_cat_pred_goodness)
        return result

    @cached_property
    def avg_probe_sim_list(self):
        avg_term_sims = np.mean(self.term_simmat, axis=1)
        result = [avg_term_sims[self.hub.train_terms.term_id_dict[probe]]
                  for probe in self.hub.probe_store.types]
        return result

    @cached_property
    def avg_probe_noun_sim_n_list(self):
        result = self.calc_avg_probe_pos_sim_n_list('noun')
        return result

    @cached_property
    def avg_probe_verb_sim_n_list(self):
        result = self.calc_avg_probe_pos_sim_n_list('verb')
        return result

    @cached_property
    def avg_probe_pronoun_sim_n_list(self):
        result = self.calc_avg_probe_pos_sim_n_list('pronoun')
        return result

    @cached_property
    def avg_probe_pp_list(self):
        result = self.pp_traj_df[[column_name for column_name in self.pp_traj_df
                                  if column_name.endswith('')]].iloc[self.timepoint].tolist()
        return result

    @cached_property
    def avg_probe_fs_o_list(self):
        avg_probe_p_o_list = self.fs_traj_df[[column_name for column_name in self.fs_traj_df
                                              if column_name.endswith('_p_o')]].iloc[self.timepoint].values
        avg_probe_r_o_list = self.fs_traj_df[[column_name for column_name in self.fs_traj_df
                                              if column_name.endswith('_r_o')]].iloc[self.timepoint].values
        result = 2 * (avg_probe_p_o_list * avg_probe_r_o_list) / (avg_probe_p_o_list + avg_probe_r_o_list)
        return result

    @cached_property
    def avg_probe_ba_o_list(self):
        avg_probe_p_o_list = self.ba_traj_df[[column_name for column_name in self.ba_traj_df
                                              if column_name.endswith('_p_o')]].iloc[self.timepoint].values
        avg_probe_r_o_list = self.ba_traj_df[[column_name for column_name in self.ba_traj_df
                                              if column_name.endswith('_r_o')]].iloc[self.timepoint].values
        result = (avg_probe_p_o_list + avg_probe_r_o_list) / 2  # * 100  # TODO multiply by 100?
        return result

    def sort_cats_by_ba(self, context_type):
        avg_probe_p_o_list = self.ba_traj_df[[column_name for column_name in self.ba_traj_df
                                              if column_name.endswith('p_' + context_type[0])]].iloc[
            self.timepoint].values
        avg_probe_r_o_list = self.ba_traj_df[[column_name for column_name in self.ba_traj_df
                                              if column_name.endswith('r_' + context_type[0])]].iloc[
            self.timepoint].values
        avg_probe_ba_list = (avg_probe_p_o_list + avg_probe_r_o_list) / 2
        probe_cat_list = [self.hub.probe_store.probe_cat_dict[probe]
                          for probe in self.hub.probe_store.types]
        df = pd.DataFrame(data={'avg_probe_ba': avg_probe_ba_list,
                                'cat': probe_cat_list})
        result = df.groupby('cat').mean().sort_values(by='avg_probe_ba').index.tolist()
        return result

    def get_sorted_cat_avg_probe_ba_lists(self, cats_to_sort_by, context_type='ordered'):
        if context_type == 'shuffled':
            avg_probe_fs_list = self.avg_probe_ba_s_list
        elif context_type == 'none':
            avg_probe_fs_list = self.avg_probe_ba_n_list
        elif context_type == 'ordered':
            avg_probe_fs_list = self.avg_probe_ba_o_list
        else:
            raise AttributeError('rnnlab: Invalid arg to "context_type".')
        assert len(self.hub.probe_store.types) == len(avg_probe_fs_list)
        result = []
        for cat in cats_to_sort_by:
            cat_avg_probe_fs_list = [avg_probe_fs for probe, avg_probe_fs
                                     in zip(self.hub.probe_store.types, avg_probe_fs_list)
                                     if probe in self.hub.probe_store.cat_probe_list_dict[cat]]
            result.append(cat_avg_probe_fs_list)
        return result

    def get_traj(self, name):
        if name == 'test_pp':
            result = self.globals_traj_df['test_pp'].tolist()
        elif name == 'train_pp':
            result = self.globals_traj_df['train_pp'].tolist()
        elif name == 'probes_pp':
            result = self.pp_traj_df[[column_name for column_name in self.pp_traj_df]].mean(axis=1).tolist()
        elif name == 'probes_fs_p_o':
            result = self.fs_traj_df[[column_name for column_name in self.fs_traj_df
                                      if column_name.endswith('_p_o')]].mean(axis=1).tolist()
        elif name == 'probes_fs_r_o':
            result = self.fs_traj_df[[column_name for column_name in self.fs_traj_df
                                      if column_name.endswith('_r_o')]].mean(axis=1).tolist()
        elif name == 'probes_fs_o':
            avg_probe_p_o_list = self.fs_traj_df[[column_name for column_name in self.fs_traj_df
                                                  if column_name.endswith('_p_o')]].mean(axis=1).values
            avg_probe_r_o_list = self.fs_traj_df[[column_name for column_name in self.fs_traj_df
                                                  if column_name.endswith('_r_o')]].mean(axis=1).values
            result = 2 * (avg_probe_p_o_list * avg_probe_r_o_list) / (avg_probe_p_o_list + avg_probe_r_o_list)
        elif name == 'probes_ba_p_o':
            result = self.ba_traj_df[[column_name for column_name in self.ba_traj_df
                                      if column_name.endswith('_p_o')]].mean(axis=1).tolist()
        elif name == 'probes_ba_r_o':
            result = self.ba_traj_df[[column_name for column_name in self.ba_traj_df
                                      if column_name.endswith('_r_o')]].mean(axis=1).tolist()
        elif name == 'probes_ba_o':
            avg_probe_p_o_list = self.ba_traj_df[[column_name for column_name in self.ba_traj_df
                                                  if column_name.endswith('_p_o')]].mean(axis=1).values
            avg_probe_r_o_list = self.ba_traj_df[[column_name for column_name in self.ba_traj_df
                                                  if column_name.endswith('_r_o')]].mean(axis=1).values
            result = (avg_probe_p_o_list + avg_probe_r_o_list) / 2
        else:
            raise AttributeError('rnnlab: Invalid argument passed to "name": "{}".'.format(name))
        return result

    def get_data_step_axis(self):
        result = sorted(np.asarray([int(mb_name) for mb_name in self.ba_traj_df.index]))  # TODO sort?
        return result

    # ////////////////////////////////////////////////////////////////////////// acts

    def get_single_probe_exemplar_acts_df(self, probe):
        df_ids = self.acts_df.index == probe
        result = self.acts_df.loc[df_ids]
        return result

    def get_multi_probe_prototype_acts_df(self):
        print('Loading prototype probe activations...')
        result = self.acts_df.groupby(self.acts_df.index, sort=True).mean()
        return result

    def get_multi_probe_exemplar_acts_df(self, num_exemplars):
        print('Loading exemplar probe activations using {} exemplars each...'.format(num_exemplars))
        result = self.acts_df.groupby(self.acts_df.index, sort=True).head(num_exemplars)
        return result

    def get_single_cat_probe_prototype_acts_df(self, cat):
        cat_probes = self.hub.probe_store.cat_probe_list_dict[cat]
        df_filtered = self.acts_df.loc[self.acts_df.index.isin(cat_probes)]
        result = df_filtered.groupby(df_filtered.index, sort=True).mean()
        return result

    def get_multi_cat_prototype_acts_df(self):
        result = self.acts_df.groupby(self.acts_df.index, sort=True).mean()
        return result

    # ////////////////////////////////////////////////////////////////////////// misc  # TODO move below somewhere else?

    def make_phrase_pps(self, terms, first_term=GlobalConfigs.OOV_SYMBOL):
        print('Making phrase_pps...')
        terms = [first_term] + terms  # to get pp value for very first term
        num_terms = len(terms)
        task_id = 0
        pps = []
        for n in range(num_terms):
            term_ids = [self.hub.train_terms.term_id_dict[term] for term in
                        terms[:n + 2]]  # add to to compensate for 0index and y
            window_mat = np.asarray(term_ids)[:, np.newaxis][-self.bptt_steps:].T
            x, y = np.split(window_mat, [-1], axis=1)
            x2 = np.tile(np.eye(GlobalConfigs.NUM_TASKS)[task_id], [1, x.shape[1], 1])
            feed_dict = {self.graph.x: x, self.graph.x2: x2, self.graph.y: y}
            pp = np.asscalar(self.sess.run(self.graph.pps, feed_dict=feed_dict))
            pps.append(pp)
        pps = pps[:-1]  # compensate for '.' insertion
        return pps

    def make_softmax_by_cat_mat(self, cat_type,
                                max_x=2 ** 16, max_term_windows=16, remove_punct=False):  # memory maxes out > 100
        # make cat_term_ids_list
        if cat_type == 'pos':
            cats = [cat + 's' for cat in sorted(GlobalConfigs.POS_TAGS_DICT.keys())]
            if remove_punct:
                cats.remove('punctuations')
            cat_terms_list = [getattr(self.hub, cat) for cat in cats]
        elif cat_type == 'probes':
            cats = self.hub.probe_store.cats
            cat_terms_list = [self.hub.probe_store.cat_probe_list_dict[cat] for cat in cats]
        else:
            raise AttributeError('rnnlab: Invalid arg to "cat_type".')
        cat_term_ids_list = [[self.hub.train_terms.term_id_dict[term] for term in cat_terms]
                             for cat_terms in cat_terms_list]
        # make mat
        num_cats = len(cat_terms_list)
        result = np.zeros((num_cats, num_cats))
        for row_id, cat_terms in enumerate(cat_terms_list):
            contexts_list = [self.hub.get_term_id_windows(cat_term, roll_left=True, num_samples=max_term_windows)
                             for cat_term in cat_terms]
            contexts = [c for cs in contexts_list for c in cs]
            if not contexts:
                print('rnnlab WARNING: Did not find category "{}"'.format(cats[row_id]))
                continue
            x = np.vstack(contexts)[:max_x, :]
            feed_dict = {self.graph.x: x}
            softmax_probs = self.sess.run(self.graph.softmax_probs, feed_dict=feed_dict)
            avg_softmax_probs = softmax_probs.mean(axis=0)
            result[row_id, :] = [np.sum(avg_softmax_probs[cat_term_ids]) for cat_term_ids in cat_term_ids_list]
        return result

    def calc_avg_cat_context_cat_vs_item_goodness_list(self, context_dist=2, min_context_f=10):
        def calc_kl_divergence(p, q, epsilon=0.00001):
            pe = p + epsilon
            qe = q + epsilon
            divergence = np.sum(pe * np.log2(pe / qe))
            return divergence

        cat_y_dict = {cat: [] for cat in self.hub.probe_store.cats}
        context_d = {}
        no_context_d = {'freq_by_probe': {probe: 0.0 for probe in self.hub.probe_store.types},
                        'freq_by_cat': {cat: 0.0 for cat in self.hub.probe_store.cats}}
        max_num_tokens = int(self.mb_name) * self.mb_size // self.num_iterations
        pbar = pyprind.ProgBar(max_num_tokens)
        for loc, token in enumerate(self.hub.reordered_tokens[:max_num_tokens]):
            pbar.update()
            if token in self.hub.probe_store.types:
                # context
                context = tuple(self.hub.reordered_tokens[loc + d] for d in range(-context_dist, 0) if d != 0)
                # update context_d
                cat = self.hub.probe_store.probe_cat_dict[token]
                if context not in context_d:
                    context_d[context] = {'freq_by_probe': {probe: 0.0 for probe in self.hub.probe_store.types},
                                          'freq_by_cat': {cat: 0.0 for cat in self.hub.probe_store.cats},
                                          'freq': 1,
                                          'y': 0}  # must be numeric for sorting
                    context_d[context]['freq_by_probe'][token] += 1
                    context_d[context]['freq_by_cat'][cat] += 1
                else:
                    context_d[context]['freq_by_probe'][token] += 1
                    context_d[context]['freq_by_cat'][cat] += 1
                    context_d[context]['freq'] += 1
                # update no_context_d
                no_context_d['freq_by_probe'][token] += 1
                no_context_d['freq_by_cat'][cat] += 1
                # y
                if context_d[context]['freq'] > min_context_f:
                    # observed
                    probes_observed_probs = np.array(
                        [context_d[context]['freq_by_probe'][probe] / context_d[context]['freq']
                         for probe in self.hub.probe_store.types])
                    # expected
                    most_common_cat = sorted(self.hub.probe_store.cats,
                                             key=lambda c: context_d[context]['freq_by_cat'][c])[-1]
                    cat_probes = self.hub.probe_store.cat_probe_list_dict[most_common_cat]
                    probes_expected_probs = np.array(
                        [no_context_d['freq_by_probe'][probe] / no_context_d['freq_by_cat'][most_common_cat]
                         if probe in cat_probes else 0.0
                         for probe in self.hub.probe_store.types])
                    # y
                    y = calc_kl_divergence(probes_expected_probs, probes_observed_probs)  # asymmetric
                    cat_y_dict[cat].append(y)
        # result
        result = []
        for cat in self.hub.probe_store.cats:
            avg_cat_context_cat_sensitivity = np.mean(cat_y_dict[cat])
            result.append(avg_cat_context_cat_sensitivity)
        return result

    def calc_probes_cat_pred_goodness(self):
        try:
            del self.__dict__['avg_probe_cat_pred_goodness_list']  # invalidate cache
        except KeyError:
            pass
        result = np.mean(self.avg_probe_cat_pred_goodness_list)
        return result

    def calc_softmax_vec(self, window_size=1, num_random_locs=4096 * 8):
        if window_size is None:
            window_size = self.bptt_steps
        # get windows from corpus
        np.random.seed(4)
        random_locs = np.random.randint(self.bptt_steps, self.hub.train_terms.num_tokens,
                                        size=num_random_locs)
        windows_list = [self.hub.train_terms.token_ids[loc - self.bptt_steps: loc]
                        for loc in random_locs]
        x = np.vstack(windows_list)[:, :window_size]
        feed_dict = {self.graph.x: x}
        result = self.sess.run(self.graph.softmax_probs, feed_dict=feed_dict).mean(axis=0)
        return result

    def calc_norm_of_w(self, w_name, norm_order=None):
        """
        Calculates matrix norm of weight matrix specified by "w_name" as a rough estimate of fit complexity
        """
        if w_name == 'wx':
            w = self.graph.wx
            axis = 1
        elif w_name == 'wy':
            w = self.graph.wy
            axis = 0
        else:
            raise AttributeError('rnnlab: Invalid arg to "w_name".')
        result = np.linalg.norm(self.sess.run(w), ord=norm_order, axis=axis)
        return result

    def calc_w_sim(self, term_ids, w_name):
        """
        Calculate average of pairwise similarities between weights corresponding to "term_ids"
        """
        np.random.seed(1)
        random_term_ids = np.random.choice(self.hub.train_terms.num_types, size=len(term_ids), replace=False)
        if w_name == 'wx':
            w = self.sess.run(self.graph.wx)
            w_filtered = w[term_ids, :]
            w_random = w[random_term_ids, :]
        elif w_name == 'wy':
            w = self.sess.run(self.graph.wy)
            w_filtered = w[:, term_ids].T
            w_random = w[:, random_term_ids].T
        else:
            raise AttributeError('rnnlab: Invalid arg to "w_name"')
        if len(term_ids) == 0:  # no punctuations found, for example
            result = np.nan
        else:
            self_sim = cosine_similarity(w_filtered, w_filtered).mean().mean()
            random_sim = cosine_similarity(w_random, w_random).mean().mean()
            result = self_sim / abs(random_sim)
        return result

    def calc_windows_sim(self, randomized_position, pos=None,
                         context_is_uniform=False, window_size=None, num_terms=100, max_windows=10, is_hidden=True):
        np.random.seed(2)
        if window_size is None:
            window_size = self.bptt_steps
        assert window_size in range(self.bptt_steps + 1)
        # make windows_mat
        if pos is None:
            terms = np.random.choice(self.hub.train_terms.types, size=num_terms, replace=False)
        elif pos in GlobalConfigs.POS_TAGS_DICT.keys():
            terms = getattr(self.hub, pos + 's')
        else:
            raise AttributeError('rnnlab: Invalid arg to "pos".')
        windows_list = []
        for n, term in enumerate(terms[:num_terms]):
            windows = self.hub.get_term_id_windows(term, num_samples=max_windows)
            try:
                windows_list.append(windows[-max_windows:])
            except IndexError:  # did not find any windows
                print('Did not find windows for "{}"'.format(term))
        windows_mat = np.vstack(windows_list)[:, -window_size:]
        if context_is_uniform:
            windows_mat = np.repeat(windows_mat[:, -1][:, np.newaxis], window_size, axis=1)
        if randomized_position is not None:
            windows_mat[:, -randomized_position: -1] = np.random.randint(
                self.hub.train_terms.num_types, size=(windows_mat.shape[0], randomized_position - 1))
        # make acts_mat
        feed_dict = {self.graph.x: windows_mat}
        if is_hidden:
            acts_mat = self.sess.run(self.graph.representation, feed_dict=feed_dict)
        else:
            acts_mat = self.sess.run(self.graph.softmax_probs, feed_dict=feed_dict)
        # sim
        result = cosine_similarity(acts_mat).mean().mean()
        return result

    def calc_avg_probe_pos_sim_n_list(self, pos):
        pos_term_ids = [self.hub.train_terms.term_id_dict[term] for term in getattr(self.hub, pos + 's')]
        probe_term_ids = [self.hub.train_terms.term_id_dict[probe] for probe in self.hub.probe_store.types]
        term_acts_n = self.graph.wx.eval(self.sess)  # define graph before sess
        term_sims_n = cosine_similarity(term_acts_n)
        result = term_sims_n[probe_term_ids, :][:, pos_term_ids].mean(axis=1)
        return result

    def calc_pos1_pos2_sim_and_all_but_pos2_sim_n(self, pos1, pos2, w_name):
        # make term_ids
        if pos1 == 'probe':
            pos1_term_ids = [self.hub.train_terms.term_id_dict[probe] for probe in self.hub.probe_store.types]
            pos2_term_ids = pos1_term_ids
        else:
            pos1_term_ids = [self.hub.train_terms.term_id_dict[term] for term in getattr(self.hub, pos1 + 's')]
            pos2_term_ids = [self.hub.train_terms.term_id_dict[term] for term in getattr(self.hub, pos2 + 's')]
        other_pos_term_ids = list(set(np.arange(self.hub.train_terms.num_types)) - set(pos2_term_ids))
        # calc term_sims
        if w_name == 'wy':
            term_acts = self.graph.wy.eval(self.sess).T  # define graph before sess
            term_sims = cosine_similarity(term_acts)
        elif w_name == 'wx':
            term_acts = self.graph.wx.eval(self.sess)
            term_sims = cosine_similarity(term_acts)
        else:
            raise AttributeError('rnnlab: Invalid arg to "w_name".')
        # calc
        pos1_pos2_sim = term_sims[pos1_term_ids, :][:, pos2_term_ids].mean().mean()
        pos1_all_but_pos2_sim = term_sims[pos1_term_ids, :][:, other_pos_term_ids].mean().mean()
        result = [pos1_pos2_sim, pos1_all_but_pos2_sim]
        return result


