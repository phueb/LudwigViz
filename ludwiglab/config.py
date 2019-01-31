from pathlib import Path


class Dirs:
    lab = Path('/') / 'media' / 'lab'
    src = Path().cwd() / 'ludwigcluster'


class Interface:
    common_timepoint = 10


class Figs:
    NUM_PCS = 5




class AppConfigs(object):
    # miscellaneous
    APP_DEVICE = '/gpu:0'
    COMMON_TIMEPOINT = 1  # TODO put btn into app
    SAVE_PROBES_BAS_MAT = False
    DEFAULT_FIELD_STR = 'Your input here.'  # must not contain valid terms
    SWITCH_MODEL_COMPARISON = True
    SAVE_SIM_SIMMAT = False  # save sim simmat to file when requesting model comparison
    SAVE_CAT_FS_MAT = False  # save cat fs mat to file when requesting model_comparison

    # misc
    REINIT_TIMEPOINT = None
    
    MAX_NUM_MODELS = 10  # max number of models possible to train per config (helps keep memory usage low)
    DISK_USAGE_MAX = 90
    DEVICE = 'cpu' if platform.system() != 'Linux' else 'gpu'
    DEFAULT_FIELDNAMES = ['num_types', 'flavor']
    MAX_CLUSTER_TRAIN_CYCLES = 10  # max number of times a worker can complete a training
    
    

    # generating
    NUM_SAMPLES_LIST = [1, 50]
    GENERATE_NUM_WORDS = 10
    GENERATE_NUM_PHRASES = 5
    EXCLUDED_TERMS_FROM_GENERATING = []
    TASK_BTN_STR_TASK_NAME_DICT = {'Predict next terms': 'predict',
                                   'Verify Semantic Category Membership': 'sem-cat_task',
                                   'Verify Synonym Status': 'synonym_task',
                                   'Verify Antonym Status': 'match_task'}

    MODEL_BTN_NAME_INFO_DICT = {
        'avg_traj': ('Average Probes Trajectories', None),
        'cat_task_stats': ('Categorization Task Stats', None),
        'syn_task_stats': ('Synonym Task Stats', None),
        'dim_red': ('Dimensionality Reduction', None),
        'principal_comps': ('Principal Components of Activations', None),
        'cat_sim': ('Category Similarity Heatmap Dendrogram', None),
        'probe_sims': ('Probe Similarity Heatmap Dendrogram', 'probe'),
        'ba_by_probe': ('Balanced Accuracy Breakdown by Probe', None),
        'hierarch_cluster': ('Single-Category Hierarchical Clustering', None),
        'corpus_stats': ('Corpus Statistics', None),
        'compare_fs_trajs': ('Avg Probe F1-score Trajectories Comparison', 'probe'),
        'cum_freq_trajs': ('Probe Cumulative Frequency Trajectories', None),
        'term_freqs': ('Term Frequency Histogram', 'term'),
        'probe_context': ('Most frequent terms in context window', 'term'),
        'neighbors': ('Nearest Neighbors', 'term'),
        'probe_acts': ('Probe Activations Dendrogram Heatmap (not all acts shown)', 'probe'),
        'multi_hierarch_cluster': ('Multi-Category Hierrachical Clustering', 'cat'),
        'probes_by_act': ('Probes sorted by level of hidden unit activation', 'int'),
        'probe_last_cat_corr': ('Multi Probe Activations & last Activation', 'probe'),
        'probe_cat_corr': ('Probe Activation & Category Activations', 'probe'),
        'probe_probe_corr': ('Probe Activation & Alternate Probe Activations', 'probe'),
        'avg_probe_fs_corrs': ('Avg Probe F1-score Correlations', None),
        'avg_probe_pp_corrs': ('Avg Probe Perplexity Correlations', None),
        'abstraction': ('Abstraction Learning', None),
        'context_distance': ('Fit of Similarity Space to Distributional Statistics at different Distances', None),
        'probe_successors': ('Perplexity for Predicting Probe Successors', 'probe'),
        'cat_prediction': ('Category Prediction Goodness', None),
        'softmax_by_cat': ('Softmax Activation of Category when Predicting Terms', None),
        'phrase_pp': ('Perplexity over Phrase', 'term')}
    MULTI_GROUP_BTN_NAME_INFO_DICT = {
        'compare_to_sg': ('Sim Space Comparison', ['corpus_name', 'sem_probes_name', 'syn_probes_name']),
        'ba_by_cat': ('Probes Bal Accuracy by Category', ['corpus_name', 'sem_probes_name', 'syn_probes_name']),
        'cat_prediction': ('Category Prediction Goodness', []),
        'perplexity': ('Perplexity', []),
        'w_coherence': ('Weights Coherence', []),
        'probes_ba': ('Probes Balanced Accuracies', []),
        'probe_term_sims': ('Probe to Term Similarities', ['corpus_name', 'sem_probes_name', 'syn_probes_name']),
        'probe_pos_sim': ('Probe to POS Similarities', ['corpus_name', 'sem_probes_name', 'syn_probes_name']),
        'ap_by_cat': ('Category Prediction Average Precision', ['corpus_name', 'sem_probes_name', 'syn_probes_name']),
        'w_norm': ('Norm of Weights', []),
        'softmax': ('Softmax Probabilities', [])}
    TWO_GROUP_BTN_NAME_INFO_DICT = {
        'cat_probe_ba_diff': ('Balanced Accuracy Differences', ['corpus_name', 'sem_probes_name', 'syn_probes_name']),
        'correlations': ('Correlations', ['corpus_name', 'sem_probes_name', 'syn_probes_name'])}


# figs ////////////////////////////////////////////////////////////////////

class FigsConfigs(object):
    # global
    DPI = 96
    MAX_FIG_WIDTH = 7  # inches
    AXLABEL_FONT_SIZE = 14  # todo 12
    TICKLABEL_FONT_SIZE = 10
    LEG_FONTSIZE = 10
    LINEWIDTH = 2
    MARKERSIZE = 10
    FILL_ALPHA = 0.5

    # miscellaneous
    PCA_FREQ_THR = 100
    TAGS = ['UH', 'NN']
    PROBE_FREQ_YLIM = 1000
    NUM_PROBES_IN_QUARTILE = 5
    ALTERNATE_PROBES = ['cat', 'plate', 'dad', 'two', 'meat', 'tuesday', 'january']
    NUM_PCA_LOADINGS = 200
    CLUSTER_PCA_ITEM_ROWS = False
    CLUSTER_PCA_CAT_ROWS = True
    CLUSTER_PCA_CAT_COLS = True
    SKIPGRAM_MODEL_ID = 0
    NUM_PCS = 10  # how many principal_comps
    MAX_NUM_ACTS = 200  # max number of exemplar activations to use when working with exemplars
    LAST_PCS = [3]
    NUM_TIMEPOINTS_ACTS_CORR = 5
    DEFAULT_NUM_WALK_TIMEPOINTS = 5
    CAT_CLUSTER_XLIM = 1  # xlim for cat clusters (each fig should have same xlim enabling comparison)
    SVD_IS_TERMS = True
    NUM_EARLIEST_TOKENS = 100 * 1000
    NUM_PROBES_BAS_TIMEPOINTS = 6

    POS_TERMS_DICT = {'Interjection': ['okay', 'ow', 'whee', 'mkay', 'ohh', 'hey'],
                      'Onomatopoeia': ['moo', 'quack', 'meow', 'woof', 'vroom', 'oink'],
                      'Adverb': ['almost', 'afterwards', 'always', 'exactly', 'very', 'not'],
                      'Article': ['the', 'a', 'an'],
                      'Demonstrative': ['this', 'that', 'these', 'those'],
                      'Possessive': ['my', 'your', 'his', 'her', 'its', 'our'],
                      'Preposition': ['in', 'on', 'at', 'by', 'over', 'through', 'to'],
                      'Quantifier': ['few', 'any', 'much', 'many', 'most', 'some'],
                      'Conjunction': ['and', 'that', 'but', 'or', 'as', 'if'],
                      'Punctuation': ['.', '!', '?', ','],
                      'Number': ['two', 'three', 'seven', 'ten', 'hundred', 'thousand'],
                      'Pronoun': ['he', 'she', 'we', 'they', 'us', 'him']}
    PCA_ITEM_LIST1 = ['woof', 'oink', 'quack', 'meow', 'baa', 'mmm', 'whoa',
                      'zoom', 'hah', 'ohh', 'the', 'a', 'an', 'my', 'your',
                      'big', 'little', 'red', 'blue', 'good', 'couch', 'mirror',
                      'bathtub', 'table', 'rug', 'play', 'share', 'find', 'get', 'finish']
    PCA_ITEM_LIST2 = ['stir', 'chew', 'pour', 'drink', 'hold', 'juice', 'milk', 'cheerio', 'spaghetti',
                      'cracker', 'fork', 'spoon', 'napkin', 'knife', 'tissue',
                      'finger', 'hand', 'tongue', 'tooth', 'nose', 'grew', 'visit',
                      'ran', 'came', 'arrive', 'museum', 'aquarium', 'zoo', 'market',
                      'forest', 'hyena', 'flounder', 'beast', 'queen', 'prince', 'ago',
                      'merrily', 'bitsy', 'peep', 'weensie']

