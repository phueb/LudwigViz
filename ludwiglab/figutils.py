from operator import itemgetter
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from math import gcd
import sys
import time
from scipy.stats import linregress
import matplotlib.pyplot as plt

from configs import GlobalConfigs, FigsConfigs


def split_probes_into_quartiles_by_fs(model):
    avg_probe_fs_list = model.avg_probe_fs_o_list
    sorted_avg_fss, sorted_probes = zip(*sorted(zip(
        avg_probe_fs_list, model.hub.probe_store.types), key=itemgetter(0)))
    sorted_probes = list(sorted_probes)
    while True:
        try:
            q1, q2, q3, q4 = np.split(np.asarray(sorted_probes), 4)
        except ValueError:
            sorted_probes.pop(0)
        else:
            break
    return q1, q2, q3, q4


def calc_shannon_entropy(l):
    prob_dict = {i: list(l).count(i) / float(len(l)) for i in l}
    probs = np.array(list(prob_dict.values()))
    return - probs.dot(np.log2(probs))


def calc_cat_sim_mat(model):
    # probe simmat
    probes_acts_df = model.get_multi_probe_prototype_acts_df()
    probe_simmat = cosine_similarity(probes_acts_df.values)
    # make category sim dict
    num_probes = len(model.hub.probe_store.types)
    num_cats = len(model.hub.probe_store.cats)
    cat_sim_dict = {cat_outer: {cat_inner: [] for cat_inner in model.hub.probe_store.cats}
                    for cat_outer in model.hub.probe_store.cats}
    for i in range(num_probes):
        probe1 = model.hub.probe_store.types[i]
        cat1 = model.hub.probe_store.probe_cat_dict[probe1]
        for j in range(num_probes):
            if i != j:
                probe2 = model.hub.probe_store.types[j]
                cat2 = model.hub.probe_store.probe_cat_dict[probe2]
                sim = probe_simmat[i, j]
                cat_sim_dict[cat1][cat2].append(sim)
    # make category simmat
    cat_simmat = np.zeros([num_cats, num_cats], float)
    for i in range(num_cats):
        cat1 = model.hub.probe_store.cats[i]
        for j in range(num_cats):
            cat2 = model.hub.probe_store.cats[j]
            sims = np.array(cat_sim_dict[cat1][cat2])  # this contains a list of sims
            sim_mean = sims.mean()
            cat_simmat[model.hub.probe_store.cats.index(cat1), model.hub.probe_store.cats.index(cat2)] = sim_mean
    return cat_simmat


def human_format(num, pos):  # pos is required for formatting mpl axis ticklabels
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format(num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def plot_best_fit_line(ax, xys, fontsize, color='red', zorder=3, x_pos=0.05, y_pos=0.9, plot_p=True):
    x, y = zip(*xys)
    try:
        best_fit_fxn = np.polyfit(x, y, 1, full=True)
    except Exception as e:  # cannot fit line
        print('rnnlab: Cannot fit line.', e)
        return
    slope = best_fit_fxn[0][0]
    intercept = best_fit_fxn[0][1]
    xl = [min(x), max(x)]
    yl = [slope * xx + intercept for xx in xl]
    # plot line
    ax.plot(xl, yl, linewidth=FigsConfigs.LINEWIDTH, c=color, zorder=zorder)
    # plot rsqrd
    variance = np.var(y)
    residuals = np.var([(slope * xx + intercept - yy) for xx, yy in zip(x, y)])
    Rsqr = np.round(1 - residuals / variance, decimals=3)
    if Rsqr > 0.5:
        fontsize += 5
    ax.text(x_pos, y_pos, '$R^2$ = {}'.format(Rsqr), transform=ax.transAxes, fontsize=fontsize)
    if plot_p:
        p = np.round(linregress(x, y)[3], decimals=8)
        ax.text(x_pos, y_pos - 0.05, 'p = {}'.format(p), transform=ax.transAxes, fontsize=fontsize - 2)


def get_neighbors_from_token_simmat(model, token_simmat, term, num_neighbors):
    token_id = model.train_terms.term_id_dict[term]
    neighbor_tuples = [(token, sim) for token, sim in zip(model.train_terms.types, token_simmat[token_id])]
    neighbor_tuples_sorted = sorted(neighbor_tuples, key=itemgetter(1), reverse=True)[1:num_neighbors]
    neighbors = [tuple[0] for tuple in neighbor_tuples_sorted]
    return neighbors


def reload_acts_df(model_name, mb_name, hub_mode):
    start = time.time()
    path = GlobalConfigs.RUNS_DIR / model_name / 'Data_Frame' / 'acts_df_{}.h5'.format(mb_name)
    result = pd.read_hdf(path, key='acts_{}_df'.format(hub_mode))
    print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
    return result


def extract_n_elements(l, n):
    while not gcd(n, len(l)) == n:
        l.pop(0)
    len_div = len(l)
    step = len_div // n
    ids = [i - 1 for i in np.arange(step, len_div + step, step)]  # -1 for indexing
    elements = np.asarray(l)[ids].tolist()
    return elements


def make_cats_sorted_by_ba_diff(models1, models2):
    cat_list = models1[0].hub.probe_store.cats
    num_cats = len(cat_list)
    cats_sorted_by_ba = models1[0].sort_cats_by_ba('ordered')
    # load data
    cat_ba_means = []
    for models in [models1, models2]:
        cat_ba_mat = np.zeros((len(models), num_cats))
        for model_id, model in enumerate(models):
            sorted_cat_avg_probe_ba_lists = model.get_sorted_cat_avg_probe_ba_lists(cats_sorted_by_ba)
            cat_ba_mat[model_id, :] = [np.mean(cat_avg_probe_ba_list)
                                       for cat_avg_probe_ba_list in sorted_cat_avg_probe_ba_lists]
        cat_ba_means.append(np.mean(cat_ba_mat, axis=0))
    # make_most_different_cats
    cat_ba_differences = np.absolute(np.subtract(*cat_ba_means))
    most_different_cat_ids = np.argsort(cat_ba_differences)[::-1]
    result = [cats_sorted_by_ba[cat_id] for cat_id in most_different_cat_ids]
    return result


def to_trajs(model, fn, traj_args, *args):
    num_args = len(traj_args)
    result = [[] for _ in range(num_args)]
    for mb_name in model.ckpt_mb_names:
        model.graph = model.make_graph(mb_name)
        for n, arg in enumerate(traj_args):
            r = fn(arg, *args)
            result[n].append(r)
    return result


def to_traj(model, fn, *args):  # only works for metrics using graph, as other metrics might use cached dfs
    result = []
    for mb_name in model.ckpt_mb_names:
        model.graph = model.make_graph(mb_name)
        r = fn(*args)
        result.append(r)
    return result


def add_double_legend(ax, lines_list, labels, model_descs, y_offset=-0.2):  # requires figure height = 6
    box = ax.get_position()
    num_model_groups = len(model_descs)
    shrink_prop = 0.1 * num_model_groups  # TODO this doesn't always work well
    ax.set_position([box.x0, box.y0 + box.height * shrink_prop,  # Shrink vertically to make room for legend
                     box.width, box.height * (1 - shrink_prop)])
    leg1 = plt.legend([l[0] for l in lines_list], model_descs, loc='upper center',
                      bbox_to_anchor=(0.5, y_offset), ncol=2, frameon=False, fontsize=FigsConfigs.LEG_FONTSIZE)
    for lines in lines_list:
        for line in lines:
            line.set_color('black')
    plt.legend(lines_list[0], labels, loc='upper center',
               bbox_to_anchor=(0.5, y_offset + 0.1), ncol=3, frameon=False, fontsize=FigsConfigs.LEG_FONTSIZE)
    plt.gca().add_artist(leg1)  # order of legend creation matters here


def add_single_legend(ax, model_descs, y_offset=-0.25):
    box = ax.get_position()
    shrink_prop = 0.1 * len(model_descs)
    ax.set_position([box.x0, box.y0 + box.height * shrink_prop,  # Shrink vertically to make room for legend
                     box.width, box.height * (1 - shrink_prop)])
    plt.legend(loc='center', fontsize=FigsConfigs.LEG_FONTSIZE,
               bbox_to_anchor=(0.5, y_offset), ncol=2, frameon=False)