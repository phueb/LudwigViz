import numpy as np
import pandas as pd
from operator import itemgetter
import sys
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import paired_cosine_distances
from itertools import cycle, islice, zip_longest
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import linalg
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.manifold import TSNE

from figutils import human_format
from figutils import split_probes_into_quartiles_by_fs
from figutils import plot_best_fit_line
from figutils import calc_cat_sim_mat
from figutils import reload_acts_df
from figutils import extract_n_elements
from configs import GlobalConfigs, FigsConfigs, AppConfigs


rcParams['figure.max_open_warning'] = 0


def make_avg_traj_figs(model):
    def make_avg_traj_fig(traj_name):
        """
        Returns fig showing trajectory of "traj_name"
        """
        start = time.time()

        # load data
        x = model.get_data_step_axis()
        y = model.get_traj(traj_name)
        if 'pp' in traj_name:
            y = np.log(y)  # log is natural log
            traj_name = 'Nat Log of ' + traj_name
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel(traj_name, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        # plot
        ax.plot(x, y, '-', linewidth=FigsConfigs.LINEWIDTH, color='black')
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_avg_traj_fig(traj_name)
            # for traj_name in ['probes_fs_o', 'probes_fs_s', 'probes_fs_n', 'probes_pp']]  # todo remove fs?
            for traj_name in ['probes_ba_o', 'probes_pp']]
    return figs


def make_cat_task_stats_figs(model):
    def make_cat_task_acc_trajs_fig():
        """
        Returns fig showing category task accuracy trajectories for each fold
        """
        start = time.time()

        # load data
        xys = []
        for test_fold_id in range(GlobalConfigs.NUM_TEST_FOLDS):
            y_train = np.squeeze(np.add(*[model.get_trajs_mat([test_fold_id], traj)
                                          for traj in ['cat_task_train_yes_acc', 'cat_task_train_no_acc']]) / 2.0)
            y_test = np.squeeze(np.add(*[model.get_trajs_mat([test_fold_id], traj)
                                         for traj in ['cat_task_test_yes_acc', 'cat_task_test_no_acc']]) / 2.0)
            x = model.get_data_step_axis()
            xys.append((test_fold_id, x, y_train, y_test))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('Category Task Accuracy', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        ax.set_ylim([-10, 110])
        ax.axhline(y=50, linestyle='--', c='grey', linewidth=FigsConfigs.LINEWIDTH / 2.0)
        # plot
        for (test_fold_id, x, y_train, y_test) in xys:
            ax.plot(x, y_train, '-', linewidth=FigsConfigs.LINEWIDTH, label='fold {} train'.format(test_fold_id))
            ax.plot(x, y_test, '-', linewidth=FigsConfigs.LINEWIDTH, label='fold {} test'.format(test_fold_id))
        plt.legend(loc='best')
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_cat_task_acc_vs_probe_fs_fig():
        """
            Returns fig showing test accuracies for probes of a single category
            """
        start = time.time()

        labels = ['yes', 'no']
        # load data
        xys = []
        for label in labels:
            xs, ys = [], []
            for test_fold_id in range(GlobalConfigs.NUM_TEST_FOLDS):
                # x
                df_traj = model.cat_task_traj_df.filter(items=['{}_{}_fold{}'.format(
                    probe, label, test_fold_id) for probe in model.hub.probe_store.types])
                x = df_traj.values.sum(axis=0)
                xs.append(x)
                # y
                probes = [column_name.split('_')[0] for column_name in
                          df_traj.columns]  # not all probes need be included
                y = [model.avg_probe_fs_o_list[n] for n, probe in enumerate(model.hub.probe_store.types)
                     if probe in probes]
                ys.append(y)
            test_fold_ids = list(range(GlobalConfigs.NUM_TEST_FOLDS))
            xys.append((label, xs, ys, test_fold_ids))
        # fig
        fig, axarr = plt.subplots(2, GlobalConfigs.NUM_TEST_FOLDS,
                                  figsize=(FigsConfigs.MAX_FIG_WIDTH, 6), dpi=FigsConfigs.DPI)
        for axrow, (label, xs, ys, test_fold_ids) in zip(axarr, xys):
            for ax, x, y, test_fold_id in zip(axrow, xs, ys, test_fold_ids):
                ax.set_title('fold {}'.format(test_fold_id), loc='left')
                ax.set_xlabel('Probe Test Num Correct "{}"'.format(label), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
                ax.set_ylabel('Avg Probe Balanced Accuracy', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(axis='both', which='both', top='off', right='off')
                # plot
                ax.scatter(x, y, s=2, lw=0, color='black')
                # plot best fit line
                xys = zip(x, y)
                plot_best_fit_line(ax, xys, fontsize=12)
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_cat_cat_task_test_acc_trajs_fig(cat):
        """
        Returns fig showing test accuracies for probes of a single category
        """
        start = time.time()

        max_num_correct = len(model.get_data_step_axis())
        labels = ['"yes"', '"no"']
        # load data
        xys = []
        for test_fold_id in range(GlobalConfigs.NUM_TEST_FOLDS):
            # filter columns by partial column name
            df_traj_yes = model.cat_task_traj_df.filter(items=['{}_yes_fold{}'.format(
                probe, test_fold_id) for probe in model.hub.probe_store.cat_probe_list_dict[cat]])
            df_traj_no = model.cat_task_traj_df.filter(items=['{}_no_fold{}'.format(
                probe, test_fold_id) for probe in model.hub.probe_store.cat_probe_list_dict[cat]])
            ys = (df_traj_yes.values.sum(axis=0), df_traj_no.values.sum(axis=0))
            # xticklabels
            probes_yes = [column_name.split('_')[0] for column_name in df_traj_yes.columns]
            probes_no = [column_name.split('_')[0] for column_name in df_traj_no.columns]
            assert probes_yes == probes_no
            xticklabels = probes_yes
            num_xticklabels = len(xticklabels)
            xys.append((test_fold_id, ys, labels, xticklabels, num_xticklabels))
        # fig
        fig, axarr = plt.subplots(1, GlobalConfigs.NUM_TEST_FOLDS,
                                  figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        fig.suptitle(cat)
        for ax, (test_fold_id, ys, labels, xticklabels, num_xticklabels) in zip(axarr, xys):
            ax.set_title('fold {}'.format(test_fold_id), loc='left')
            ax.set_ylabel('Test Num Correct', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xticks(np.arange(num_xticklabels))
            ax.set_xticklabels(xticklabels, fontsize=FigsConfigs.TICKLABEL_FONT_SIZE, rotation=90)
            ax.tick_params(axis='both', which='both', top='off', right='off')
            ax.set_ylim([-1, max_num_correct + 1])
            ax.axhline(y=0, linestyle='--', c='grey', linewidth=FigsConfigs.LINEWIDTH / 2.0)
            ax.axhline(y=max_num_correct, linestyle='--', c='grey', linewidth=FigsConfigs.LINEWIDTH / 2.0)
            # plot
            for label, y in zip(labels, ys):
                ax.plot(y, 'o', linewidth=FigsConfigs.LINEWIDTH, label=label)
            x_overlap = np.where(ys[0] == ys[1])[0]
            y_overlap = ys[0][x_overlap]
            ax.plot(x_overlap, y_overlap, 'o', linewidth=FigsConfigs.LINEWIDTH,
                    color='black')  # TODO plot black when overlapping
            mean = np.mean(np.add(*ys) / 2.0)
            ax.axhline(y=mean, linestyle='-', linewidth=FigsConfigs.LINEWIDTH, label='mean', color='black')
        plt.legend(bbox_to_anchor=(1.0, 1.2), borderaxespad=0.0, fontsize=FigsConfigs.LEG_FONTSIZE)
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_cat_task_acc_trajs_fig(),
            make_cat_task_acc_vs_probe_fs_fig()] + \
           [make_cat_cat_task_test_acc_trajs_fig(cat)
            for cat in model.hub.probe_store.cats]
    return figs


def make_syn_task_stats_figs(model):
    def make_syn_task_acc_trajs_fig():  # TODO test
        """
        Returns fig showing synonym task accuracy trajectories for each fold
        """
        start = time.time()

        # load data
        xys = []
        for test_fold_id in range(GlobalConfigs.NUM_TEST_FOLDS):
            y_train = np.squeeze(np.add(*[model.get_trajs_mat([test_fold_id], traj)
                                          for traj in ['syn_task_train_yes_acc', 'syn_task_train_no_acc']]) / 2.0)
            y_test = np.squeeze(np.add(*[model.get_trajs_mat([test_fold_id], traj)
                                         for traj in ['syn_task_test_yes_acc', 'syn_task_test_no_acc']]) / 2.0)

            print(y_train.shape, y_test.shape)

            x = model.get_data_step_axis()
            xys.append((test_fold_id, x, y_train, y_test))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('Synonym Task Accuracy', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        ax.set_ylim([-10, 110])
        ax.axhline(y=50, linestyle='--', c='grey', linewidth=FigsConfigs.LINEWIDTH / 2.0)
        # plot
        for (test_fold_id, x, y_train, y_test) in xys:
            ax.plot(x, y_train, '-', linewidth=FigsConfigs.LINEWIDTH, label='fold {} train'.format(test_fold_id))
            ax.plot(x, y_test, '-', linewidth=FigsConfigs.LINEWIDTH, label='fold {} test'.format(test_fold_id))
        plt.legend(loc='best')
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_syn_task_acc_trajs_fig()]
    return figs


def make_dim_red_figs(model):
    def make_probes_acts_2d_fig(sv_nums=(1, 2),
                                perplexity=30,
                                label_probe=False,
                                is_subtitled=True):
        """
        Returns fig showing probe activations in 2D space using SVD & TSNE.
        """
        start = time.time()

        palette = np.array(sns.color_palette("hls", model.hub.probe_store.num_cats))
        # load data
        probes_acts_df = model.get_multi_probe_prototype_acts_df()
        probes_acts_cats = [model.hub.probe_store.probe_cat_dict[probe] for probe in model.hub.probe_store.types]
        u, s, v = linalg.svd(probes_acts_df.values, full_matrices=False)
        pcs = np.dot(u, np.diag(s))
        explained_variance = np.var(pcs, axis=0)
        full_variance = np.var(probes_acts_df.values, axis=0)
        expl_var_perc = explained_variance / full_variance.sum() * 100
        act_2d_svd = u[:, sv_nums]  # this is correct, and gives same results as pca
        acts_2d_tsne = TSNE(perplexity=perplexity).fit_transform(probes_acts_df.values)
        # fig
        fig, axarr = plt.subplots(2, 1, figsize=(FigsConfigs.MAX_FIG_WIDTH, 14), dpi=FigsConfigs.DPI)
        for n, probes_acts_2d in enumerate([act_2d_svd, acts_2d_tsne]):
            # plot
            palette_ids = [model.hub.probe_store.cats.index(probes_acts_cat) for probes_acts_cat in probes_acts_cats]
            axarr[n].scatter(probes_acts_2d[:, 0], probes_acts_2d[:, 1], lw=0, s=8, c=palette[palette_ids])
            axarr[n].axis('off')
            axarr[n].axis('tight')
            descr_str = ', '.join(['sv {}: var {:2.0f}%'.format(i, expl_var_perc[i]) for i in sv_nums])
            if is_subtitled:
                axarr[n].set_title(['SVD ({})'.format(descr_str), 't-SNE'][n], fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            # add the labels for each cat
            for cat in model.hub.probe_store.cats:
                probes_acts_2d_ids = np.where(np.asarray(probes_acts_cats) == cat)[0]
                xtext, ytext = np.median(probes_acts_2d[probes_acts_2d_ids, :], axis=0)
                txt = axarr[n].text(xtext, ytext, str(cat), fontsize=FigsConfigs.TICKLABEL_FONT_SIZE,
                                    color=palette[model.hub.probe_store.cats.index(cat)])
                txt.set_path_effects([
                    PathEffects.Stroke(linewidth=FigsConfigs.LINEWIDTH, foreground="w"), PathEffects.Normal()])
            # add the labels for each probe
            if label_probe:
                for probe in model.hub.probe_store.types:
                    probes_acts_2d_ids = np.where(np.asarray(model.hub.probe_store.types) == probe)[0]
                    xtext, ytext = np.median(probes_acts_2d[probes_acts_2d_ids, :], axis=0)
                    txt = axarr[n].text(xtext, ytext, str(probe), fontsize=FigsConfigs.TICKLABEL_FONT_SIZE)
                    txt.set_path_effects([
                        PathEffects.Stroke(linewidth=FigsConfigs.LINEWIDTH, foreground="w"), PathEffects.Normal()])
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_cat_acts_2d_walk_fig(last_pc):
        """
        Returns fig showing evolution of average category activations in 2D space using SVD.
        """
        start = time.time()

        palette = np.array(sns.color_palette("hls", model.hub.probe_store.num_cats))
        num_saved_mb_names = len(model.ckpt_mb_names)
        num_walk_timepoints = min(num_saved_mb_names, FigsConfigs.DEFAULT_NUM_WALK_TIMEPOINTS)
        walk_mb_names = extract_n_elements(model.ckpt_mb_names, num_walk_timepoints)
        # fit pca model on last data_step
        pca_model = sklearnPCA(n_components=last_pc)
        model.acts_df = reload_acts_df(model.model_name, model.ckpt_mb_names[-1], model.hub.mode)
        cat_prototype_acts_df = model.get_multi_cat_prototype_acts_df()
        pca_model.fit(cat_prototype_acts_df.values)
        # transform acts from remaining data_steps with pca model
        cat_acts_2d_mats = []
        for mb_name in walk_mb_names:
            model.acts_df = reload_acts_df(model.model_name, mb_name, model.hub.mode)
            cat_prototype_acts_df = model.get_multi_cat_prototype_acts_df()
            cat_act_2d_pca = pca_model.transform(cat_prototype_acts_df.values)
            cat_acts_2d_mats.append(cat_act_2d_pca[:, last_pc - 2:])
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 7), dpi=FigsConfigs.DPI)
        for cat_id, cat in enumerate(model.hub.probe_store.cats):
            x, y = zip(*[acts_2d_mat[cat_id] for acts_2d_mat in cat_acts_2d_mats])
            ax.plot(x, y, c=palette[cat_id], lw=FigsConfigs.LINEWIDTH)
            xtext, ytext = cat_acts_2d_mats[-1][cat_id, :]
            txt = ax.text(xtext, ytext, str(cat), fontsize=8,
                          color=palette[cat_id])
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=FigsConfigs.LINEWIDTH, foreground="w"), PathEffects.Normal()])
        ax.axis('off')
        x_maxval = np.max(np.dstack(cat_acts_2d_mats)[:, 0, :]) * 1.2
        y_maxval = np.max(np.dstack(cat_acts_2d_mats)[:, 1, :]) * 1.2
        ax.set_xlim([-x_maxval, x_maxval])
        ax.set_ylim([-y_maxval, y_maxval])
        ax.axhline(y=0, linestyle='--', c='grey', linewidth=1.0)
        ax.axvline(x=0, linestyle='--', c='grey', linewidth=1.0)
        ax.set_title('PCA (Components {} and {}) Walk across Timepoints'.format(last_pc - 1, last_pc))
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_probes_acts_2d_fig()] + \
           [make_cat_acts_2d_walk_fig(last_pc)
            for last_pc in FigsConfigs.LAST_PCS]
    return figs


def make_principal_comps_figs(model):
    def make_principal_comps_termgroup_heatmap_fig(tokengroup_tokens_dict,
                                                   label_expl_var=False):
        """
        Returns fig showing heatmap of loadings of probes on principal components by "tokengroup"
        """
        start = time.time()

        tokengroup_list = sorted(tokengroup_tokens_dict.keys())
        num_tokengroups = len(tokengroup_list)
        pc_labels = model.hub.train_terms.types
        # load data
        acts_mat = model.term_acts_mat
        pca_model = sklearnPCA(n_components=FigsConfigs.NUM_PCS)
        pcs = pca_model.fit_transform(acts_mat)
        expl_vars = np.asarray(pca_model.explained_variance_ratio_) * 100
        # make pca_mat
        pca_mat = np.zeros((num_tokengroups, FigsConfigs.NUM_PCS))
        for pc_id, pc in enumerate(pcs.transpose()):
            for tokengroup_id, tokengroup in enumerate(tokengroup_list):
                tokens = tokengroup_tokens_dict[tokengroup]
                loadings = [loading for loading, token in zip(pc, pc_labels) if token in tokens]
                pca_mat[tokengroup_id, pc_id] = sum(loadings) / len(tokens)
        # fig
        fig, ax_heatmap = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 0.2 * num_tokengroups),
                                       dpi=FigsConfigs.DPI)
        divider = make_axes_locatable(ax_heatmap)
        ax_colorbar = divider.append_axes("right", 0.2, pad=0.1)
        # cluster rows
        lnk0 = linkage(pdist(pca_mat))
        dg0 = dendrogram(lnk0,
                         no_plot=True)
        z = pca_mat[dg0['leaves'], :]
        # heatmap
        max_extent = ax_heatmap.get_ylim()[1]
        vmin, vmax = round(np.min(z), 1), round(np.max(z), 1)
        im = ax_heatmap.imshow(z[::-1], aspect='auto',
                               cmap=plt.cm.jet,
                               extent=(0, max_extent, 0, max_extent),
                               vmin=vmin, vmax=vmax)
        # colorbar
        cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='vertical')
        cb.ax.set_xticklabels([vmin, vmax], fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        cb.set_label('Average Loading', labelpad=-10, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # set heatmap ticklabels
        xlim = ax_heatmap.get_xlim()[1]
        halfxw = 0.5 * xlim / FigsConfigs.NUM_PCS
        ax_heatmap.xaxis.set_ticks(np.linspace(halfxw, xlim - halfxw, FigsConfigs.NUM_PCS))
        if label_expl_var:
            ax_heatmap.xaxis.set_ticklabels(['PC {} ({:.1f} %)'.format(pc_id + 1, expl_var)
                                             for pc_id, expl_var in zip(range(FigsConfigs.NUM_PCS), expl_vars)])
        else:
            ax_heatmap.xaxis.set_ticklabels(['PC {}'.format(pc_id + 1)
                                             for pc_id in range(FigsConfigs.NUM_PCS)])
        ylim = ax_heatmap.get_ylim()[1]
        halfyw = 0.5 * ylim / num_tokengroups
        ax_heatmap.yaxis.set_ticks(np.linspace(halfyw, ylim - halfyw, num_tokengroups))
        ax_heatmap.yaxis.set_ticklabels(np.array(tokengroup_list)[dg0['leaves']])
        # Hide all tick lines
        lines = (ax_heatmap.xaxis.get_ticklines() +
                 ax_heatmap.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        # set label rotation and fontsize
        xlbls = ax_heatmap.xaxis.get_ticklabels()
        plt.setp(xlbls, rotation=-90)
        plt.setp(xlbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ylbls = ax_heatmap.yaxis.get_ticklabels()
        plt.setp(ylbls, rotation=0)
        plt.setp(ylbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        fig.subplots_adjust(bottom=0.2)  # make room for tick labels
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_principal_comps_item_heatmap_fig(pca_item_list, label_expl_var=True):
        """
        Returns fig showing heatmap of loadings of probes on principal components by custom list of words
        """
        start = time.time()

        # load data
        num_items = len(pca_item_list)
        acts_mat = model.term_acts_mat
        pca_model = sklearnPCA(n_components=FigsConfigs.NUM_PCS)
        pcs = pca_model.fit_transform(acts_mat)
        expl_vars = np.asarray(pca_model.explained_variance_ratio_) * 100
        # make item_pca_mat
        item_pca_mat = np.zeros((len(pca_item_list), FigsConfigs.NUM_PCS))
        for pc_id, pc in enumerate(pcs.transpose()):
            for item_id, item in enumerate(pca_item_list):
                try:
                    item_loading = [loading for loading, token in zip(pc, model.hub.train_terms.types)
                                    if token == item][0]
                except IndexError:  # if item not in vocab
                    item_loading = 0
                    print('Not in vocabulary: "{}"'.format(item))
                item_pca_mat[item_id, pc_id] = item_loading
        # fig
        width = min(FigsConfigs.MAX_FIG_WIDTH, FigsConfigs.NUM_PCS * 1.5 - 1)
        fig, ax_heatmap = plt.subplots(figsize=(width, 0.2 * num_items + 2), dpi=FigsConfigs.DPI)
        divider = make_axes_locatable(ax_heatmap)
        ax_colorbar = divider.append_axes("top", 0.2, pad=0.2)
        # cluster rows
        if FigsConfigs.CLUSTER_PCA_ITEM_ROWS:
            lnk0 = linkage(pdist(item_pca_mat))
            dg0 = dendrogram(lnk0,
                             no_plot=True)
            z = item_pca_mat[dg0['leaves'], :]
            yticklabels = np.array(pca_item_list)[dg0['leaves']]
        else:
            z = item_pca_mat
            yticklabels = pca_item_list
        # heatmap
        max_extent = ax_heatmap.get_ylim()[1]
        vmin, vmax = round(np.min(z), 1), round(np.max(z), 1)
        im = ax_heatmap.imshow(z[::-1], aspect='auto',
                               cmap=plt.cm.jet,
                               extent=(0, max_extent, 0, max_extent),
                               vmin=vmin, vmax=vmax)
        # colorbar
        cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='horizontal')
        cb.ax.set_xticklabels([vmin, vmax], rotation=0, fontsize=FigsConfigs.LEG_FONTSIZE)
        cb.set_label('Loading', labelpad=-40, rotation=0, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # set heatmap ticklabels
        xlim = ax_heatmap.get_xlim()[1]
        halfxw = 0.5 * xlim / FigsConfigs.NUM_PCS
        ax_heatmap.xaxis.set_ticks(np.linspace(halfxw, xlim - halfxw, FigsConfigs.NUM_PCS))
        if label_expl_var:
            ax_heatmap.xaxis.set_ticklabels(['PC {} ({:.1f} %)'.format(pc_id + 1, expl_var)
                                             for pc_id, expl_var in zip(range(FigsConfigs.NUM_PCS), expl_vars)])
        else:
            ax_heatmap.xaxis.set_ticklabels(['PC {}'.format(pc_id + 1)
                                             for pc_id in range(FigsConfigs.NUM_PCS)])
        ylim = ax_heatmap.get_ylim()[1]
        halfyw = 0.5 * ylim / num_items
        ax_heatmap.yaxis.set_ticks(np.linspace(halfyw, ylim - halfyw, num_items))
        ax_heatmap.yaxis.set_ticklabels(yticklabels)
        # Hide all tick lines
        lines = (ax_heatmap.xaxis.get_ticklines() +
                 ax_heatmap.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        # set label rotation and fontsize
        xlbls = ax_heatmap.xaxis.get_ticklabels()
        plt.setp(xlbls, rotation=-90)
        plt.setp(xlbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ylbls = ax_heatmap.yaxis.get_ticklabels()
        plt.setp(ylbls, rotation=0)
        plt.setp(ylbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        fig.subplots_adjust(bottom=0.2)  # make room for tick labels
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_principal_comps_table_fig(pc_id):
        """
        Returns fig showing tokens along principal component axis
        """
        start = time.time()

        num_cols = 2
        col_labels = ['Low End', 'High End']
        # load data
        acts_mat = model.term_acts_mat if FigsConfigs.SVD_IS_TERMS else model.get_multi_probe_prototype_acts_df().values
        tokens = model.hub.train_terms.types if FigsConfigs.SVD_IS_TERMS else model.hub.probe_store.types
        pca_model = sklearnPCA(n_components=FigsConfigs.NUM_PCS)
        pcs = pca_model.fit_transform(acts_mat)
        expl_var_perc = np.asarray(pca_model.explained_variance_ratio_) * 100
        pc = pcs[:, pc_id]
        # sort and filter
        sorted_pc, sorted_tokens = list(zip(*sorted(zip(pc, tokens), key=itemgetter(0))))
        col0_strs = ['{} {:.2f} (freq: {:,})'.format(term, loading, sum(model.hub.term_part_freq_dict[term]))
                     for term, loading in zip(
                sorted_tokens[:FigsConfigs.NUM_PCA_LOADINGS], sorted_pc[:FigsConfigs.NUM_PCA_LOADINGS])
                     if sum(model.hub.term_part_freq_dict[term]) > FigsConfigs.PCA_FREQ_THR]
        col1_strs = ['{} {:.2f} ({:,})'.format(term, loading, sum(model.hub.term_part_freq_dict[term]))
                     for term, loading in zip(
                sorted_tokens[-FigsConfigs.NUM_PCA_LOADINGS:][::-1], sorted_pc[-FigsConfigs.NUM_PCA_LOADINGS:][::-1])
                     if sum(model.hub.term_part_freq_dict[term]) > FigsConfigs.PCA_FREQ_THR]
        # make probes_mat
        max_rows = max(len(col0_strs), len(col1_strs))
        probes_mat = np.chararray((max_rows, num_cols), itemsize=40, unicode=True)
        probes_mat[:] = ''  # initialize so that mpl can read table
        probes_mat[:len(col0_strs), 0] = col0_strs
        probes_mat[:len(col1_strs), 1] = col1_strs
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 0.25 * max_rows), dpi=FigsConfigs.DPI)
        ax.set_title('Principal Component {} ({:.2f}% var)'.format(
            pc_id, expl_var_perc[pc_id]), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.axis('off')
        # plot
        table_ = ax.table(cellText=probes_mat, colLabels=col_labels,
                          loc='center', colWidths=[0.3] * num_cols)
        table_.auto_set_font_size(False)
        table_.set_fontsize(8)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_principal_comps_line_fig(pc_id):
        """
        Returns fig showing loadings on a specified principal component
        """
        start = time.time()

        # load data
        acts_mat = model.term_acts_mat if FigsConfigs.SVD_IS_TERMS else model.get_multi_probe_prototype_acts_df().values
        pca_model = sklearnPCA(n_components=FigsConfigs.NUM_PCS)
        pcs = pca_model.fit_transform(acts_mat)
        pc = pcs[:, pc_id]
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        # plot
        ax.plot(sorted(pc), '-', linewidth=FigsConfigs.LINEWIDTH, color='black')
        ax.set_xticklabels([])
        ax.set_xlabel('Token IDs', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('PC {} Loading'.format(pc_id), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_scree_fig():
        """
        Returns fig showing amount of variance accounted for by each principal component
        """
        start = time.time()

        # load data
        acts_mat = model.term_acts_mat if FigsConfigs.SVD_IS_TERMS else model.get_multi_probe_prototype_acts_df().values
        pca_model = sklearnPCA(n_components=acts_mat.shape[1])
        pca_model.fit(acts_mat)
        expl_var_perc = np.asarray(pca_model.explained_variance_ratio_) * 100
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xticklabels([])
        ax.set_xlabel('Principal Component', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('% Var Explained (Cumulative)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylim([0, 100])
        # plot
        ax.plot(expl_var_perc.cumsum(), '-', linewidth=FigsConfigs.LINEWIDTH, color='black')
        ax.plot(expl_var_perc.cumsum()[:FigsConfigs.NUM_PCS], 'o', linewidth=FigsConfigs.LINEWIDTH, color='black')
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    pos_terms_dict = FigsConfigs.POS_TERMS_DICT
    pos_terms_dict['Nouns'] = model.hub.nouns
    pos_terms_dict['Verbs'] = model.hub.verbs
    pos_terms_dict['Adjectives'] = model.hub.adjectives
    terms_with_abstract_relation = model.hub.get_terms_related_to_cat('SPACE')
    terms_after_terms = model.hub.get_term_set_prop_near_terms(['story'], 1)
    figs = [make_principal_comps_termgroup_heatmap_fig(pos_terms_dict),
            make_principal_comps_termgroup_heatmap_fig(model.hub.probe_store.cat_probe_list_dict),
            make_principal_comps_item_heatmap_fig(terms_with_abstract_relation[:30] +
                                                  terms_with_abstract_relation[-30:]),
            make_principal_comps_item_heatmap_fig(terms_after_terms[:30] +
                                                  terms_after_terms[-30:]),
            make_principal_comps_item_heatmap_fig(
                ['did', 'made', 'went', 'saw', 'was', 'had', 'got', 'happened', 'ate', 'said', 'tried',
                 'do', 'make', 'go', 'see', 'is', 'has', 'get', 'happens', 'eat', 'say', 'try']),
            make_principal_comps_item_heatmap_fig(FigsConfigs.PCA_ITEM_LIST1),
            make_principal_comps_item_heatmap_fig(FigsConfigs.PCA_ITEM_LIST2),
            make_scree_fig()] + \
           [make_principal_comps_table_fig(pc_id) for pc_id in range(FigsConfigs.NUM_PCS)] + \
           [make_principal_comps_line_fig(pc_id) for pc_id in range(FigsConfigs.NUM_PCS)]
    return figs


def make_probe_sims_figs(model, field_input):
    def make_probe_sim_dh_fig_jw(probes,
                                 num_colors=None,
                                 label_doc_id=True,
                                 vmin=0.90, vmax=1.0,
                                 cluster=False):
        """
        Returns fig showing dendrogram heatmap of similarity matrix of "probes"
        """
        start = time.time()

        num_probes = len(probes)
        assert num_probes > 1
        # load data
        probes_acts_df = model.get_multi_probe_prototype_acts_df()
        probe_ids = [model.hub.probe_store.probe_id_dict[probe] for probe in probes]
        probes_acts_df_filtered = probes_acts_df.iloc[probe_ids]
        probe_simmat = cosine_similarity(probes_acts_df_filtered.values)
        print('Probe simmat  min: {} max {}'.format(np.min(probe_simmat), np.max(probe_simmat)))
        print('Fig  min: {} max {}'.format(vmin, vmax))
        # fig
        fig, ax_heatmap = plt.subplots(figsize=(11, 7), dpi=FigsConfigs.DPI)
        if label_doc_id:
            plt.title('Trained on {:,} terms'.format(model.mb_size * int(model.mb_name)),
                      fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax_heatmap.yaxis.tick_right()
        divider = make_axes_locatable(ax_heatmap)
        ax_dend = divider.append_axes("bottom", 0.8, pad=0.0)  # , sharex=ax_heatmap)
        ax_colorbar = divider.append_axes("left", 0.1, pad=0.4)
        ax_freqs = divider.append_axes("right", 4.0, pad=0.0)  # , sharey=ax_heatmap)
        # dendrogram
        ax_dend.set_frame_on(False)
        lnk0 = linkage(pdist(probe_simmat))
        if num_colors is None or num_colors <= 1:
            left_threshold = -1
        else:
            left_threshold = 0.5 * (lnk0[1 - num_colors, 2] +
                                    lnk0[-num_colors, 2])
        dg0 = dendrogram(lnk0, ax=ax_dend,
                         orientation='bottom',
                         color_threshold=left_threshold,
                         no_labels=True,
                         no_plot=not cluster)
        if cluster:
            # Reorder the values in x to match the order of the leaves of the dendrograms
            z = probe_simmat[dg0['leaves'], :]  # sorting rows
            z = z[:, dg0['leaves']]  # sorting columns for symmetry
            simmat_labels = np.array(probes)[dg0['leaves']]
        else:
            z = probe_simmat
            simmat_labels = probes
        # probe freq bar plot
        doc_id = model.doc_id if label_doc_id else model.num_docs  # faster if not label_doc_id
        probe_freqs = [sum(model.hub.term_part_freq_dict[probe][:doc_id]) * model.num_iterations
                       for probe in simmat_labels]
        y = range(num_probes)
        ax_freqs.barh(y, probe_freqs, color='black')
        ax_freqs.set_xlabel('Freq')
        ax_freqs.set_xlim([0, FigsConfigs.PROBE_FREQ_YLIM])
        ax_freqs.set_xticks([0, FigsConfigs.PROBE_FREQ_YLIM])
        ax_freqs.set_xticklabels([0, FigsConfigs.PROBE_FREQ_YLIM])
        ax_freq_lim0 = ax_freqs.get_ylim()[0] + 0.5
        ax_freq_lim1 = ax_freqs.get_ylim()[1] - 0.5
        ax_freqs.set_ylim([ax_freq_lim0, ax_freq_lim1])  # shift ticks to match heatmap
        ax_freqs.yaxis.set_ticks(y)
        ax_freqs.yaxis.set_ticklabels(simmat_labels, color='white')
        # heatmap
        max_extent = ax_dend.get_xlim()[1]
        im = ax_heatmap.imshow(z[::-1], aspect='auto',
                               cmap=plt.cm.jet,
                               extent=(0, max_extent, 0, max_extent),
                               vmin=vmin, vmax=vmax)
        # colorbar
        cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='vertical')
        cb.ax.set_xticklabels([vmin, vmax], fontsize=FigsConfigs.AXLABEL_FONT_SIZE, rotation=90)
        cb.set_label('Correlation Coefficient', labelpad=-10, fontsize=FigsConfigs.AXLABEL_FONT_SIZE, rotation=90)
        # set heatmap ticklabels
        ax_heatmap.xaxis.set_ticks([])
        ax_heatmap.xaxis.set_ticklabels([])
        ax_heatmap.yaxis.set_ticks([])
        ax_heatmap.yaxis.set_ticklabels([])
        # Hide all tick lines
        lines = (ax_heatmap.xaxis.get_ticklines() +
                 ax_heatmap.yaxis.get_ticklines() +
                 ax_dend.xaxis.get_ticklines() +
                 ax_dend.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        # set label rotation and fontsize
        xlbls = ax_heatmap.xaxis.get_ticklabels()
        plt.setp(xlbls, rotation=-90)
        plt.setp(xlbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ylbls = ax_heatmap.yaxis.get_ticklabels()
        plt.setp(ylbls, rotation=0)
        plt.setp(ylbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # make dendrogram labels invisible
        plt.setp(ax_dend.get_yticklabels() + ax_dend.get_xticklabels(),
                 visible=False)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_probe_sim_dh_fig(probes,
                              num_colors=None,
                              label_doc_id=True,
                              vmin=0.90, vmax=1.0,
                              cluster=True):
        """
        Returns fig showing dendrogram heatmap of similarity matrix of "probes"
        """
        start = time.time()

        num_probes = len(probes)
        # load data
        probes_acts_df = model.get_multi_probe_prototype_acts_df()
        probe_ids = [model.hub.probe_store.probe_id_dict[probe] for probe in probes]
        probes_acts_df_filtered = probes_acts_df.iloc[probe_ids]
        probe_simmat = cosine_similarity(probes_acts_df_filtered.values)
        print('Probe simmat  min: {} max {}'.format(np.min(probe_simmat), np.max(probe_simmat)))
        print('Fig  min: {} max {}'.format(vmin, vmax))
        # fig
        fig, ax_heatmap = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 7), dpi=FigsConfigs.DPI)
        if label_doc_id:
            plt.title('Trained on {:,} terms'.format(model.mb_size * int(model.mb_name)),
                      fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax_heatmap.yaxis.tick_right()
        divider = make_axes_locatable(ax_heatmap)
        ax_denright = divider.append_axes("right", 0.8, pad=0.0, sharey=ax_heatmap)
        ax_denright.set_frame_on(False)
        ax_colorbar = divider.append_axes("left", 0.1, pad=0.4)
        ax_freqs = divider.append_axes("bottom", 0.5, pad=0.0)
        # dendrogram
        lnk0 = linkage(pdist(probe_simmat))
        if num_colors is None or num_colors <= 1:
            left_threshold = -1
        else:
            left_threshold = 0.5 * (lnk0[1 - num_colors, 2] +
                                    lnk0[-num_colors, 2])
        dg0 = dendrogram(lnk0, ax=ax_denright,
                         orientation='right',
                         color_threshold=left_threshold,
                         no_labels=True,
                         no_plot=not cluster)
        if cluster:
            # Reorder the values in x to match the order of the leaves of the dendrograms
            z = probe_simmat[dg0['leaves'], :]  # sorting rows
            z = z[:, dg0['leaves']]  # sorting columns for symmetry
            simmat_labels = np.array(probes)[dg0['leaves']]
        else:
            z = probe_simmat
            simmat_labels = probes
        # probe freq bar plot
        doc_id = model.doc_id if label_doc_id else model.num_docs  # faster if not label_doc_id
        probe_freqs = [sum(model.hub.term_part_freq_dict[probe][:doc_id]) for probe in simmat_labels]
        x = range(num_probes)
        ax_freqs.bar(x, probe_freqs, color='black')
        ax_freqs.yaxis.tick_right()
        ax_freqs.set_ylabel('Freq')
        ax_freqs.set_ylim([0, FigsConfigs.PROBE_FREQ_YLIM])
        ax_freqs.set_yticks([0, FigsConfigs.PROBE_FREQ_YLIM])
        ax_freqs.set_yticklabels([0, FigsConfigs.PROBE_FREQ_YLIM])
        ax_freq_lim0 = ax_freqs.get_xlim()[0] + 0.5
        ax_freq_lim1 = ax_freqs.get_xlim()[1] - 0.5
        ax_freqs.set_xlim([ax_freq_lim0, ax_freq_lim1])  # shift ticks to match heatmap
        ax_freqs.xaxis.set_ticks(x)
        ax_freqs.xaxis.set_ticklabels(simmat_labels, rotation=90)
        # heatmap
        max_extent = ax_denright.get_ylim()[1]
        im = ax_heatmap.imshow(z[::-1], aspect='auto',
                               cmap=plt.cm.jet,
                               extent=(0, max_extent, 0, max_extent),
                               vmin=vmin, vmax=vmax)
        # colorbar
        cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='vertical')
        cb.ax.set_xticklabels([vmin, vmax], fontsize=FigsConfigs.AXLABEL_FONT_SIZE, rotation=90)
        cb.set_label('Correlation Coefficient', labelpad=-10, fontsize=FigsConfigs.AXLABEL_FONT_SIZE, rotation=90)
        # set heatmap ticklabels
        ax_heatmap.xaxis.set_ticks([])
        ax_heatmap.xaxis.set_ticklabels([])
        ax_heatmap.yaxis.set_ticks([])
        ax_heatmap.yaxis.set_ticklabels([])
        # Hide all tick lines
        lines = (ax_heatmap.xaxis.get_ticklines() +
                 ax_heatmap.yaxis.get_ticklines() +
                 ax_denright.xaxis.get_ticklines() +
                 ax_denright.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        # set label rotation and fontsize
        xlbls = ax_heatmap.xaxis.get_ticklabels()
        plt.setp(xlbls, rotation=-90)
        plt.setp(xlbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ylbls = ax_heatmap.yaxis.get_ticklabels()
        plt.setp(ylbls, rotation=0)
        plt.setp(ylbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # make dendrogram labels invisible
        plt.setp(ax_denright.get_yticklabels() + ax_denright.get_xticklabels(),
                 visible=False)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_probe_sim_dh_fig_jw(field_input),
            make_probe_sim_dh_fig(field_input)]
    return figs


def make_cat_sim_figs(model):
    def make_cat_sim_dh_fig(num_colors=None, y_title=False, vmin=0.0, vmax=1.0):
        """
        Returns fig showing dendrogram heatmap of category similarity matrix
        """
        start = time.time()

        # load data
        cat_simmat = calc_cat_sim_mat(model)
        cat_simmat_labels = model.hub.probe_store.cats
        print('Cat simmat  min: {} max {}'.format(np.min(cat_simmat), np.max(cat_simmat)))
        print('Fig  min: {} max {}'.format(vmin, vmax))
        # fig
        fig, ax_heatmap = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 7), dpi=FigsConfigs.DPI)
        ax_heatmap.yaxis.tick_right()
        divider = make_axes_locatable(ax_heatmap)
        ax_denright = divider.append_axes("right", 0.8, pad=0.0, sharey=ax_heatmap)
        ax_denright.set_frame_on(False)
        ax_colorbar = divider.append_axes("top", 0.1, pad=0.4)
        # dendrogram
        lnk0 = linkage(pdist(cat_simmat))
        if num_colors is None or num_colors <= 1:
            left_threshold = -1
        else:
            left_threshold = 0.5 * (lnk0[1 - num_colors, 2] +
                                    lnk0[-num_colors, 2])
        dg0 = dendrogram(lnk0, ax=ax_denright,
                         orientation='right',
                         color_threshold=left_threshold,
                         no_labels=True)
        # Reorder the values in x to match the order of the leaves of the dendrograms
        z = cat_simmat[dg0['leaves'], :]  # sorting rows
        z = z[:, dg0['leaves']]  # sorting columns for symmetry
        # heatmap
        max_extent = ax_denright.get_ylim()[1]
        im = ax_heatmap.imshow(z[::-1], aspect='auto',
                               cmap=plt.cm.jet,
                               extent=(0, max_extent, 0, max_extent),
                               vmin=vmin, vmax=vmax)
        # colorbar
        cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='horizontal')
        cb.ax.set_xticklabels([vmin, vmax], fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        cb.set_label('Correlation Coefficient', labelpad=-10, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # set heatmap ticklabels
        xlim = ax_heatmap.get_xlim()[1]
        ncols = len(cat_simmat_labels)
        halfxw = 0.5 * xlim / ncols
        ax_heatmap.xaxis.set_ticks(np.linspace(halfxw, xlim - halfxw, ncols))
        ax_heatmap.xaxis.set_ticklabels(np.array(cat_simmat_labels)[dg0['leaves']])  # for symmetry
        ylim = ax_heatmap.get_ylim()[1]
        nrows = len(cat_simmat_labels)
        halfyw = 0.5 * ylim / nrows
        if y_title:
            ax_heatmap.yaxis.set_ticks(np.linspace(halfyw, ylim - halfyw, nrows))
            ax_heatmap.yaxis.set_ticklabels(np.array(cat_simmat_labels)[dg0['leaves']])
        # Hide all tick lines
        lines = (ax_heatmap.xaxis.get_ticklines() +
                 ax_heatmap.yaxis.get_ticklines() +
                 ax_denright.xaxis.get_ticklines() +
                 ax_denright.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        # set label rotation and fontsize
        xlbls = ax_heatmap.xaxis.get_ticklabels()
        plt.setp(xlbls, rotation=-90)
        plt.setp(xlbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ylbls = ax_heatmap.yaxis.get_ticklabels()
        plt.setp(ylbls, rotation=0)
        plt.setp(ylbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # make dendrogram labels invisible
        plt.setp(ax_denright.get_yticklabels() + ax_denright.get_xticklabels(),
                 visible=False)
        fig.subplots_adjust(bottom=0.2)  # make room for tick labels
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_probe_sim_hist_fig(num_bins=1000):  # TODO make this for all tokens not just probes
        """
        Returns fig showing histogram of similarities learned by "model"
        """
        start = time.time()

        # load data
        probes_acts_df = model.get_multi_probe_prototype_acts_df()
        probe_simmat = cosine_similarity(probes_acts_df.values)
        probe_simmat[np.tril_indices(probe_simmat.shape[0], -1)] = np.nan
        probe_simmat_values = probe_simmat[~np.isnan(probe_simmat)]
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_xlabel('Similarity', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('Frequency', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.xaxis.grid(True)
        # plot
        step_size = 1.0 / num_bins
        bins = np.arange(-1, 1, step_size)
        hist, _ = np.histogram(probe_simmat_values, bins=bins)
        x_binned = bins[:-1]
        ax.plot(x_binned, hist, '-', linewidth=FigsConfigs.LINEWIDTH, c='black')
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_cat_sim_dh_fig(),
            make_probe_sim_hist_fig()]
    return figs


def make_ba_by_probe_figs(model):
    def make_ba_breakdown_annotated_fig(context_type):
        """
        Returns fig showing ranking of each probe's avg_probe_fs broken down by category
        """
        start = time.time()
        cats_per_axis = 16
        x_cycler = cycle(range(cats_per_axis))
        # load data
        cats_sorted_by_fs = model.sort_cats_by_ba(context_type)
        sorted_cat_avg_probe_fs_lists = model.get_sorted_cat_avg_probe_ba_lists(cats_sorted_by_fs, 'ordered')
        mean_fs = np.mean(model.avg_probe_fs_o_list)
        xys = []
        for cat, cat_avg_probe_fs_list in zip(cats_sorted_by_fs, sorted_cat_avg_probe_fs_lists):
            cat_probes = model.hub.probe_store.cat_probe_list_dict[cat]
            x = [next(x_cycler)] * len(cat_probes)
            y = cat_avg_probe_fs_list
            xys.append((x, y, cat_probes))
        # fig
        num_axes = len(model.hub.probe_store.cats) // cats_per_axis + 1
        fig, axarr = plt.subplots(num_axes, figsize=(FigsConfigs.MAX_FIG_WIDTH, 6 * num_axes), dpi=FigsConfigs.DPI)
        for n, ax in enumerate(axarr):
            # truncate data
            xys_truncated = xys[n * cats_per_axis: (n + 1) * cats_per_axis]
            cats_sorted_by_fs_truncated = cats_sorted_by_fs[n * cats_per_axis: (n + 1) * cats_per_axis]
            # axis
            ax.set_ylabel('Avg Probe Balanced Accuracy ({})'.format(context_type),
                          fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            ax.set_xticks(np.arange(cats_per_axis), minor=False)  # shifts xtick labels right
            ax.set_xticklabels(cats_sorted_by_fs_truncated, minor=False, fontsize=FigsConfigs.TICKLABEL_FONT_SIZE,
                               rotation=90)
            ax.set_xlim([0, cats_per_axis])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='both', top='off', right='off')
            ax.axhline(y=mean_fs, alpha=FigsConfigs.FILL_ALPHA, c='grey', linestyle='--', zorder=1)
            # plot
            annotated_y_ints_long_words_prev_cat = []
            for (x, y, cat_probes) in xys_truncated:
                ax.plot(x, y, 'b.', alpha=0)  # this needs to be plot for annotation to work
                # annotate points
                annotated_y_ints = []
                annotated_y_ints_long_words_curr_cat = []
                for x_, y_, probe in zip(x, y, cat_probes):
                    y_int = int(y_)
                    # if annotation coordinate exists or is affected by long word from previous cat, skip to next probe
                    if y_int not in annotated_y_ints and y_int not in annotated_y_ints_long_words_prev_cat:
                        ax.annotate(probe, xy=(x_, y_int), xytext=(2, 0), textcoords='offset points', va='bottom',
                                    fontsize=7)
                        annotated_y_ints.append(y_int)
                        if len(probe) > 7:
                            annotated_y_ints_long_words_curr_cat.append(y_int)
                annotated_y_ints_long_words_prev_cat = annotated_y_ints_long_words_curr_cat
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_freq_vs_fs_pp_fig():
        """
        Returns fig showing probe freq vs avg_probe_fs and vs avg_probe_pp correlation
        """
        start = time.time()
        # load data
        probe_freq_list = [np.sum(model.hub.term_part_freq_dict[probe]) for probe in model.hub.probe_store.types]
        x = probe_freq_list
        ylim1, ylim2 = 100, model.hub.train_terms.num_types
        xys = [(x, model.avg_probe_fs_o_list, 'Avg Probe Balanced Accuracy', ylim1),
               (x, model.avg_probe_pp_list, 'Avg Probe Perplexity', ylim2)]
        # fig
        fig, axarr = plt.subplots(1, 2, figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), sharex='all', dpi=FigsConfigs.DPI)
        for ax_id, (x, y, ylabel, ylim) in enumerate(xys):
            axarr[ax_id].set_ylim([0, ylim])
            axarr[ax_id].set_xlim([0, np.mean(x)])
            axarr[ax_id].spines['right'].set_visible(False)
            axarr[ax_id].spines['top'].set_visible(False)
            axarr[ax_id].tick_params(axis='both', which='both', top='off', right='off')
            axarr[ax_id].set_ylabel(ylabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            axarr[ax_id].set_xlabel('Probe Frequency', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            # plot
            axarr[ax_id].scatter(x, y, s=10)
            # best fit line
            xys_ = zip(x, y)
            if np.any(x):
                plot_best_fit_line(axarr[ax_id], xys_, fontsize=8, x_pos=0.75, y_pos=0.95)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_fs_vs_pp_fig():
        """
        Returns fig showing probe_pp vs probe_fs correlation broken down by category
        """
        start = time.time()

        # load data
        probe_cat_list = [model.hub.probe_store.probe_cat_dict[probe] for probe in model.hub.probe_store.types]
        probe_cat_id_list = [model.hub.probe_store.cats.index(cat) for cat in probe_cat_list]
        xys = []
        for cat_id, cat in enumerate(model.hub.probe_store.cats):
            y = np.asarray(model.avg_probe_fs_o_list)[np.asarray(probe_cat_id_list) == cat_id]
            x = np.asarray(model.avg_probe_pp_list)[np.asarray(probe_cat_id_list) == cat_id]
            xys.append((x, y, cat))
        # fig
        subplot_cols = 5
        subplot_rows = model.hub.probe_store.num_cats // subplot_cols + 1
        fig, axarr = plt.subplots(subplot_rows, subplot_cols, figsize=(FigsConfigs.MAX_FIG_WIDTH, 1.5 * subplot_rows),
                                  sharex='all', sharey='all', dpi=FigsConfigs.DPI)
        text = 'Avg Probe Perplexity'
        fig.text(0.5, 0.05, text, ha='center', va='center')
        fig.text(0.05, 0.5, 'Avg Probe Balanced Accuracy', ha='center', va='center', rotation='vertical')
        cat_id = 0
        for row_id in range(subplot_rows):
            for col_id in range(subplot_cols):
                axarr[row_id, col_id].margins(0.2)
                axarr[row_id, col_id].set_ylim([0, 100])
                axarr[row_id, col_id].set_xlim([0, model.hub.train_terms.num_types])
                axarr[row_id, col_id].spines['right'].set_visible(False)
                axarr[row_id, col_id].spines['top'].set_visible(False)
                axarr[row_id, col_id].tick_params(axis='both', which='both', top='off', right='off')
                axarr[row_id, col_id].set_xticks([0, model.hub.train_terms.num_types])
                axarr[row_id, col_id].set_xticklabels(['0', str(model.hub.train_terms.num_types)],
                                                      minor=False, fontsize=FigsConfigs.TICKLABEL_FONT_SIZE,
                                                      rotation=90)
                if not cat_id == len(xys):
                    x, y, cat = xys[cat_id]
                    cat_id += 1
                else:
                    break
                # plot
                axarr[row_id, col_id].scatter(x, y, s=10, zorder=2)
                axarr[row_id, col_id].axhline(y=50, linestyle='--', zorder=1, c='grey', linewidth=1.0)
                props = dict(boxstyle='round', facecolor='white', alpha=FigsConfigs.FILL_ALPHA)
                axarr[row_id, col_id].text(0.05, 0.9, cat, transform=axarr[row_id, col_id].transAxes,
                                           fontsize=8, verticalalignment='bottom', bbox=props)
                # best fit line
                xys_ = zip(x, y)
                if not np.any(np.isnan(x)):
                    plot_best_fit_line(axarr[row_id, col_id], xys_, fontsize=8, x_pos=0.05, y_pos=0.1, zorder=3)
        fig.subplots_adjust(bottom=0.10)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_ba_breakdown_annotated_fig(context_type) for context_type in ['ordered', 'shuffled', 'none']] +\
           [make_freq_vs_fs_pp_fig(),
            make_fs_vs_pp_fig()]
    return figs


def make_hierarch_cluster_figs(model):
    def make_cat_cluster_fig(cat, bottom_off=False, max_probes=20, metric='cosine', x_max=FigsConfigs.CAT_CLUSTER_XLIM):
        """
        Returns fig showing hierarchical clustering of probes in single category
        """
        start = time.time()
        # load data
        cat_prototypes_df = model.get_single_cat_probe_prototype_acts_df(cat)
        if len(cat_prototypes_df) > max_probes:
            ids = np.random.choice(len(cat_prototypes_df) - 1, max_probes, replace=False)
            cat_prototypes_df = cat_prototypes_df.iloc[ids]
            probes_in_cat = cat_prototypes_df.index
        else:
            probes_in_cat = cat_prototypes_df.index.tolist()
        # fig
        rcParams['lines.linewidth'] = 2.0
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
        # dendrogram
        dist_matrix = pdist(cat_prototypes_df.values, metric=metric)
        linkages = linkage(dist_matrix, method='complete')
        dendrogram(linkages,
                   ax=ax,
                   leaf_label_func=lambda x: probes_in_cat[x],
                   orientation='right',
                   leaf_font_size=8)
        ax.set_title(cat)
        ax.set_xlim([0, x_max])
        ax.tick_params(axis='both', which='both', top='off', right='off', left='off')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if bottom_off:
            ax.xaxis.set_ticklabels([])  # hides ticklabels
            ax.tick_params(axis='both', which='both', bottom='off')
            ax.spines['bottom'].set_visible(False)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_cat_cluster_fig(cat) for cat in model.hub.probe_store.cats]
    return figs


def make_corpus_stats_figs(model):
    def make_cat_count_fig():
        """
        Returns fig showing sum of frequencies in terms of all members in each category
        """
        start = time.time()
        # load data
        cat_probe_count_dict = {cat: 0 for cat in model.hub.probe_store.cats}
        for term, locs in model.hub.term_reordered_locs_dict.items():
            if term in model.hub.probe_store.types:
                cat = model.hub.probe_store.probe_cat_dict[term]
                cat_probe_count_dict[cat] += len(locs)

        sorted_cats, sorted_cat_counts = zip(*sorted(cat_probe_count_dict.items(), key=lambda i: i[1]))

        x = np.arange(model.hub.probe_store.num_cats)
        # correlate sorted_cat_counts with cat_fs
        sorted_cat_avg_probe_fs_lists = model.get_sorted_cat_avg_probe_ba_lists(sorted_cats)
        df = pd.DataFrame(data={'cat': sorted_cats,
                                'sorted_cat_counts': sorted_cat_counts,
                                'cat_fs': [np.mean(cat_avg_probe_fs_list)
                                           for cat_avg_probe_fs_list in sorted_cat_avg_probe_fs_lists]})
        df['num_types'] = df['cat'].apply(
            lambda cat: len(model.hub.probe_store.cat_probe_list_dict[cat]))  # this is just for printing info
        corr = df.corr()['cat_fs'].loc['sorted_cat_counts']
        # fig
        fig, ax = plt.subplots(1, figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xticks(np.arange(model.hub.probe_store.num_cats), minor=False)
        ax.set_xticklabels(sorted_cats, minor=False, fontsize=FigsConfigs.TICKLABEL_FONT_SIZE, rotation=90)
        ax.set_ylabel('Train Frequency', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_title('Correlation cat_fs & probe count: {:.2f}'.format(corr))
        # plot
        ax.plot(x, sorted_cat_counts, '-', linewidth=FigsConfigs.LINEWIDTH)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_sent_lens_fig():
        start = time.time()
        # load data
        ys = [('Avg Utterance Length',
               model.hub.make_sentence_length_stat(model.hub.reordered_tokens, is_avg=True)),
              ('Std Utterance Length',
               model.hub.make_sentence_length_stat(model.hub.reordered_tokens, is_avg=False))]
        # fig
        fig, axarr = plt.subplots(2, figsize=(FigsConfigs.MAX_FIG_WIDTH, 7), dpi=FigsConfigs.DPI)
        for n, (ylabel, y) in enumerate(ys):
            axarr[n].spines['top'].set_visible(False)
            axarr[n].spines['right'].set_visible(False)
            axarr[n].tick_params(axis='both', which='both', top='off', right='off')
            axarr[n].xaxis.set_major_formatter(FuncFormatter(human_format))
            axarr[n].set_ylabel(ylabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            axarr[n].set_xlabel('Utterance (Chronological)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            # plot
            axarr[n].plot(y, linewidth=FigsConfigs.LINEWIDTH)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_doc_entropy_fig():
        start = time.time()

        # load data
        x = range(model.num_docs)
        y = model.hub.part_entropies
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Corpus Partition', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('Shannon Entropy', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # plot
        ax.plot(x, y, '-', linewidth=FigsConfigs.LINEWIDTH)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_cat_count_fig(),
            make_doc_entropy_fig()]
    if '.' in model.hub.train_terms.types:  # figs require puncuation
        figs += [make_sent_lens_fig()]
    return figs


def make_probe_acts_figs(model, field_input):
    def make_acts_dh_fig(probe=None, cluster_cols=True, cluster_rows=True):
        """
        Returns fig showing dendrogram heatmap of exemplar activations of single probe or all probes in model
        """
        start = time.time()
        # make acts_mat
        if probe:
            single_acts_df = model.get_single_probe_exemplar_acts_df(probe)
            acts_mat = single_acts_df.values[:FigsConfigs.MAX_NUM_ACTS]
        else:
            multi_acts_df = model.get_multi_probe_prototype_acts_df()
            acts_mat = multi_acts_df.values[:FigsConfigs.MAX_NUM_ACTS]
        vmin, vmax = round(np.min(acts_mat), 1), round(np.max(acts_mat), 1)
        print('Acts mat | min: {:.2} max: {:.2}'.format(vmin, vmax))
        # fig
        fig, ax_heatmap = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6), dpi=FigsConfigs.DPI)
        ax_heatmap.yaxis.tick_right()
        ax_heatmap.set_xlabel('Hidden Units', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        num_acts = len(acts_mat)
        if probe:
            ax_heatmap.set_ylabel('{} Examples of "{}"'.format(num_acts, probe), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        else:
            ax_heatmap.set_ylabel('All {}  probes'.format(num_acts), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        divider = make_axes_locatable(ax_heatmap)
        ax_dendleft = divider.append_axes("right", 1.0, pad=0.0, sharey=ax_heatmap)
        ax_dendleft.set_frame_on(False)
        ax_dendtop = divider.append_axes("top", 1.0, pad=0.0, sharex=ax_heatmap)
        ax_dendtop.set_frame_on(False)
        ax_colorbar = divider.append_axes("left", 0.2, pad=0.5)
        rcParams['lines.linewidth'] = 0.5  # set linewidth of dendrogram
        # side dendrogram
        if cluster_rows:
            lnk0 = linkage(pdist(acts_mat))
            dg0 = dendrogram(lnk0,
                             ax=ax_dendleft,
                             orientation='right',
                             color_threshold=-1,
                             no_labels=True)
            z = acts_mat[dg0['leaves'], :]  # sorting rows
        else:
            z = acts_mat
        # top dendrogram
        if cluster_cols:
            lnk1 = linkage(pdist(acts_mat.T))
            dg1 = dendrogram(lnk1,
                             ax=ax_dendtop,
                             color_threshold=-1,
                             no_labels=True)
            z = z[:, dg1['leaves']]  # reorder cols to match leaves of dendrogram
        else:
            z = z
        # heatmap
        im = ax_heatmap.imshow(z[::-1],
                               aspect='auto',
                               cmap=plt.cm.jet,
                               interpolation='nearest',
                               extent=(0, ax_dendtop.get_xlim()[1], 0, ax_dendleft.get_ylim()[1]),
                               vmin=vmin,
                               vmax=vmax)
        # colorbar
        cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax])
        cb.ax.set_xticklabels([vmin, vmax])
        cb.set_label('Strength of Activation', labelpad=-50, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # hide heatmap labels
        ax_heatmap.xaxis.set_ticklabels([])
        ax_heatmap.yaxis.set_ticklabels([])
        # Hide all tick lines
        lines = (ax_heatmap.xaxis.get_ticklines() +
                 ax_heatmap.yaxis.get_ticklines() +
                 ax_dendleft.xaxis.get_ticklines() +
                 ax_dendleft.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        lines = (ax_dendtop.xaxis.get_ticklines() +
                 ax_dendtop.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        # make dendrogram labels invisible
        plt.setp(ax_dendleft.get_yticklabels() + ax_dendleft.get_xticklabels(),
                 visible=False)
        plt.setp(ax_dendtop.get_xticklabels() + ax_dendtop.get_yticklabels(),
                 visible=False)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_acts_dh_fig(probe) for probe in field_input]
    return figs


def make_compare_fs_trajs_figs(model, field_input):
    def make_comp_fs_traj_fig(probes):
        """
        Returns fig showing avg_probe_fs trajectories for first and second half of "probes"
        """
        start = time.time()

        assert len(probes) % 2 == 0
        palette = iter(sns.color_palette("hls", 2))
        # load data
        xys = []
        midpoint = len(probes) // 2
        for n, probes_half in enumerate([probes[:midpoint], probes[midpoint:]]):

            avg_probe_fs_trajs_mat = model.fs_traj_df[[col for col in model.fs_traj_df
                                                       if col.endswith('_o')
                                                       and col.split('_')[0] in probes_half]].values
            y = np.mean(avg_probe_fs_trajs_mat, axis=0)
            std = np.std(avg_probe_fs_trajs_mat, axis=0)
            num_probes_half = avg_probe_fs_trajs_mat.shape[0]
            x = model.get_data_step_axis()
            xys.append((x, y, std, n, num_probes_half))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_ylim([40, 100])
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('Avg Probe Balanced Accuracy', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.axhline(50, linestyle='--', color='grey')
        # plot
        for (x, y, std, n, num_probes_half) in xys:
            ax.plot(x, y, c=next(palette), label='group {} (mean +/- std) n={}'.format(n, num_probes_half))
            ax.fill_between(x, y + std, y - std, alpha=FigsConfigs.FILL_ALPHA, color='grey')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=FigsConfigs.LEG_FONTSIZE, loc='best')
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_comp_freqs_fig(probes):
        """
        Returns fig showing average frequency trajectories of multiple classes of tokens
        """
        assert model.configs_dict['num_parts'] > 1  # x axis is documents
        start = time.time()

        palette = iter(sns.color_palette("hls", 2))
        # load data
        xys = []
        midpoint = len(probes) // 2
        for n, probes_half in enumerate([probes[:midpoint], probes[midpoint:]]):
            doc_freq_trajs = []
            for probe in probes_half:
                doc_freq_traj = model.hub.term_part_freq_dict[probe]
                doc_freq_trajs.append(doc_freq_traj)
            x = range(model.configs_dict['num_parts'])
            y = np.mean(doc_freq_trajs, axis=0)
            std = np.std(doc_freq_trajs, axis=0)
            xys.append((x, y, std, n))
        # fig
        fig, ax = plt.subplots(1, figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_xlabel('Partitions', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('Frequency', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.axhline(0, linestyle='--', color='grey')
        # plot
        for (x, y, ci, n) in xys:
            ax.plot(x, y, '-', linewidth=FigsConfigs.LINEWIDTH, c=next(palette),
                    label='group {} avg probe freq +/- std'.format(n))
            ax.fill_between(x, y + ci, y - ci, alpha=FigsConfigs.FILL_ALPHA, color='grey')
        ax.legend(fontsize=FigsConfigs.LEG_FONTSIZE, loc='best')
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_comp_fs_traj_fig(field_input),
            make_comp_freqs_fig(field_input)]
    return figs


def make_cum_freq_trajs_figs(model):
    def make_cfreq_traj_fig(probes):
        """
        Returns fig showing cumulative frequency trajectories of "probes"
        """
        start = time.time()

        palette = iter(sns.color_palette("hls", len(probes)))
        # load data
        xys = []
        for probe in probes:
            x = range(model.num_docs)
            y = np.cumsum(model.hub.term_part_freq_dict[probe])
            if x:
                last_y, last_x = y[-1], x[-1]
            else:
                last_y, last_x = 0, 0  # in case x is empty
            xys.append((x, y, last_x, last_y, probe))
        y_thr = np.max([xy[3] for xy in xys]) / 10  # threhsold is at third from max
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_title(model.hub.probe_store.probe_cat_dict[probes[0]])
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('Cumulative Frequency', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        # plot
        for (x, y, last_x, last_y, probe) in xys:
            ax.plot(x, y, '-', linewidth=1.0, c=next(palette))
            if last_y > y_thr:
                plt.annotate(probe, xy=(last_x, last_y),
                             xytext=(0, 0), textcoords='offset points',
                             va='center', fontsize=FigsConfigs.LEG_FONTSIZE, bbox=dict(boxstyle='round', fc='w'))
        ax.legend(fontsize=FigsConfigs.LEG_FONTSIZE, loc='upper left')
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_cfreq_traj_fig(cat_probes)
            for cat_probes in model.hub.probe_store.cat_probe_list_dict.values()]
    return figs


def make_term_freqs_figs(model,
                         field_input):
    def make_term_freq_hist_fig(terms, caption=None):
        start = time.time()
        palette = iter(sns.color_palette("hls", len(terms)))
        props = dict(boxstyle='round', facecolor='white', alpha=FigsConfigs.FILL_ALPHA)
        # load data
        xys = []
        x = range(model.configs_dict['num_parts'])
        for term in terms:
            y = model.hub.term_part_freq_dict[term]
            xys.append((x, y, term))
        # fig
        fig, ax = plt.subplots(1, figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_xlabel('Partition', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('Frequency', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        # plot
        for (x, y, term) in xys:
            try:
                probe_id = model.hub.probe_store.probe_id_dict[term]
            except KeyError:  # is not probe
                avg_probe_fs = '/'
                avg_probe_pp = 0
            else:
                avg_probe_fs = model.avg_probe_fs_o_list[probe_id]
                avg_probe_pp = model.avg_probe_pp_list[probe_id]

            ax.plot(x, y, '-', linewidth=FigsConfigs.LINEWIDTH, c=next(palette),
                    label='{} (total freq : {}, avg_fs: {:2.3}, avg_pp: {:,})'.format(
                        term, int(sum(model.hub.term_part_freq_dict[term])),
                        avg_probe_fs, int(avg_probe_pp)))
        if caption is not None:
            ax.text(0.05, 0.85, caption, transform=ax.transAxes,
                    fontsize=FigsConfigs.TICKLABEL_FONT_SIZE, verticalalignment='bottom', bbox=props)
        ax.legend(fontsize=FigsConfigs.LEG_FONTSIZE, loc='upper right')  # loc should not interfere with textbox
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_probe_freq_start_end_comp_fs_corr_fig():

        half_num_docs = model.num_docs // 2
        # load data
        x = []
        for probe in model.hub.probe_store.types:
            probe_doc_freqs = model.hub.term_part_freq_dict[probe]
            probe_freq_start = sum(probe_doc_freqs[:half_num_docs])
            probe_freq_end = sum(probe_doc_freqs[-half_num_docs:])
            x_ = probe_freq_end - probe_freq_start
            x.append(x_)
        y = model.avg_probe_fs_o_list
        # fig
        fig, ax = plt.subplots(1, figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_xlabel('Probe Freq Second Half - First Half', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('Avg Probe Balanced Accuracy', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        # plot
        ax.scatter(x, y, s=FigsConfigs.MARKERSIZE, facecolor='black')
        # best fit line
        if not np.any(np.isnan(x)):
            plot_best_fit_line(ax, zip(x, y), fontsize=12, x_pos=0.75)
        ax.legend(fontsize=FigsConfigs.LEG_FONTSIZE, loc='best')
        fig.tight_layout()
        return fig

    figs = [make_term_freq_hist_fig(field_input),
            make_probe_freq_start_end_comp_fs_corr_fig()] + \
           [make_term_freq_hist_fig(q[:FigsConfigs.NUM_PROBES_IN_QUARTILE])
            for q in split_probes_into_quartiles_by_fs(model)]
    return figs


def make_probe_context_figs(model, field_input):
    def make_probe_context_fig(probe, num_context_words=20):
        """
        Returns fig showing tokens that occur most frequently before "probe"
        """
        start = time.time()

        probe_id = model.hub.probe_store.probe_id_dict[probe]
        bptt_steps = int(model.configs_dict['bptt_steps'])
        num_h_axes = 3
        num_v_axes = ((bptt_steps) // num_h_axes)
        if num_h_axes * num_v_axes < bptt_steps:
            num_v_axes += 1
        # load data
        probe_x_mat = model.hub.probe_x_mats[model.hub.probe_store.probe_id_dict[probe]]
        probe_context_df = pd.DataFrame(index=[probe_id] * len(probe_x_mat), data=probe_x_mat)
        context_dict_list = []
        for bptt_step in range(bptt_steps - 1):
            crosstab_df = pd.crosstab(probe_context_df[bptt_step], probe_context_df.index)
            context_dict = crosstab_df.sort_values(probe_id)[-num_context_words:].to_dict('index')
            context_dict = {model.hub.train_terms.types[token_id]: d[probe_id] for token_id, d in context_dict.items()}
            context_dict_list.append(context_dict)
        # add unigram freqs
        unigram_dict = dict(model.hub.train_terms.term_freq_dict.most_common(num_context_words))
        context_dict_list.insert(0, unigram_dict)
        # fig
        fig, axarr = plt.subplots(num_v_axes, num_h_axes,
                                  figsize=(FigsConfigs.MAX_FIG_WIDTH, num_v_axes * 4), dpi=FigsConfigs.DPI)
        fig.suptitle('"{}"'.format(probe))
        for ax, context_dict, bptt_step in zip_longest(
                axarr.flatten(), context_dict_list, range(bptt_steps + 1)[::-1]):
            # plot
            if not context_dict:
                ax.axis('off')
                continue
            ytlabels, token_freqs = zip(*sorted(context_dict.items(), key=itemgetter(1), reverse=True))
            mat = np.asarray(token_freqs)[:, np.newaxis]
            sns.heatmap(mat, ax=ax, square=True, annot=False,
                        annot_kws={"size": 6}, cbar_kws={"shrink": .5}, cmap='jet', fmt='d')
            # colorbar
            cbar = ax.collections[0].colorbar
            cbar.set_label('Frequency')
            # ax (needs to be below plot for axes to be labeled)
            ax.set_yticks(range(num_context_words))
            ax.set_yticklabels(ytlabels[::-1], rotation=0)  # reverse labels because default is upwards direction
            ax.set_xticklabels([])
            if bptt_step != bptt_steps:
                ax.set_title('Terms at distance t-{}'.format(bptt_step))
            else:
                ax.set_title('Unigram frequencies')
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_probe_context_alternative_fig(probe, num_context_words=20):
        """
        Returns fig showing tokens that occur most frequently before "probe" and includes unigram frequency information
        """
        start = time.time()

        probe_id = model.hub.probe_store.probe_id_dict[probe]
        x = np.arange(num_context_words)
        bptt_steps = int(model.configs_dict['bptt_steps'])
        num_h_axes = 3
        num_v_axes = ((bptt_steps) // num_h_axes)
        if num_h_axes * num_v_axes < bptt_steps:
            num_v_axes += 1
        # load data
        ytlabels, unigram_freqs = zip(*islice(model.hub.train_terms.type_freq_dict_no_oov.items(), 0, num_context_words))
        unigram_freqs_norm = np.divide(unigram_freqs, np.max(unigram_freqs).astype(np.float))
        probe_x_mat = model.hub.probe_x_mats[model.hub.probe_store.probe_id_dict[probe]]
        probe_context_df = pd.DataFrame(index=[probe_id] * len(probe_x_mat), data=probe_x_mat)
        xys = []
        for bptt_step in range(bptt_steps - 1):
            crosstab_df = pd.crosstab(probe_context_df[bptt_step], probe_context_df.index)
            context_dict = crosstab_df.sort_values(probe_id)[-num_context_words:].to_dict('index')
            context_dict = {model.hub.train_terms.types[token_id]: d[probe_id] for token_id, d in context_dict.items()}
            tups = sorted(context_dict.items(), key=lambda i: model.hub.train_terms.term_freq_dict[i[0]], reverse=True)
            y_unnorm = [tup[1] for tup in tups[:num_context_words]]
            y = np.divide(y_unnorm, np.max(y_unnorm).astype(np.float))
            distance = (bptt_steps - 1) - bptt_step
            xys.append((x, y, distance))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
        ax.set_ylabel('Term Normalized Frequency', fontsize=FigsConfigs.AXLABEL_FONT_SIZE, labelpad=0.0)
        ax.set_xticks(x, minor=False)
        ax.set_xticklabels(ytlabels, minor=False, fontsize=FigsConfigs.AXLABEL_FONT_SIZE, rotation=90)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # plot
        ax.plot(x, unigram_freqs_norm, '--', linewidth=FigsConfigs.LINEWIDTH, label='unigram frequencies',
                color='black')
        for x, y, distance in xys:
            ax.plot(x, y, '-', linewidth=FigsConfigs.LINEWIDTH, label='distance t-{}'.format(distance))
        plt.tight_layout()
        plt.legend()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_probe_context_fig(probe) for probe in field_input] + \
           [make_probe_context_alternative_fig(probe) for probe in field_input]
    return figs


def make_neighbors_figs(model, field_input):
    def make_neighbors_table_fig(terms):
        """
        Returns fig showing 10 nearest neighbors from "model" for "tokens"
        """
        start = time.time()
        # load data
        neighbors_mat_list = []
        col_labels_list = []
        num_cols = 5  # fixed
        num_neighbors = 10  # fixed
        for i in range(0, len(terms), num_cols):  # split probes into even sized lists
            col_labels = terms[i:i + num_cols]
            neighbors_mat = np.chararray((num_neighbors, num_cols), itemsize=20, unicode=True)
            neighbors_mat[:] = ''  # initialize so that mpl can read table
            # make column
            for col_id, term in enumerate(col_labels):
                term_id = model.hub.train_terms.term_id_dict[term]
                token_sims = model.term_simmat[term_id]
                neighbor_tuples = [(model.hub.train_terms.types[term_id], token_sim) for term_id, token_sim in
                                   enumerate(token_sims)
                                   if model.hub.train_terms.types[term_id] != term]
                neighbor_tuples = sorted(neighbor_tuples, key=itemgetter(1), reverse=True)[:num_neighbors]
                neighbors_mat_col = ['{:>15} {:.2f}'.format(tuple[0], tuple[1])
                                     for tuple in neighbor_tuples if tuple[0] != term]
                neighbors_mat[:, col_id] = neighbors_mat_col
            # collect info for plotting
            neighbors_mat_list.append(neighbors_mat)
            length_diff = num_cols - len(col_labels)
            col_labels_list.append(col_labels + [' '] * length_diff)
        # fig
        num_tables = max(2, len(neighbors_mat_list))  # max 2 otherwise axarr is  not indexable
        fig, axarr = plt.subplots(num_tables, 1,
                                  figsize=(FigsConfigs.MAX_FIG_WIDTH, num_tables * (num_neighbors / 4.)),
                                  dpi=FigsConfigs.DPI)
        for ax, neighbors_mat, col_labels in zip_longest(axarr, neighbors_mat_list, col_labels_list):
            ax.axis('off')
            if neighbors_mat is not None:  # this allows turning off of axis even when neighbors_mat list length is < 2
                table_ = ax.table(cellText=neighbors_mat, colLabels=col_labels,
                                  loc='center', colWidths=[0.2] * num_cols)
                table_.auto_set_font_size(False)
                table_.set_fontsize(8)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_sg_neighbors_table_fig(terms, model_id=FigsConfigs.SKIPGRAM_MODEL_ID):
        """
        Returns fig showing nearest 10 neighbors from pre-trained skip-gram model for "tokens"
        """
        start = time.time()

        # load data
        token_simmat = np.load(GlobalConfigs.SG_DIR / 'sg_token_simmat_model_{}.npy'.format(model_id))
        token_list = np.load(GlobalConfigs.SG_DIR, 'sg_types.npy')
        token_id_dict = {token: token_id for token_id, token in enumerate(token_list)}
        # make neighbors_mat
        neighbors_mat_list = []
        col_labels_list = []
        num_cols = 5  # fixed
        num_neighbors = 10  # fixed
        for i in range(0, len(terms), num_cols):  # split probes into even sized lists
            col_labels = terms[i:i + num_cols]
            neighbors_mat = np.chararray((num_neighbors, num_cols), itemsize=20, unicode=True)
            neighbors_mat[:] = ''  # initialize so that mpl can read table
            # make column
            for col_id, token in enumerate(col_labels):
                token_id = token_id_dict[token]
                token_sims = token_simmat[token_id]
                neighbor_tuples = [(token_list[token_id], token_sim) for token_id, token_sim in enumerate(token_sims)
                                   if token_list[token_id] != token]
                neighbor_tuples = sorted(neighbor_tuples, key=itemgetter(1), reverse=True)[:num_neighbors]
                neighbors_mat_col = ['{:>15} {:.2f}'.format(tuple[0], tuple[1])
                                     for tuple in neighbor_tuples if tuple[0] != token]
                neighbors_mat[:, col_id] = neighbors_mat_col
            # collect info for plotting
            neighbors_mat_list.append(neighbors_mat)
            length_diff = num_cols - len(col_labels)
            col_labels_list.append(['{} (skip-gram)'.format(token_) for token_ in col_labels] + [' '] * length_diff)
        # fig
        num_tables = max(2, len(neighbors_mat_list))  # max 2 otherwise axarr is  not indexable
        fig, axarr = plt.subplots(num_tables, 1,
                                  figsize=(FigsConfigs.MAX_FIG_WIDTH, num_tables * (num_neighbors / 4.)),
                                  dpi=FigsConfigs.DPI)
        for ax, neighbors_mat, col_labels in zip_longest(axarr, neighbors_mat_list, col_labels_list):
            ax.axis('off')
            if neighbors_mat is not None:  # this allows turning off of axis even when neighbors_mat list length is < 2
                table_ = ax.table(cellText=neighbors_mat, colLabels=col_labels,
                                  loc='center', colWidths=[0.2] * num_cols)
                table_.auto_set_font_size(False)
                table_.set_fontsize(8)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    # figs = [make_neighbors_table_fig(field_input),
    #         make_sg_neighbors_table_fig(field_input)]  # TODO implement sg
    figs = [make_neighbors_table_fig(field_input)]
    return figs


def make_multi_hierarch_cluster_figs(model, field_input):
    def make_multi_cat_clust_fig(cats, metric='cosine'):  # TODO make into config
        """
        Returns fig showing hierarchical clustering of probes from multiple categories
        """
        start = time.time()
        # load data
        df = pd.DataFrame(pd.concat((model.get_single_cat_probe_prototype_acts_df(cat) for cat in cats), axis=0))
        cat_acts_mat = df.values
        cats_probe_list = df.index
        # fig
        rcParams['lines.linewidth'] = 2.0
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 5 * len(cats)), dpi=FigsConfigs.DPI)
        # dendrogram
        dist_matrix = pdist(cat_acts_mat, metric)
        linkages = linkage(dist_matrix, method='complete')
        dendrogram(linkages,
                   ax=ax,
                   labels=cats_probe_list,
                   orientation='right',
                   leaf_font_size=10)
        ax.tick_params(axis='both', which='both', top='off', right='off', left='off')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_multi_cat_clust_fig(field_input)]
    return figs


def make_probes_by_act_figs(model, field_input):
    def make_probes_by_act_fig(hidden_unit_ids, num_probes=20):
        """
        Returns fig showing probes with highest activation at "hidden_unit_ids"
        """
        start = time.time()
        hidden_unit_ids = [int(i) for i in hidden_unit_ids]
        # load data
        multi_acts_df = model.get_multi_probe_prototype_acts_df()
        acts_mat = multi_acts_df.values
        # make probes_by_acts_mat
        probe_by_acts_mat_list = []
        col_labels_list = []
        num_cols = 5  # fixed
        for i in range(0, len(hidden_unit_ids), num_cols):  # split into even sized lists
            hidden_unit_ids_ = hidden_unit_ids[i:i + num_cols]
            probes_by_acts_mat = np.chararray((num_probes, num_cols), itemsize=20, unicode=True)
            probes_by_acts_mat[:] = ''  # initialize so that mpl can read table
            # make column
            for col_id, hidden_unit_id in enumerate(hidden_unit_ids_):
                acts_mat_col = acts_mat[:, hidden_unit_id]
                tups = [(model.hub.probe_store.types[probe_id], act) for probe_id, act in enumerate(acts_mat_col)]
                sorted_tups = sorted(tups, key=itemgetter(1), reverse=True)[:num_probes]
                probe_by_act_mat_col = ['{:>15} {:.2f}'.format(tup[0], tup[1]) for tup in sorted_tups]
                probes_by_acts_mat[:, col_id] = probe_by_act_mat_col
            # collect info for plotting
            probe_by_acts_mat_list.append(probes_by_acts_mat)
            length_diff = num_cols - len(hidden_unit_ids_)
            for i in range(length_diff):
                hidden_unit_ids_.append(' ')  # add space so table can be read properly
            col_labels_list.append(['hidden unit #{}'.format(hidden_unit_id)
                                    if hidden_unit_id != ' ' else ' '
                                    for hidden_unit_id in hidden_unit_ids_])
        # fig
        num_tables = len(probe_by_acts_mat_list)
        if num_tables == 1:
            fig, axarr = plt.subplots(1, 1,
                                      figsize=(FigsConfigs.MAX_FIG_WIDTH, num_tables * (num_probes / 4.)),
                                      dpi=FigsConfigs.DPI)
            axarr = [axarr]
        else:
            fig, axarr = plt.subplots(num_tables, 1,
                                      figsize=(FigsConfigs.MAX_FIG_WIDTH, num_tables * (num_probes / 4.)),
                                      dpi=FigsConfigs.DPI)
        for ax, probes_by_acts_mat, col_labels in zip(axarr, probe_by_acts_mat_list, col_labels_list):
            ax.axis('off')
            table_ = ax.table(cellText=probes_by_acts_mat, colLabels=col_labels, loc='center')
            table_.auto_set_font_size(False)
            table_.set_fontsize(8)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_probes_by_act_fig(field_input)]
    return figs


def make_probe_last_cat_corr_figs(model, field_input):
    def make_multi_probe_last_cat_corr_traj_fig(probe):
        """
        Returns fig showing correlation between exemplar activations of "Probe"
         and its prototype activation at the last time step.
        """
        start = time.time()

        cat = model.hub.probe_store.probe_cat_dict[probe]
        num_acts = min(FigsConfigs.MAX_NUM_ACTS, model.get_single_probe_exemplar_acts_df(probe).values.shape[0])
        n = min(model.timepoint, FigsConfigs.NUM_TIMEPOINTS_ACTS_CORR)
        corr_mb_names = ['0000000'] + extract_n_elements(model.ckpt_mb_names, n)
        num_corr_mb_names = len(corr_mb_names)
        traj_mat_ax0 = np.zeros((num_acts, num_corr_mb_names))
        traj_mat_ax1 = np.zeros((num_acts, num_corr_mb_names))
        # load data
        model.acts_df = reload_acts_df(model.model_name, corr_mb_names[-1], model.hub.mode)
        last_avg_cat_act = np.mean(model.get_single_cat_probe_prototype_acts_df(cat).values, axis=0)
        last_avg_token_act = np.mean(model.get_single_probe_exemplar_acts_df(probe).values, axis=0)
        for n, mb_name in enumerate(corr_mb_names):
            model.acts_df = reload_acts_df(model.model_name, mb_name, model.hub.mode)
            probe_act = model.get_single_probe_exemplar_acts_df(probe).values[:num_acts]
            traj_mat_ax0[:, n] = [np.corrcoef(token_act, last_avg_token_act)[1, 0] for token_act in probe_act]
            traj_mat_ax1[:, n] = [np.corrcoef(token_act, last_avg_cat_act)[1, 0] for token_act in probe_act]
        xys_ax0, xys_ax1 = [], []
        for row_ax0, row_ax1 in zip(traj_mat_ax0, traj_mat_ax1):
            len_row_ax0 = len(row_ax0)
            x = range(len_row_ax0)
            y = row_ax0
            xys_ax0.append((x, y))
            y = row_ax1
            xys_ax1.append((x, y))
        # fig
        fig, axarr = plt.subplots(2, 1, figsize=(FigsConfigs.MAX_FIG_WIDTH, 7), dpi=FigsConfigs.DPI)
        for n, ax in enumerate(axarr):
            if n == 0:
                ax.set_ylabel('Correlation', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            else:
                ax.set_ylabel('Correlation', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            ax.set_xticks(range(num_corr_mb_names))
            xticklabels = ['{:,}'.format(int(mb_name)) for mb_name in corr_mb_names]
            ax.set_xticklabels(xticklabels, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='both', top='off', right='off')
            ax.set_ylim([0, 1])
        # plot
        for (x, y) in xys_ax0:
            axarr[0].plot(x, y, '-')
            props = dict(boxstyle='round', facecolor='white', alpha=FigsConfigs.FILL_ALPHA)
            axarr[0].text(0.05, 0.9, 'Correlation of "{}" with last avg act of "{}"'.format(probe, probe),
                          transform=axarr[0].transAxes, fontsize=FigsConfigs.LEG_FONTSIZE,
                          verticalalignment='bottom', bbox=props)
        for (x, y) in xys_ax1:
            axarr[1].plot(x, y, '-')
            props = dict(boxstyle='round', facecolor='white', alpha=FigsConfigs.FILL_ALPHA)
            axarr[1].text(0.05, 0.9, 'Correlation of "{}" with last avg act of {}'.format(probe, cat),
                          transform=axarr[1].transAxes, fontsize=FigsConfigs.LEG_FONTSIZE,
                          verticalalignment='bottom', bbox=props)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_multi_probe_last_cat_corr_traj_fig(probe)
            for probe in field_input]
    return figs


def make_probe_cat_corr_figs(model, field_input):
    def make_probe_cat_corr_traj_fig(probe, slice_id, num_slices):
        """
        Returns fig showing correlation between  probe prototype activation and category activations.
        """
        start = time.time()

        sliced_cats = model.hub.probe_store.cats[slice_id:slice_id + num_slices]
        num_sliced_cats = len(sliced_cats)
        # load data
        traj_mat = np.zeros((num_sliced_cats, len(model.ckpt_mb_names)))
        for n, mb_name in enumerate(model.ckpt_mb_names):
            model.acts_df = reload_acts_df(model.model_name, mb_name, model.hub.mode)
            probe_act = np.mean(model.get_single_probe_exemplar_acts_df(probe).values, axis=0)
            probe_act_repeated = [probe_act] * num_sliced_cats
            cat_acts_mat = model.get_multi_cat_prototype_acts_df().values[slice_id:slice_id + num_slices]
            traj_mat[:, n] = [np.corrcoef(act1, act2)[1, 0] for act1, act2 in zip(probe_act_repeated, cat_acts_mat)]
        x = model.get_data_step_axis()
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
        ax.set_ylabel('Correlation'.format(probe), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.set_ylim([0, 1])
        # plot
        for traj, cat in zip(traj_mat, sliced_cats):
            ax.plot(x, traj, '-', linewidth=FigsConfigs.LINEWIDTH, label=cat)
        props = dict(boxstyle='round', facecolor='white', alpha=FigsConfigs.FILL_ALPHA)
        ax.text(0.05, 0.9, probe, transform=ax.transAxes, fontsize=FigsConfigs.LEG_FONTSIZE, verticalalignment='bottom',
                bbox=props)
        plt.legend(loc='best')
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_probe_cat_corr_traj_fig(field_input[0], slice_id, 5)
            for slice_id in range(0, model.hub.probe_store.num_cats, 5)]
    return figs


def make_probe_probe_corr_figs(model, field_input):
    def make_probe_probe_corr_traj_fig(probes):
        """
        Returns fig showing correlation between  probe prototype activation and activations of "alternate probes".
        """
        start = time.time()

        num_leg_per_col = 3
        probes_iter = iter(probes)
        num_probes = len(probes)
        assert num_probes % 2 != 0  # need uneven number of axes to insert legend
        num_rows = num_probes // 2 if num_probes % 2 == 0 else num_probes // 2 + 1
        num_alt_probes = len(FigsConfigs.ALTERNATE_PROBES)
        # load data
        probe_traj_mat_dict = {probe: np.zeros((num_alt_probes, len(model.ckpt_mb_names)))
                               for probe in probes}
        y_mins = []
        for n, mb_name in enumerate(model.ckpt_mb_names):
            model.acts_df = reload_acts_df(model.model_name, mb_name, model.hub.mode)
            probe_acts_df = model.get_multi_probe_prototype_acts_df()
            df_ids = probe_acts_df.index.isin(FigsConfigs.ALTERNATE_PROBES)
            alt_probes_acts = probe_acts_df.loc[df_ids].values
            for probe in probes:
                df_id = probe_acts_df.index == probe
                probe_act = probe_acts_df.iloc[df_id].values
                probe_probe_corrs = [np.corrcoef(probe_act, alt_probe_act)[1, 0]
                                     for alt_probe_act in alt_probes_acts]
                probe_traj_mat_dict[probe][:, n] = probe_probe_corrs
                if n != 0:
                    y_min = min(probe_probe_corrs[1:])
                    y_mins.append(y_min)
        x = model.get_data_step_axis()
        # fig
        fig, axarr = plt.subplots(num_rows, 2, figsize=(FigsConfigs.MAX_FIG_WIDTH, 2 * num_rows), dpi=FigsConfigs.DPI)
        for row_id, row in enumerate(axarr):
            for ax_id, ax in enumerate(row):
                try:
                    probe = next(probes_iter)
                except StopIteration:
                    ax.axis('off')
                    last_ax = axarr[row_id, ax_id - 1]  # make legend for last ax
                    handle, label = last_ax.get_legend_handles_labels()
                    ax.legend(handle, label, loc=6, ncol=num_probes // num_leg_per_col)
                    continue
                else:
                    ax.set_title(probe, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
                    if ax_id % 2 == 0:
                        ax.set_ylabel('Correlation'.format(probe), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
                    if row_id == num_rows - 1:
                        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.tick_params(axis='both', which='both', top='off', right='off')
                    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
                    ax.set_ylim([min(y_mins), 1])
                # plot
                for alt_probe_id, alt_probe in enumerate(FigsConfigs.ALTERNATE_PROBES):
                    traj = probe_traj_mat_dict[probe][alt_probe_id]
                    ax.plot(x, traj, '-', linewidth=FigsConfigs.LINEWIDTH, label=alt_probe)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_probe_probe_corr_traj_fig(field_input)]
    return figs


def make_avg_probe_pp_corrs_figs(model, num_tags=1, num_cats=1, num_terms=50):
    def make_avg_probe_pp_vs_avg_loc_fig():
        """
        Returns fig showing scatter plot of probe_pps and context goodness for all probes
        """
        start = time.time()

        # load data
        x = [model.hub.calc_avg_reordered_loc(probe) for probe in model.hub.probe_store.types]
        y = model.avg_probe_pp_list
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_ylabel('Avg Probe Perplexity', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        xlabel = 'Probe Avg Corpus Location'
        ax.set_xlabel(xlabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # plot
        ax.scatter(x, y, s=FigsConfigs.MARKERSIZE, facecolor='black', zorder=2)
        ax.axhline(y=50, linestyle='--', c='grey', zorder=1)
        xys_ = zip(x, y)
        plot_best_fit_line(ax, xys_, FigsConfigs.LEG_FONTSIZE)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_avg_probe_pp_vs_cat_context_fig(cat):
        """
        Returns fig showing scatter plot of probe_pps and number of times "cat" member in context for all probes
        """
        start = time.time()

        # load data
        x = [np.sum([1 if term in model.hub.probe_store.cat_probe_list_dict[cat] else 0
                     for term in model.hub.probe_context_terms_dict[probe]]) /
             len(model.hub.probe_context_terms_dict[probe])
             for probe in model.hub.probe_store.types]
        x = np.round(x, 2)
        y = model.avg_probe_pp_list
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_ylabel('Avg Probe Perplexity', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        xlabel = 'Percentage of context with members of {}'.format(cat)
        ax.set_xlabel(xlabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # plot
        ax.scatter(x, y, s=FigsConfigs.MARKERSIZE, facecolor='black', zorder=2)
        ax.axhline(y=50, linestyle='--', c='grey', zorder=1)
        xys_ = zip(x, y)
        plot_best_fit_line(ax, xys_, FigsConfigs.LEG_FONTSIZE, x_pos=0.9)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_avg_probe_pp_vs_terms_context_fig(terms):
        """
        Returns fig showing scatter plot of probe_pps and number of times "terms" in context for all probes
        """
        start = time.time()

        # load data
        x = [np.sum([1 if context_term in terms else 0
                     for context_term in model.hub.probe_context_terms_dict[probe]]) /
             len(model.hub.probe_context_terms_dict[probe])
             for probe in model.hub.probe_store.types]
        x = np.round(x, 2)
        y = model.avg_probe_pp_list
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_ylabel('Avg Probe Perplexity', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        xlabel = 'Percentage of context with {}{}'.format(
            ', '.join(['"{}"'.format(term) for term in terms[:5]]), ', etc.' if len(terms) > 5 else '')
        ax.set_xlabel(xlabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # plot
        ax.scatter(x, y, s=FigsConfigs.MARKERSIZE, facecolor='black', zorder=2)
        ax.axhline(y=50, linestyle='--', c='grey', zorder=1)
        xys_ = zip(x, y)
        plot_best_fit_line(ax, xys_, FigsConfigs.LEG_FONTSIZE, x_pos=0.9)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    tups = sorted(model.hub.term_avg_reordered_loc_dict.items(), key=lambda i: i[1])
    terms_sorted_by_avg_loc = list(zip(*tups))[0][:num_terms]
    figs = [make_avg_probe_pp_vs_avg_loc_fig()] + \
           [make_avg_probe_pp_vs_terms_context_fig(model.hub.train_terms.types[:num_terms])] +\
           [make_avg_probe_pp_vs_terms_context_fig(terms_sorted_by_avg_loc[:num_terms])] +\
           [make_avg_probe_pp_vs_cat_context_fig(cat) for cat in model.hub.probe_store.cats[:num_cats]]
    return figs


def make_avg_probe_fs_corrs_figs(model, num_tags=1, num_cats=1, num_terms=1000):
    def make_avg_probe_fs_vs_avg_loc_fig():
        """
        Returns fig showing scatter plot of probe_fss and context goodness for all probes
        """
        start = time.time()

        # load data
        x = [model.hub.calc_avg_reordered_loc(probe) for probe in model.hub.probe_store.types]
        y = model.avg_probe_fs_o_list
        assert len(x) == len(y)
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_ylabel('Avg Probe Balanced Accuracy (%)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        xlabel = 'Probe Avg Corpus Location'
        ax.set_xlabel(xlabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # plot
        ax.scatter(x, y, s=FigsConfigs.MARKERSIZE, facecolor='black', zorder=2)
        ax.axhline(y=50, linestyle='--', c='grey', zorder=1)
        xys_ = zip(x, y)
        plot_best_fit_line(ax, xys_, FigsConfigs.LEG_FONTSIZE)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_avg_probe_fs_vs_num_periods_fig():
        """
        Returns fig showing scatter plot of probe_fss and num_periods for all probes
        """
        start = time.time()

        # load data
        x = model.hub.probe_num_periods_in_context_list
        y = model.avg_probe_fs_o_list
        assert len(x) == len(y)
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_ylabel('Avg Probe Balanced Accuracy (%)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        xlabel = 'Probe Num Periods'
        ax.set_xlabel(xlabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # plot
        ax.scatter(x, y, s=FigsConfigs.MARKERSIZE, facecolor='black', zorder=2)
        ax.axhline(y=50, linestyle='--', c='grey', zorder=1)
        xys_ = zip(x, y)
        plot_best_fit_line(ax, xys_, FigsConfigs.LEG_FONTSIZE)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_avg_probe_fs_vs_tag_entropy_fig():
        """
        Returns fig showing scatter plot of probe_fss and context_overlap for all probes
        """
        start = time.time()

        # load data
        x = model.hub.probe_tag_entropy_list
        y = model.avg_probe_fs_o_list
        assert len(x) == len(y)
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_ylabel('Avg Probe Balanced Accuracy (%)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        xlabel = 'Probe Tag Entropy'
        ax.set_xlabel(xlabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # plot
        ax.scatter(x, y, s=FigsConfigs.MARKERSIZE, facecolor='black', zorder=2)
        ax.axhline(y=50, linestyle='--', c='grey', zorder=1)
        xys_ = zip(x, y)
        plot_best_fit_line(ax, xys_, FigsConfigs.LEG_FONTSIZE)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_avg_probe_fs_vs_tag_context_fig(tag, tag_freq_thr=1):
        """
        Returns fig showing scatter plot of probe_fss and numbe rof times "tag" in context for all probes
        """
        start = time.time()

        # load data
        x = [np.sum([1 if model.hub.train_terms.term_tags_dict[term][tag] > tag_freq_thr else 0
                     for term in model.hub.probe_context_terms_dict[probe]]) /
             len(model.hub.probe_context_terms_dict[probe])
             for probe in model.hub.probe_store.types]
        x = np.round(x, 2)
        y = model.avg_probe_fs_o_list
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_ylabel('Avg Probe Balanced Accuracy (%)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        xlabel = 'Percentage of context terms tagged as "{}"'.format(tag)
        ax.set_xlabel(xlabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # plot
        ax.scatter(x, y, s=FigsConfigs.MARKERSIZE, facecolor='black', zorder=2)
        ax.axhline(y=50, linestyle='--', c='grey', zorder=1)
        xys_ = zip(x, y)
        plot_best_fit_line(ax, xys_, FigsConfigs.LEG_FONTSIZE, x_pos=0.9)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_avg_probe_fs_vs_cat_context_fig(cat):
        """
        Returns fig showing scatter plot of probe_fss and number of times "cat" member in context for all probes
        """
        start = time.time()

        # load data
        x = [np.sum([1 if term in model.hub.probe_store.cat_probe_list_dict[cat] else 0
                     for term in model.hub.probe_context_terms_dict[probe]]) /
             len(model.hub.probe_context_terms_dict[probe])
             for probe in model.hub.probe_store.types]
        x = np.round(x, 2)
        y = model.avg_probe_fs_o_list
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_ylabel('Avg Probe Balanced Accuracy (%)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        xlabel = 'Percentage of context with members of {}'.format(cat)
        ax.set_xlabel(xlabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # plot
        ax.scatter(x, y, s=FigsConfigs.MARKERSIZE, facecolor='black', zorder=2)
        ax.axhline(y=50, linestyle='--', c='grey', zorder=1)
        xys_ = zip(x, y)
        plot_best_fit_line(ax, xys_, FigsConfigs.LEG_FONTSIZE, x_pos=0.9)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_avg_probe_fs_vs_terms_context_fig(terms):
        """
        Returns fig showing scatter plot of probe_fss and number of times "terms" in context for all probes
        """
        start = time.time()

        # load data
        x = [np.sum([1 if context_term in terms else 0
                     for context_term in model.hub.probe_context_terms_dict[probe]]) /
             len(model.hub.probe_context_terms_dict[probe])
             for probe in model.hub.probe_store.types]
        x = np.round(x, 2)
        y = model.avg_probe_fs_o_list
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_ylabel('Avg Probe Balanced Accuracy (%)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        xlabel = 'Percentage of context with {}{}'.format(
            ', '.join(['"{}"'.format(term) for term in terms[:5]]), ', etc.' if len(terms) > 5 else '')
        ax.set_xlabel(xlabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # plot
        ax.scatter(x, y, s=FigsConfigs.MARKERSIZE, facecolor='black', zorder=2)
        ax.axhline(y=50, linestyle='--', c='grey', zorder=1)
        xys_ = zip(x, y)
        plot_best_fit_line(ax, xys_, FigsConfigs.LEG_FONTSIZE, x_pos=0.9)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    tups = sorted(model.hub.term_avg_reordered_loc_dict.items(), key=lambda i: i[1])
    terms_sorted_by_avg_loc = list(zip(*tups))[0][:num_terms]
    figs = [make_avg_probe_fs_vs_avg_loc_fig(),
            make_avg_probe_fs_vs_tag_entropy_fig(),
            make_avg_probe_fs_vs_num_periods_fig()] + \
           [make_avg_probe_fs_vs_tag_context_fig(tag) for tag in model.hub.train_tags.types[:num_tags]] + \
           [make_avg_probe_fs_vs_cat_context_fig(cat) for cat in model.hub.probe_store.cats[:num_cats]] + \
           [make_avg_probe_fs_vs_terms_context_fig(terms_sorted_by_avg_loc)]
    return figs


def make_phrase_pp_figs(model, field_input):
    def make_phrase_pp_fig(terms):
        """
        Returns fig showing trajectory of Probes Perplexity
        """
        start = time.time()

        # load data
        num_terms = len(terms)
        x = range(num_terms)
        y = model.make_phrase_pps(terms)
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_xticks(x)
        ax.set_xticklabels(terms)
        ax.set_ylabel('Perplexity', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.yaxis.grid(True)
        # plot
        ax.plot(x, y, '-', linewidth=FigsConfigs.LINEWIDTH, color='black')
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_phrase_pp_fig(field_input)]
    return figs


def make_probe_successors_figs(model, field_input, chunk_size=30):
    def make_successor_df(probe):
        probe_id = model.hub.probe_store.probe_id_dict[probe]
        probe_y_mat = model.hub.probe_y_mats[model.hub.probe_store.probe_id_dict[probe]]
        result = pd.DataFrame(index=[probe_id] * len(probe_y_mat), data=probe_y_mat)

        # TODO this used to group by Y to show probe_pp grouped by Y
        # TODO but probe_pps are no longer available - where to get them? recalculate?

        raise NotImplementedError('rnnlab: Not implemented')

        return result

    def make_example_windows_fig(df, probe):
        """
        Returns fig showing pp for successors to "probe"
        """
        start = time.time()

        # fig
        fig, ax = plt.subplots(1, figsize=(FigsConfigs.MAX_FIG_WIDTH, 4))
        ax.set_ylabel('Perplexity', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim([0, model.hub.train_terms.num_types])
        # plot
        print(df)
        df.plot(kind='bar', x='Y0', y='probe_pp', yerr='probe_pp_std', ax=ax, legend=False)
        ax.set_xlabel('Word following "{}"'.format(probe), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = []
    for probe in field_input:
        successor_df = make_successor_df(probe)
        for n, pred_df_chunk in successor_df.groupby(np.arange(len(successor_df)) // chunk_size):
            figs.append(make_example_windows_fig(pred_df_chunk, probe))
    return figs


def make_cat_prediction_figs(model):
    def make_cat_prediction_fig():
        """
        Returns fig showing a measure of show well probes are predicted by category
        """
        start = time.time()
        # load data
        x = np.arange(model.hub.probe_store.num_cats)
        tups = []
        for n, (cat, cat_probes) in enumerate(model.hub.probe_store.cat_probe_list_dict.items()):
            cat_probe_ids = [model.hub.probe_store.probe_id_dict[cat_probe] for cat_probe in cat_probes]
            cat_probes_cat_pred_goodness = np.asarray(model.avg_probe_cat_pred_goodness_list)[cat_probe_ids].mean()
            tups.append((cat, cat_probes_cat_pred_goodness))
        xticklabels, y = zip(*sorted(tups, key=lambda t: t[1]))
        # fig
        fig, ax = plt.subplots(1, figsize=(FigsConfigs.MAX_FIG_WIDTH, 4))
        ax.set_ylabel('Category Prediction Goodness', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xlabel('Category', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=FigsConfigs.TICKLABEL_FONT_SIZE, rotation=90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # plot
        ax.bar(x, y, color='black')
        ax.axhline(y=1.0, linestyle='--', color='grey', lw=FigsConfigs.LINEWIDTH)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_cat_prediction_fig()]
    return figs


def make_abstraction_figs(model):
    def make_probe_abstraction_goodness_vs_cat_size_fig():
        """
        Returns fig showing probe_abstraction_goodness vs probe_cat_size correlation
        """
        start = time.time()

        # load data
        probe_abstraction_goodness_d = {}
        for probe, avg_probe_fs, avg_probe_fs_n in zip(model.hub.probe_store.types,
                                                       model.avg_probe_fs_o_list,
                                                       model.avg_probe_fs_n_list):
            probe_abstraction_goodness_d[probe] = avg_probe_fs_n - avg_probe_fs
        x = [len(model.hub.probe_store.cat_probe_list_dict[model.hub.probe_store.probe_cat_dict[probe]])
             for probe in model.hub.probe_store.types]
        y = [probe_abstraction_goodness_d[probe] for probe in model.hub.probe_store.types]
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_ylabel('Probe Abstraction Goodness', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xlabel('Probe Category Size', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.axhline(y=0, color='grey', zorder=0)
        # plot
        ax.scatter(x=x, y=y, color='black')
        # plot best fit line
        xys = zip(x, y)
        plot_best_fit_line(ax, xys, fontsize=FigsConfigs.LEG_FONTSIZE,  x_pos=0.90, y_pos=0.1, plot_p=True)
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_probe_abstraction_goodness_vs_prop_periods_context_fig():
        """
        Returns fig showing probe_abstraction_goodness vs num_periods correlation
        """
        start = time.time()

        # load data
        probe_abstraction_goodness_d = {}
        for probe, avg_probe_fs, avg_probe_fs_n in zip(model.hub.probe_store.types,
                                                       model.avg_probe_fs_o_list,
                                                       model.avg_probe_fs_n_list):
            probe_abstraction_goodness_d[probe] = avg_probe_fs_n - avg_probe_fs
        x = model.hub.probe_num_periods_in_context_list
        y = [probe_abstraction_goodness_d[probe] for probe in model.hub.probe_store.types]
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_ylabel('Probe Abstraction Goodness', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xlabel('Proportion of Periods in Probe Contexts', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.axhline(y=0, color='grey', zorder=0)
        # plot
        ax.scatter(x=x, y=y, color='black')
        # plot best fit line
        xys = zip(x, y)
        plot_best_fit_line(ax, xys, fontsize=FigsConfigs.LEG_FONTSIZE,  x_pos=0.90, y_pos=0.1, plot_p=True)
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_probe_abstraction_goodness_vs_cat_size_fig(),
            make_probe_abstraction_goodness_vs_prop_periods_context_fig()]
    return figs


def make_context_distance_figs(model):
    def make_compare_term_simmats_fig(num_most_frequent=10,
                                      dists=(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7)):
        """
        Returns fig showing similarity between a computed similarity space and that of the model
        """
        start = time.time()

        most_frequent_terms = model.hub.get_most_frequent_terms(num_most_frequent)
        # load data
        y = []
        for dist in dists:
            term_context_mat = np.zeros((model.hub.train_terms.num_types, num_most_frequent))
            for n, term in enumerate(model.hub.train_terms.types):
                terms_near_term = model.hub.get_terms_near_term(term, dist)
                context_vec = [terms_near_term.count(term) for term in most_frequent_terms]
                term_context_mat[n] = context_vec
            term_context_simmat = cosine_similarity(term_context_mat)
            # calc fit
            fit = paired_cosine_distances(term_context_simmat, model.term_simmat).mean()
            y.append(fit)
        x = np.asarray(dists)
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        plt.title('Terms')
        ax.set_ylabel('Fit', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xlabel('Context Distance', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.axhline(y=0, color='grey', zorder=0)
        ax.set_xticks(dists)
        ax.set_xticklabels(dists)
        # plot
        width = 0.3
        ax.bar(x, y, width, color='black')
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_compare_probe_simmats_fig(num_most_frequent=10,
                                       dists=(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7)):
        """
        Returns fig showing similarity between a computed similarity space and that of the model
        """
        start = time.time()

        most_frequent_terms = model.hub.get_most_frequent_terms(num_most_frequent)
        # load data
        y = []
        for dist in dists:
            probe_context_mat = np.zeros((model.hub.probe_store.num_probes, num_most_frequent))
            for n, probe in enumerate(model.hub.probe_store.types):
                terms_near_probe = model.hub.get_terms_near_term(probe, dist)
                context_vec = [terms_near_probe.count(term) for term in most_frequent_terms]
                probe_context_mat[n] = context_vec
            probe_context_simmat = cosine_similarity(probe_context_mat)
            # calc fit
            fit = paired_cosine_distances(probe_context_simmat, model.probe_simmat_o).mean()
            y.append(fit)
        x = np.asarray(dists)
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        plt.title('Probes')
        ax.set_ylabel('Fit', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xlabel('Context Distance', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.axhline(y=0, color='grey', zorder=0)
        ax.set_xticks(dists)
        ax.set_xticklabels(dists)
        # plot
        width = 0.3
        ax.bar(x, y, width, color='black')
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_compare_term_simmats_fig(),
            make_compare_probe_simmats_fig()]
    return figs


def make_softmax_by_cat_figs(model):
    def make_softmax_by_cat_fig(cat_type):
        """
        Returns fig showing heatmap of average softmax probs for members of syntactic categories
        """
        start = time.time()
        # load data
        if cat_type == 'pos':
            cats = [cat + 's' for cat in sorted(GlobalConfigs.POS_TAGS_DICT.keys())]
        elif cat_type == 'probes':
            cats = model.hub.probe_store.cats
        else:
            raise AttributeError('rnnlab: Invalid arg to "cat_type".')
        softmax_by_cat_mat = model.make_softmax_by_cat_mat(cat_type)
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 5))
        # plot
        sns.heatmap(softmax_by_cat_mat, ax=ax, square=True, annot=False,
                    annot_kws={"size": 6}, cbar_kws={"shrink": .5},
                    vmin=0, vmax=1.0, cmap='jet', fmt='d')
        # colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(['0.0', '0.5', '1.0'])
        cbar.set_label('Sum of Softmax Probabilities')
        # ax (needs to be below plot for axes to be labeled)
        ax.set_xticks(np.arange(len(cats)) + 0.5)
        ax.set_yticks(np.arange(len(cats)) + 0.5)
        ax.set_yticklabels(cats, rotation=0)
        ax.set_xticklabels(cats, rotation=90)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_softmax_by_cat_fig('probes'),
            make_softmax_by_cat_fig('pos')]
    return figs


model_btn_name_figs_fn_dict = {k: globals()['make_{}_figs'.format(k)]
                               for k in AppConfigs.MODEL_BTN_NAME_INFO_DICT.keys()}
