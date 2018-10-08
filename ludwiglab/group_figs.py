import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
from itertools import chain
from scipy import stats
from itertools import cycle
import copy
import time
import numpy as np
import os
from operator import itemgetter
import sys
import pandas as pd

from figutils import human_format
from figutils import plot_best_fit_line
from figutils import make_cats_sorted_by_ba_diff
from figutils import to_trajs
from figutils import to_traj
from figutils import add_double_legend
from figutils import add_single_legend
from configs import GlobalConfigs, FigsConfigs, AppConfigs

rcParams['figure.max_open_warning'] = 0


def make_probe_term_sims_figs(model_groups, model_descs):
    def make_term_sim_hist_fig(num_bins=1000):
        """
        Returns fig showing histogram of group averages of similarities
        """
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        # load data
        xys = []
        for models, model_desc in zip(model_groups, model_descs):
            term_sims_list = []
            for model in models:
                term_simmat = model.term_simmat[:]
                term_simmat[np.tril_indices(term_simmat.shape[0], -1)] = np.nan
                term_sims = term_simmat[~np.isnan(term_simmat)]
                term_sims_list.append(term_sims)
            avg_term_sims = np.mean(np.asarray(term_sims_list), axis=0)
            step_size = 1.0 / num_bins
            bins = np.arange(-1, 1, step_size)
            y, _ = np.histogram(avg_term_sims, bins=bins)
            x = bins[:-1]
            xys.append((x, y, model_desc))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4))
        # axis 1
        ax.set_xlabel('Avg Term Similarity', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('Frequency', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xlim([-1, 1])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.xaxis.grid(True)
        # plot sim hist
        for (x, y, model_desc) in xys:
            ax.plot(x, y, '-', linewidth=FigsConfigs.LINEWIDTH, c=next(palette), label=model_desc)
        plt.tight_layout()
        add_single_legend(ax, model_descs)  # TODO do this in other places too
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_window_sim_traj_fig(window_sizes=(1, 1, 2), pos=None,
                                 context_is_uniform=False, randomized_positions=(1, 1, 2)):
        """
        Returns fig showing trajectories of similarity between hidden states for windwos of varying size
        """
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        assert len(window_sizes) == 3
        assert len(randomized_positions) == 3
        # load data
        xys = []
        x = model_groups[0][0].get_data_step_axis()
        for models, model_desc in zip(model_groups, model_descs):
            trajs1_list = []
            trajs2_list = []
            trajs3_list = []
            for model in models:
                traj1, traj2, traj3 = to_trajs(model, model.calc_windows_sim,
                                               randomized_positions, pos, context_is_uniform)
                trajs1_list.append(traj1)
                trajs2_list.append(traj2)
                trajs3_list.append(traj3)
            traj_mat1 = np.asarray([traj[:len(x)] for traj in trajs1_list])
            traj_mat2 = np.asarray([traj[:len(x)] for traj in trajs2_list])
            traj_mat3 = np.asarray([traj[:len(x)] for traj in trajs3_list])
            y1 = np.mean(traj_mat1, axis=0)
            y2 = np.mean(traj_mat2, axis=0)
            y3 = np.mean(traj_mat3, axis=0)
            sem1 = [stats.sem(row) for row in np.asarray(traj_mat1).T]
            xys.append((x, y1, y2, y3, sem1, model_desc))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        title = 'random windows' if pos is None else pos
        if context_is_uniform:
            title += ' (context is repetition of last term in window)'
        plt.title(title)
        ax.set_ylabel('Similarity (+/-SEM)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        ax.set_ylim([0, 1])  # TODO improve
        # plot
        lines = []
        for n, (x, y1, y2, y3, sem1, model_desc) in enumerate(xys):
            color = next(palette)
            l1, = ax.plot(x, y1, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-')
            # l2, = ax.plot(x, y2, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='--')
            # l3, = ax.plot(x, y3, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle=':')
            l2 = l1
            l3 = l1  # TODO remove
            lines.append((copy.copy(l1), copy.copy(l2), copy.copy(l3)))
            ax.fill_between(x, np.add(y1, sem1), np.subtract(y1, sem1), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        # labels = ['context-size={}'.format(size) for size in window_sizes]
        # labels = ['randomized last\n{} context positions'.format(p - 1) for p in randomized_positions]
        labels = []  # TODO remove
        add_double_legend(ax, lines, labels, model_descs)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_window_sim_traj_fig(pos=None)]
    # figs += [make_window_sim_traj_fig(pos=pos) for pos in sorted(GlobalConfigs.POS_TAGS_DICT.keys())]
    figs += [make_term_sim_hist_fig()]
    return figs


def make_compare_to_sg_figs(model_groups, model_descs):
    def make_sim_simmat_fig(include_sg=False, vmin=0.9, config_id=1):
        """
        Returns fig showing similarity matrix of probe similarity matrices of multiple models
        """
        start = time.time()
        # init term_simmat_mat
        if include_sg:
            sg_types = list(np.load(os.path.join(GlobalConfigs.SG_DIR, 'sg_types.npy')).tolist())
            terms = list(set(model_groups[0][0].hub.train_terms.types) & set(sg_types))
            dim1 = model_groups[0][0].hub.probe_store.num_probes * len(terms)
        else:
            sg_types = None
            terms = model_groups[0][0].hub.train_terms.types
            dim1 = model_groups[0][0].hub.probe_store.num_probes * len(terms)
        num_models = len([model for models in model_groups for model in models])
        term_simmat_mat = np.zeros((num_models, dim1))
        # load data
        row_ids = iter(range(num_models))
        probe_term_ids = [model_groups[0][0].hub.train_terms.types.index(probe_)
                          for probe_ in model_groups[0][0].hub.probe_store.types]
        term_term_ids = [model_groups[0][0].hub.train_terms.types.index(term)
                         for term in terms]
        group_names = []
        for model_desc, models in zip(model_descs, model_groups):
            for model in models:
                group_name = model_desc.split('\n')[config_id].split('=')[-1]
                group_names.append(group_name)
                probe_term_simmat = model.term_simmat[probe_term_ids, :]
                flattened = probe_term_simmat[:, term_term_ids, ].flatten()
                term_simmat_mat[next(row_ids), :] = flattened
        # sim_simmat
        if include_sg:  # TODO fix
            sg_probe_term_ids = [sg_types.index(probe) for probe in model_groups[0][0].hub.probe_store.types]
            sg_term_term_ids = [sg_types.index(term) for term in terms]
            sg_token_simmat_filenames = sorted([f for f in os.listdir(GlobalConfigs.SG_DIR)
                                                if f.startswith('sg_token_simmat')])
            num_sgs = len(sg_token_simmat_filenames)
            sg_term_simmats_mat = np.zeros((num_sgs, dim1))
            for model_id, f in enumerate(sg_token_simmat_filenames):
                sg_probe_term_simmat = np.load(os.path.join(GlobalConfigs.SG_DIR, f))[sg_probe_term_ids, :]
                flattened = sg_probe_term_simmat[:, sg_term_term_ids].flatten()
                sg_term_simmats_mat[model_id, :] = flattened
            group_names += ['skip-gram'] * num_sgs
            term_simmat_mat = np.vstack((term_simmat_mat, sg_term_simmats_mat))
        else:
            term_simmat_mat = term_simmat_mat
        sim_simmat = pd.DataFrame(term_simmat_mat).T.corr().values
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, FigsConfigs.MAX_FIG_WIDTH))
        mask = np.zeros_like(sim_simmat, dtype=np.bool)
        mask[np.triu_indices_from(mask, 1)] = True
        sns.heatmap(sim_simmat, ax=ax, square=True, annot=False,
                    annot_kws={"size": 5}, cbar_kws={"shrink": .5},
                    vmin=vmin, vmax=1.0, cmap='jet')  # , mask=mask
        # colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([vmin, 1.0])
        cbar.set_ticklabels([str(vmin), '1.0'])
        cbar.set_label('Similarity between Semantic Spaces')
        # ax (needs to be below plot for axes to be labeled)
        num_group_names = len(group_names)
        ax.set_yticks(np.arange(num_group_names) + 0.5)
        ax.set_xticks(np.arange(num_group_names) + 0.5)
        ax.set_yticklabels(group_names, rotation=0)
        ax.set_xticklabels(group_names, rotation=90)
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        # export to csv for stats  # TODO put this somewhere else
        if AppConfigs.SAVE_SIM_SIMMAT:
            col1 = ['0'] * num_models1 * num_models + ['1'] * num_models2 * num_models
            col2 = (['0'] * num_models1 + ['1'] * num_models2) * num_models
            sim_simmat[np.tril_indices(sim_simmat.shape[0], 0)] = np.nan
            col3 = sim_simmat.flatten()
            sim_simmat_df = pd.DataFrame(data={'model1': col1,
                                               'model2': col2,
                                               'sim': col3})
            sim_simmat_df = sim_simmat_df.dropna()
            sim_simmat_df.to_csv(os.path.join(os.path.expanduser("~"), 'sim_simmat_df.csv'), index=False)
            print('Saved sim_simmat_df.csv to {}'.format(dir))
        return fig

    figs = [make_sim_simmat_fig()]
    return figs


def make_ba_by_cat_figs(model_groups, model_descs):
    def sort_cats_by_group1_ba(context_type):
        cats = model_groups[0][0].hub.probe_store.cats
        num_cats = len(cats)
        mat = np.zeros((len(model_groups[0]), num_cats))
        for n, model in enumerate(model_groups[0]):
            mat[n, :] = [np.mean(cat_avg_probe_ba_list) for cat_avg_probe_ba_list
                         in model.get_sorted_cat_avg_probe_ba_lists(model.hub.probe_store.cats, context_type)]
        avgs = mat.mean(axis=0).tolist()
        _, tup = zip(*sorted(zip(avgs, cats), key=lambda t: t[0]))
        result = list(tup)
        return result

    def make_cat_ba_mat_fig(context_type, sg_embed_size=512):
        """
        Returns fig showing heatmap of F1-scores from multiple models broken down by category
        """
        start = time.time()
        num_models = len([model for models in model_groups for model in models])
        cats = model_groups[0][0].hub.probe_store.cats
        hub_mode = model_groups[0][0].hub.mode
        num_cats = len(cats)
        # load data
        sorted_cats = sort_cats_by_group1_ba(context_type)
        group_names = []
        cat_ba_mat = np.zeros((num_models, num_cats))
        row_ids = iter(range(num_models))
        for model_desc, models in zip(model_descs, model_groups):
            for model_id, model in enumerate(models):
                group_name = model_desc.replace('\n', ' ').split('=')[-1]
                group_names.append(group_name)
                sorted_cat_avg_probe_ba_lists = model.get_sorted_cat_avg_probe_ba_lists(sorted_cats, context_type)
                cat_ba_mat[next(row_ids), :] = [np.mean(cat_avg_probe_ba_list)
                                                for cat_avg_probe_ba_list in sorted_cat_avg_probe_ba_lists]
        # load sg data
        path = GlobalConfigs.SG_DIR / 'sg_df_{}_{}.csv'.format(hub_mode, sg_embed_size)
        if path.exists():
            df_sg = pd.read_csv(path)
            sg_cat_ba_mat = df_sg.groupby('cat').mean().transpose()[sorted_cats].values
            num_sgs = len(sg_cat_ba_mat)
            group_names += ['skip-gram'] * num_sgs
            cat_ba_mat = np.vstack((cat_ba_mat, sg_cat_ba_mat))
        else:
            num_sgs = 0
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        plt.title('context_type="{}"'.format(context_type), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # plot
        sns.heatmap(cat_ba_mat, ax=ax, square=True, annot=False,
                    annot_kws={"size": 6}, cbar_kws={"shrink": .5},
                    vmin=0, vmax=1, cmap='jet', fmt='d')
        # colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(['0', '0.5', '1'])
        cbar.set_label('Avg Category Probe Balanced Accuracy')
        # ax (needs to be below plot for axes to be labeled)
        ax.set_yticks(np.arange(num_models + num_sgs) + 0.5)
        ax.set_xticks(np.arange(num_cats) + 0.5)
        ax.set_yticklabels(group_names, rotation=0)
        ax.set_xticklabels(sorted_cats, rotation=90)
        for t in ax.texts:
            t.set_text(t.get_text() + "%")
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        # export to csv for stats
        if AppConfigs.SAVE_CAT_FS_MAT:
            cat_ba_df = pd.DataFrame(cat_ba_mat, columns=sorted_cats)
            dir = os.path.expanduser("~")
            cat_ba_df.to_csv(os.path.join(dir, 'cat_ba_df.csv'), index=False)
            print('Saved cat_ba_df.csv to {}'.format(dir))
        return fig

    def make_ba_by_cat_fig(context_type, sg_embed_size=512):
        """
        Returns fig showing model group averages of probes balanced accuracy broken down by category
        """
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        cats = model_groups[0][-1].hub.probe_store.cats
        hub_mode = model_groups[0][-1].hub.mode
        num_cats = len(cats)
        # load data
        sorted_cats = sort_cats_by_group1_ba(context_type)
        xys = []
        for models, model_desc in zip(model_groups, model_descs):
            cat_ba_mat = np.zeros((len(models), num_cats))
            for model_id, model in enumerate(models):
                sorted_cat_avg_probe_ba_lists = model.get_sorted_cat_avg_probe_ba_lists(sorted_cats, context_type)
                cat_ba_mat[model_id, :] = [np.mean(cat_avg_probe_ba_list)
                                           for cat_avg_probe_ba_list in sorted_cat_avg_probe_ba_lists]
            x = range(num_cats)
            y = np.mean(cat_ba_mat, axis=0)
            sem = stats.sem(cat_ba_mat, axis=0)
            num_models = len(models)
            xys.append((x, y, sem, model_desc, num_models))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        plt.title('context_type="{}"'.format(context_type), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('Balanced Accuracy (+/-SEM)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE, labelpad=0.0)
        ax.set_xticks(np.arange(num_cats), minor=False)
        ax.set_xticklabels(sorted_cats, minor=False, fontsize=FigsConfigs.TICKLABEL_FONT_SIZE, rotation=90)
        ax.set_xlim([0, len(cats)])
        ax.set_ylim([0.5, 1.0])
        ax.set_axisbelow(True)  # put grid under plot lines
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # plot
        for (x, y, sem, model_desc, num_models) in xys:
            color = next(palette)
            ax.plot(x, y, '-', color=color, linewidth=FigsConfigs.LINEWIDTH,
                    label='{} n={}'.format(model_desc, num_models))
            ax.fill_between(x, np.add(y, sem), np.subtract(y, sem), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        # plot sg
        path = GlobalConfigs.SG_DIR / 'sg_df_{}_{}.csv'.format(hub_mode, sg_embed_size)
        if path.exists():
            df_sg = pd.read_csv(path)
            df_sg['avg_probe_ba_mean'] = df_sg.filter(regex="avg_probe_ba_\d").mean(axis=1)
            df_sg['avg_probe_ba_sem'] = df_sg.filter(regex="avg_probe_ba_\d").sem(axis=1)
            num_sgs = len(df_sg.filter(regex="avg_probe_ba_\d").columns)
            cat_y_dict = df_sg.groupby('cat').mean().to_dict()['avg_probe_ba_mean']
            cat_sem_dict = df_sg.groupby('cat').mean().to_dict()['avg_probe_ba_sem']
            y = [cat_y_dict[cat] for cat in sorted_cats]
            sem = [cat_sem_dict[cat] for cat in sorted_cats]
            x = range(num_cats)
            ax.plot(x, y, '-', color='black', linewidth=FigsConfigs.LINEWIDTH,
                    label='skipgram num_h{} n={}'.format(sg_embed_size, num_sgs))
            ax.fill_between(x, np.add(y, sem), np.subtract(y, sem), alpha=FigsConfigs.FILL_ALPHA, color='grey')
            sg_avg_probe_bas = df_sg.filter(regex="avg_probe_ba_\d").mean(axis=0)
            print('Skip-gram avg_probe_ba mean across models:', sg_avg_probe_bas.mean())
            print('Skip-gram avg_probe_ba sem across models:', sg_avg_probe_bas.sem())
        plt.tight_layout()
        add_single_legend(ax, model_descs, y_offset=-0.60)  # TODO do this in other places too
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_cat_ba_mat_fig('ordered'),
            make_ba_by_cat_fig('ordered')]
    return figs


def make_perplexity_figs(model_groups, model_descs):
    def make_test_and_train_pp_traj_fig(zoom=8):
        """
        Returns fig showing trajectory of test and train perplexity
        """
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        # load data
        xys = []
        y2mins = []
        for models in model_groups:
            model_test_pp_trajs = []
            model_train_pp_trajs = []
            for model in models:
                model_test_pp_trajs.append(model.get_traj('test_pp'))
                model_train_pp_trajs.append(model.get_traj('train_pp'))
            x = models[0].get_data_step_axis()
            traj_mat1 = np.asarray([np.log(traj)[:len(x)] for traj in model_test_pp_trajs])
            traj_mat2 = np.asarray([np.log(traj)[:len(x)] for traj in model_train_pp_trajs])
            y1 = np.mean(traj_mat1, axis=0)
            y2 = np.mean(traj_mat2, axis=0)
            sem1 = [stats.sem(row) for row in np.asarray(traj_mat1).T]
            sem2 = [stats.sem(row) for row in np.asarray(traj_mat2).T]
            y2mins.append(np.min(y2))
            xys.append((x, y1, y2, sem1, sem2))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        ax.set_ylabel('Cross-Entropy Error (+/-SEM)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        # ax.set_xticklabels(x, fontsize=FigsConfigs.TICKLABEL_FONT_SIZE)  # TODO set tick fontsize everywhere - but don't set x when x may be different between model group
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        ax.set_ylim([np.min(y2mins) - np.std(y2mins), np.min(y2mins) * (1 + 1/zoom)])
        # plot
        lines = []
        labels = ['test', 'train']
        for (x, y1, y2, sem1, sem2) in xys:
            color = next(palette)
            l1, = ax.plot(x, y1, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='--')
            l2, = ax.plot(x, y2, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-')
            lines.append((copy.copy(l1), copy.copy(l2)))
            ax.fill_between(x, np.add(y1, sem1), np.subtract(y1, sem1), alpha=FigsConfigs.FILL_ALPHA, color='grey')
            ax.fill_between(x, np.add(y2, sem2), np.subtract(y2, sem2), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        add_double_legend(ax, lines, labels, model_descs)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_probes_pp_traj_fig():
        """
        Returns fig showing trajectory of Probes Perplexity
        """
        start = time.time()

        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        # load data
        xys = []
        for models in model_groups:
            probes_pp_trajs_w = []
            for nn, model in enumerate(models):
                probes_pp_trajs_w.append(model.get_traj('probes_pp'))
            x = models[0].get_data_step_axis()
            traj_mat = np.asarray([traj[:len(x)] for traj in probes_pp_trajs_w])
            y = np.mean(traj_mat, axis=0)
            sem = [stats.sem(row) for row in np.asarray(traj_mat).T]
            xys.append((x, y, sem))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 5))
        plt.title(model_groups[0][0].hub.mode)
        ylabel = 'Probes Perplexity'
        ax.set_ylabel(ylabel, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        ax.set_ylim([0, 10000])  # TODO remove
        # plot
        lines = []
        # labels = ['weighted?']
        for (x, y, sem) in xys:
            color = next(palette)
            l1, = ax.plot(x, y, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-')
            lines.append((copy.copy(l1)))
            ax.fill_between(x, np.add(y, sem), np.subtract(y, sem), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        # add_double_legend(ax, lines, labels, model_descs)  # TODO for plotting different kinds of pp trajs
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_test_and_train_pp_traj_fig(),
            make_probes_pp_traj_fig()]
    return figs


def make_w_coherence_figs(model_groups, model_descs):

    def make_avg_w_coherence_traj_fig(cat_type):
        """
        Returns fig showing traj of wy coherence for "cat_type"
        """
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        # load data
        if cat_type == 'pos':
            term_ids_list = [[model_groups[0][0].hub.train_terms.term_id_dict[term]
                              for term in getattr(model_groups[0][0].hub, pos + 's')]
                             for pos in sorted(GlobalConfigs.POS_TAGS_DICT.keys())]
        elif cat_type == 'probes':
            term_ids_list = [[model_groups[0][0].hub.train_terms.term_id_dict[term]
                              for term in model_groups[0][0].hub.probe_store.cat_probe_list_dict[cat]]
                             for cat in model_groups[0][0].hub.probe_store.cats]
        else:
            raise AttributeError('rnnlab: Invalid arg to "cat_type"')
        xys = []
        for models, model_desc in zip(model_groups, model_descs):
            trajs_list1 = []
            trajs_list2 = []
            for model in models:
                traj1 = np.asarray(to_trajs(model, model.calc_w_sim, term_ids_list, 'wx')).mean(axis=0)
                traj2 = np.asarray(to_trajs(model, model.calc_w_sim, term_ids_list, 'wy')).mean(axis=0)
                trajs_list1.append(traj1)
                trajs_list2.append(traj2)
            x = models[0].get_data_step_axis()
            traj_mat1 = np.asarray([traj[:len(x)] for traj in trajs_list1])
            traj_mat2 = np.asarray([traj[:len(x)] for traj in trajs_list2])
            y1 = np.mean(traj_mat1, axis=0)
            y2 = np.mean(traj_mat2, axis=0)
            sem1 = [stats.sem(row) for row in np.asarray(traj_mat1).T]
            sem2 = [stats.sem(row) for row in np.asarray(traj_mat2).T]
            xys.append((x, y1, y2, sem1, sem2, model_desc))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        plt.title(cat_type)
        ax.set_ylabel('Weights Avg Coherence (+/-SEM)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        # plot
        lines = []
        for n, (x, y1, y2, sem1, sem2, model_desc) in enumerate(xys):
            color = next(palette)
            l1, = ax.plot(x, y1, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-')
            l2, = ax.plot(x, y2, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle=':')
            lines.append((copy.copy(l1), copy.copy(l2)))
            ax.fill_between(x, np.add(y1, sem1), np.subtract(y1, sem1), alpha=FigsConfigs.FILL_ALPHA, color='grey')
            ax.fill_between(x, np.add(y2, sem2), np.subtract(y2, sem2), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        labels = ['wx', 'wy']
        add_double_legend(ax, lines, labels, model_descs)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_w_coherence_traj_fig(pos):
        """
        Returns fig showing similarity of columns in wy corresponding to members of same POS
        """
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        term_ids = [model_groups[0][0].hub.train_terms.term_id_dict[term]
                    for term in getattr(model_groups[0][0].hub, pos + 's')]
        # load data
        xys = []
        for models, model_desc in zip(model_groups, model_descs):
            trajs1_list = []
            trajs2_list = []
            for model in models:
                traj1 = to_traj(model, model.calc_w_sim, term_ids, 'wx')
                traj2 = to_traj(model, model.calc_w_sim, term_ids, 'wy')
                trajs1_list.append(traj1)
                trajs2_list.append(traj2)
            x = models[0].get_data_step_axis()
            traj_mat1 = np.asarray([traj[:len(x)] for traj in trajs1_list])
            traj_mat2 = np.asarray([traj[:len(x)] for traj in trajs2_list])
            y1 = np.mean(traj_mat1, axis=0)
            y2 = np.mean(traj_mat2, axis=0)
            sem1 = [stats.sem(row) for row in np.asarray(traj_mat1).T]
            xys.append((x, y1, y2, sem1, model_desc))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        plt.title(pos)
        ax.set_ylabel('Weights Coherence (+/-SEM)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        # plot
        lines = []
        for n, (x, y1, y2, sem1, model_desc) in enumerate(xys):
            color = next(palette)
            l1, = ax.plot(x, y1, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-')
            l2, = ax.plot(x, y2, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle=':')
            lines.append((copy.copy(l1), copy.copy(l2)))
            ax.fill_between(x, np.add(y1, sem1), np.subtract(y1, sem1), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        labels = ['wx', 'wy']
        add_double_legend(ax, lines, labels, model_descs)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_avg_w_coherence_traj_fig(cat_type) for cat_type in ['pos', 'probes']]
    figs += [make_w_coherence_traj_fig(pos) for pos in sorted(GlobalConfigs.POS_TAGS_DICT.keys())]
    return figs


def make_probes_ba_figs(model_groups, model_descs):
    def make_probes_fs_fig():
        """
        Returns fig showing probes_fs trajs for three different context_types
        """
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        # load data
        xys = []
        for models in model_groups:
            trajs1 = []
            trajs2 = []
            trajs3 = []
            for model in models:
                trajs1.append(model.get_traj('probes_fs_p_o'))  # precision associated with f-score rather than ba
                trajs2.append(model.get_traj('probes_fs_r_o'))
                trajs3.append(model.get_traj('probes_fs_o'))
            x = models[0].get_data_step_axis()
            traj_mat1 = np.asarray([traj[:len(x)] for traj in trajs1])
            traj_mat2 = np.asarray([traj[:len(x)] for traj in trajs2])
            traj_mat3 = np.asarray([traj[:len(x)] for traj in trajs3])
            y1 = np.mean(traj_mat1, axis=0)
            y2 = np.mean(traj_mat2, axis=0)
            y3 = np.mean(traj_mat3, axis=0)
            sem1 = [stats.sem(row) for row in np.asarray(traj_mat1).T]
            sem2 = [stats.sem(row) for row in np.asarray(traj_mat2).T]
            sem3 = [stats.sem(row) for row in np.asarray(traj_mat3).T]
            xys.append((x, y1, y2, y3, sem1, sem2, sem3))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        plt.title(model_groups[0][0].hub.mode)
        ax.set_ylabel('{} (+/-SEM)'.format('F1-score'),
                      fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        ax.set_ylim([0.5, 0.8])  # TODO improve
        # plot
        lines = []
        for n, (x, y1, y2, y3, sem1, sem2, sem3) in enumerate(xys):
            color = next(palette)
            # lines
            l1, = ax.plot(x, y1, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='--')
            l2, = ax.plot(x, y2, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle=':')
            l3, = ax.plot(x, y3, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-')
            lines.append((copy.copy(l1), copy.copy(l2), copy.copy(l3)))
            # sem
            ax.fill_between(x, np.add(y1, sem1), np.subtract(y1, sem1), alpha=FigsConfigs.FILL_ALPHA, color='grey')
            ax.fill_between(x, np.add(y2, sem2), np.subtract(y2, sem2), alpha=FigsConfigs.FILL_ALPHA, color='grey')
            ax.fill_between(x, np.add(y3, sem3), np.subtract(y3, sem3), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        labels = ['precision', 'recall', 'F1-score']
        add_double_legend(ax, lines, labels, model_descs)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_probes_ba_fig():
        """
        Returns fig showing probes_ba trajs for three different context_types
        """
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        # load data
        xys = []
        for models in model_groups:
            trajs1 = []
            trajs2 = []
            trajs3 = []
            for model in models:
                trajs1.append(model.get_traj('probes_ba_p_o'))
                trajs2.append(model.get_traj('probes_ba_r_o'))
                trajs3.append(model.get_traj('probes_ba_o'))
            x = models[0].get_data_step_axis()
            traj_mat1 = np.asarray([traj[:len(x)] for traj in trajs1])
            traj_mat2 = np.asarray([traj[:len(x)] for traj in trajs2])
            traj_mat3 = np.asarray([traj[:len(x)] for traj in trajs3])
            y1 = np.mean(traj_mat1, axis=0)
            y2 = np.mean(traj_mat2, axis=0)
            y3 = np.mean(traj_mat3, axis=0)
            sem1 = [stats.sem(row) for row in np.asarray(traj_mat1).T]
            sem2 = [stats.sem(row) for row in np.asarray(traj_mat2).T]
            sem3 = [stats.sem(row) for row in np.asarray(traj_mat3).T]
            xys.append((x, y1, y2, y3, sem1, sem2, sem3))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6), dpi=150)
        if model_groups[0][0].hub.mode == 'sem':
            plt.title('Semantic Categorization', fontsize=16)
        else:
            plt.title('Syntactic Categorization', fontsize=16)
        ax.set_ylabel('{} (+/-SEM)'.format('Balanced Accuracy'),
                      fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        ax.set_ylim([0.5, 0.75])  # TODO improve
        # plot
        ax.axvline(x=model_groups[-1][0].get_data_step_axis()[-1] / 2,
                   color='grey', zorder=1)
        lines = []
        for n, (x, y1, y2, y3, sem1, sem2, sem3) in enumerate(xys):
            color = next(palette)
            # lines
            # l1, = ax.plot(x, y1, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='--')
            # l2, = ax.plot(x, y2, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle=':')
            l3, = ax.plot(x, y3, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-')
            l1 = l3  # TODO remove
            l2 = l3
            lines.append((copy.copy(l1), copy.copy(l2), copy.copy(l3)))
            # sem
            # ax.fill_between(x, np.add(y1, sem1), np.subtract(y1, sem1), alpha=FigsConfigs.FILL_ALPHA, color='grey')
            # ax.fill_between(x, np.add(y2, sem2), np.subtract(y2, sem2), alpha=FigsConfigs.FILL_ALPHA, color='grey')
            ax.fill_between(x, np.add(y3, sem3), np.subtract(y3, sem3), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        labels = ['precision', 'recall', 'balanced accuracy']
        # labels = ['balanced accuracy']
        add_double_legend(ax, lines, labels, model_descs)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def calc_split_probes_ba_traj(model, probes, context_type):
        avg_probe_p_list = model.ba_traj_df[[column_name for column_name in model.ba_traj_df
                                             if column_name.endswith('p_' + context_type[0]) and
                                             column_name[:-4] in probes]].mean(axis=1)
        avg_probe_r_list = model.ba_traj_df[[column_name for column_name in model.ba_traj_df
                                             if column_name.endswith('r_' + context_type[0]) and
                                             column_name[:-4] in probes]].mean(axis=1)
        result = (avg_probe_p_list + avg_probe_r_list) / 2
        return result

    def make_probes_ba_by_loc_fig(context_type, r_shift='@'):
        """
        Returns fig showing probes_ba trajs median-split by avg location of probes
        """
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        probes1, probes2 = model_groups[0][0].hub.split_probes_by_loc(2)
        # load data
        xys = []
        for models in model_groups:
            trajs1 = []
            trajs2 = []
            for model in models:
                trajs1.append(calc_split_probes_ba_traj(model, probes1, context_type))
                trajs2.append(calc_split_probes_ba_traj(model, probes2, context_type))
            x = models[0].get_data_step_axis()
            traj_mat1 = np.asarray([traj[:len(x)] for traj in trajs1])
            traj_mat2 = np.asarray([traj[:len(x)] for traj in trajs2])
            y1 = np.mean(traj_mat1, axis=0)
            y2 = np.mean(traj_mat2, axis=0)
            sem1 = [stats.sem(row) for row in np.asarray(traj_mat1).T]
            sem2 = [stats.sem(row) for row in np.asarray(traj_mat2).T]
            xys.append((x, y1, y2, sem1, sem2))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        plt.title(model_groups[0][0].hub.mode)
        ax.set_ylabel('"{}" Balanced Accuracy (+/-SEM)'.format(context_type), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        ax.set_ylim([0.5, 0.8])  # TODO improve
        # plot
        ax.axvline(x=model_groups[-1][0].get_data_step_axis()[-1] / 2,
                   color='grey', zorder=1)
        lines = []
        for n, (x, y1, y2, sem1, sem2) in enumerate(xys):
            # right-shift trajectories to compare trainong on same partition (use only when num_parts=2
            if r_shift in model_descs[n] and model_groups[0][0].hub.num_parts == 2:
                mid_timepoint = model_groups[0][0].hub.num_saves // 2
                print(model_groups[0][0].hub.num_saves)
                print(mid_timepoint)

                x = model_groups[0][0].hub.data_mbs

                # TODO because data_mb generation is misisng zero
                # x = np.asarray(x) / 10

                y1 = np.concatenate((np.full(mid_timepoint, np.nan), y1[:mid_timepoint + 1]))
                y2 = np.concatenate((np.full(mid_timepoint, np.nan), y2[:mid_timepoint + 1]))
                sem1 = np.concatenate((np.full(mid_timepoint, np.nan), sem1[:mid_timepoint + 1]))
                sem2 = np.concatenate((np.full(mid_timepoint, np.nan), sem2[:mid_timepoint + 1]))
            color = next(palette)
            # lines
            l1, = ax.plot(x, y1, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-')
            l2, = ax.plot(x, y2, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='--')
            lines.append((copy.copy(l1), copy.copy(l2)))
            # sem
            ax.fill_between(x, np.add(y1, sem1), np.subtract(y1, sem1), alpha=FigsConfigs.FILL_ALPHA, color='grey')
            ax.fill_between(x, np.add(y2, sem2), np.subtract(y2, sem2), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        labels = ['earliest half', 'latest half']
        add_double_legend(ax, lines, labels, model_descs)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_probes_ba_fig()]
    figs += [make_probes_ba_by_loc_fig(context_type) for context_type in ['ordered']]
    figs += [make_probes_ba_by_loc_fig(context_type, r_shift='dec_age') for context_type in ['ordered']]
    figs += [make_probes_ba_by_loc_fig(context_type, r_shift='inc_age') for context_type in ['ordered']]
    return figs


def make_w_norm_figs(model_groups, model_descs):
    def make_norms_vecs_list_list(w_name):
        result = []
        for models in model_groups:
            softmax_vecs_list = []
            for model in models:
                softmax_vecs = to_traj(model, model.calc_norm_of_w, w_name)
                softmax_vecs_list.append(softmax_vecs)
            result.append(softmax_vecs_list)
        return result

    def calc_avg_norm(norms_vec, term_ids):
        """
        Calc avg norm of all terms with term_id in "term_ids"
        """
        filtered_probs = norms_vec[term_ids]
        result = filtered_probs.mean()
        return result

    def make_w_norm_traj_fig(pos):
        """
        Returns fig showing trajectories of norm of weights by grammatical category
        """
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        term_ids1 = [model_groups[0][0].hub.train_terms.term_id_dict[term]
                     for term in getattr(model_groups[0][0].hub, pos + 's')]
        # load data
        xys = []
        x = model_groups[0][0].get_data_step_axis()
        for softmax_vecs1_list, softmax_vecs2_list, model_desc in zip(
                norms_vecs1_list_list, norms_vecs2_list_list, model_descs):
            trajs1_list = []
            trajs2_list = []
            for vecs1, vecs2 in zip(softmax_vecs1_list, softmax_vecs2_list):
                trajs1_list.append([calc_avg_norm(v, term_ids1) for v in vecs1])  # words as input
                trajs2_list.append([calc_avg_norm(v, term_ids1) for v in vecs2])  # early windows as input
            traj_mat1 = np.asarray([traj[:len(x)] for traj in trajs1_list])
            traj_mat2 = np.asarray([traj[:len(x)] for traj in trajs2_list])
            y1 = np.mean(traj_mat1, axis=0)
            y2 = np.mean(traj_mat2, axis=0)
            sem1 = [stats.sem(row) for row in np.asarray(traj_mat1).T]
            sem2 = [stats.sem(row) for row in np.asarray(traj_mat2).T]
            xys.append((x, y1, y2, sem1, sem2, model_desc))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        plt.title(pos)
        ax.set_ylabel('Norm of W (+/-SEM)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        # ax.set_ylim([0, 1.5])  # TODO improve
        # plot
        lines = []
        for n, (x, y1, y2, sem1, sem2, model_desc) in enumerate(xys):
            color = next(palette)
            l1, = ax.plot(x, y1, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-')
            l2, = ax.plot(x, y2, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='--')
            lines.append((copy.copy(l1), copy.copy(l2)))
            ax.fill_between(x, np.add(y1, sem1), np.subtract(y1, sem1), alpha=FigsConfigs.FILL_ALPHA, color='grey')
            ax.fill_between(x, np.add(y2, sem2), np.subtract(y2, sem2), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        labels = ['input weights', 'output weights']
        add_double_legend(ax, lines, labels, model_descs)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    norms_vecs1_list_list = make_norms_vecs_list_list('wx')
    norms_vecs2_list_list = make_norms_vecs_list_list('wy')
    figs = [make_w_norm_traj_fig(pos) for pos in sorted(GlobalConfigs.POS_TAGS_DICT.keys())]
    return figs


def make_softmax_figs(model_groups, model_descs, window_sizes=(1, 2, 7)):
    def make_softmax_vecs_list_list(window_size):  # compute softmax vecs once only for each model
        result = []
        for models in model_groups:
            softmax_vecs_list = []
            for model in models:
                softmax_vecs = to_traj(model, model.calc_softmax_vec, window_size)
                softmax_vecs_list.append(softmax_vecs)
            result.append(softmax_vecs_list)
        return result

    def calc_avg_softmax_prob(softmax_vec, term_ids):
        """
        Calc avg softmax probs of all terms with term_id in "term_ids"
        """
        filtered_probs = softmax_vec[term_ids]
        result = filtered_probs.mean()
        return result

    def make_softmax_traj_fig(pos):
        """
        Returns fig showing trajectories of softmax activations by grammatical category
        """
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        term_ids1 = [model_groups[0][0].hub.train_terms.term_id_dict[term]
                     for term in getattr(model_groups[0][0].hub, pos + 's')]
        # load data
        xys = []
        x = model_groups[0][0].get_data_step_axis()
        for softmax_vecs1_list, softmax_vecs2_list, softmax_vecs3_list, model_desc in zip(
                softmax_vecs1_list_list, softmax_vecs2_list_list, softmax_vecs3_list_list, model_descs):
            trajs1_list = []
            trajs2_list = []
            trajs3_list = []
            for vecs1, vecs2, vecs3 in zip(softmax_vecs1_list, softmax_vecs2_list, softmax_vecs3_list):
                trajs1_list.append([calc_avg_softmax_prob(v, term_ids1) for v in vecs1])  # words as input
                trajs2_list.append([calc_avg_softmax_prob(v, term_ids1) for v in vecs2])  # early windows as input
                trajs3_list.append([calc_avg_softmax_prob(v, term_ids1) for v in vecs3])  # late windows as input
            traj_mat1 = np.asarray([traj[:len(x)] for traj in trajs1_list])
            traj_mat2 = np.asarray([traj[:len(x)] for traj in trajs2_list])
            traj_mat3 = np.asarray([traj[:len(x)] for traj in trajs3_list])
            y1 = np.mean(traj_mat1, axis=0)
            y2 = np.mean(traj_mat2, axis=0)
            y3 = np.mean(traj_mat3, axis=0)
            sem1 = [stats.sem(row) for row in np.asarray(traj_mat1).T]
            sem2 = [stats.sem(row) for row in np.asarray(traj_mat2).T]
            sem3 = [stats.sem(row) for row in np.asarray(traj_mat3).T]
            xys.append((x, y1, y2, y3, sem1, sem2, sem3, model_desc))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        plt.title(pos)
        ax.set_ylabel('Softmax Prob (+/-SEM)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        # ax.set_ylim([0, 1])  # TODO improve
        # plot
        lines = []
        for n, (x, y1, y2, y3, sem1, sem2, sem3, model_desc) in enumerate(xys):
            color = next(palette)
            l1, = ax.plot(x, y1, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-')
            l2, = ax.plot(x, y2, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='--')
            l3, = ax.plot(x, y3, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle=':')
            lines.append((copy.copy(l1), copy.copy(l2), copy.copy(l3)))
            ax.fill_between(x, np.add(y1, sem1), np.subtract(y1, sem1), alpha=FigsConfigs.FILL_ALPHA, color='grey')
            ax.fill_between(x, np.add(y2, sem2), np.subtract(y2, sem2), alpha=FigsConfigs.FILL_ALPHA, color='grey')
            ax.fill_between(x, np.add(y3, sem3), np.subtract(y3, sem3), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        labels = ['windows size={}'.format(window_size) for window_size in window_sizes]
        add_double_legend(ax, lines, labels, model_descs)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    softmax_vecs1_list_list = make_softmax_vecs_list_list(window_sizes[0])
    softmax_vecs2_list_list = make_softmax_vecs_list_list(window_sizes[1])
    softmax_vecs3_list_list = make_softmax_vecs_list_list(window_sizes[2])
    figs = [make_softmax_traj_fig(pos) for pos in ['noun']]
    # figs = [make_softmax_traj_fig(pos) for pos in sorted(GlobalConfigs.POS_TAGS_DICT.keys())]
    return figs


def make_cat_prediction_figs(model_groups, model_descs):
    def make_cat_prediction_fig():
        """
        Returns fig showing measure indicating how well probes are predicted for all groups in "model_groups"
        """
        start = time.time()

        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        # load data
        xys = []
        for models, model_desc in zip(model_groups, model_descs):
            trajs_list = []
            for model in models:
                traj = to_traj(model, model.calc_probes_cat_pred_goodness)
                trajs_list.append(traj)
            x = models[0].get_data_step_axis()
            traj_mat1 = np.asarray([traj[:len(x)] for traj in trajs_list])
            y1 = np.mean(traj_mat1, axis=0)
            sem1 = [stats.sem(row) for row in np.asarray(traj_mat1).T]
            xys.append((x, y1, sem1, model_desc))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        plt.title(model_groups[0][0].hub.mode)
        ax.set_ylabel('Cat Prediction Goodness', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        ax.set_ylim([0, 15])  # TODO improve
        # plot
        for n, (x, y1, sem1, model_desc) in enumerate(xys):
            color = next(palette)
            ax.plot(x, y1, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-', label=model_desc)
            ax.fill_between(x, np.add(y1, sem1), np.subtract(y1, sem1), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        add_single_legend(ax, model_descs)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_cat_prediction_fig()]
    return figs


def make_probe_pos_sim_figs(model_groups, model_descs, ):
    def make_probe_pos_sim_trajs_fig(pos1, w_name, pos2=None):
        """
        Returns fig showing traj for terms specified by "pos1" similarity to part-of-speech classes
        """
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        if pos2 is None:
            pos2 = pos1
        # load data
        xys = []
        x = model_groups[0][0].get_data_step_axis()
        for models, model_desc in zip(model_groups, model_descs):
            trajs1_list = []
            trajs2_list = []
            for model in models:
                pos2_sim_traj, other_sim_traj = zip(*to_traj(model, model.calc_pos1_pos2_sim_and_all_but_pos2_sim_n,
                                                             pos1, pos2, w_name))
                trajs1_list.append(pos2_sim_traj)
                trajs2_list.append(other_sim_traj)
            traj_mat1 = np.asarray([traj[:len(x)] for traj in trajs1_list])
            traj_mat2 = np.asarray([traj[:len(x)] for traj in trajs2_list])
            y1 = np.mean(traj_mat1, axis=0)
            y2 = np.mean(traj_mat2, axis=0)
            sem1 = [stats.sem(row) for row in np.asarray(traj_mat1).T]
            sem2 = [stats.sem(row) for row in np.asarray(traj_mat2).T]
            xys.append((x, y1, y2, sem1, sem2, model_desc))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        plt.title(model_groups[0][0].hub.mode if pos1 == 'probe' else pos1)
        ax.set_ylabel('{} Similarity (+/-SEM)'.format(w_name), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        # plot
        lines = []
        ax.axhline(y=0, linestyle='--', color='grey')
        for n, (x, y1, y2, sem1, sem2, model_desc) in enumerate(xys):
            color = next(palette)
            l1, = ax.plot(x, y1, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-')
            l2, = ax.plot(x, y2, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='--')
            lines.append((copy.copy(l1), copy.copy(l2)))
            ax.fill_between(x, np.add(y1, sem1), np.subtract(y1, sem1), alpha=FigsConfigs.FILL_ALPHA, color='grey')
            ax.fill_between(x, np.add(y2, sem2), np.subtract(y2, sem2), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        labels = ['sim to {}'.format(pos2), 'sim to all pos but {}'.format(pos2)]
        add_double_legend(ax, lines, labels, model_descs)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_probe_pos_sim_trajs_fig(pos, w_name='wx') for pos in ['noun']]
    # figs = [make_probe_pos_sim_trajs_fig(pos, w_name='wx') for pos in sorted(GlobalConfigs.POS_TAGS_DICT.keys())]
    # figs += [make_probe_pos_sim_trajs_fig(pos, w_name='wy') for pos in sorted(GlobalConfigs.POS_TAGS_DICT.keys())]
    return figs


def make_cat_probe_ba_diff_figs(model_groups, model_descs):
    def make_cat_probe_ba_diff_fig(cat, max_num_probes=50, ylim=.4):
        start = time.time()
        # load data
        trajs1 = []
        for models in model_groups:
            avg_probe_ba_lists = []
            for model_id, model in enumerate(models):
                avg_probe_ba_lists.append(model.avg_probe_ba_o_list)
            trajs1.append(np.mean(np.asarray(avg_probe_ba_lists), axis=0))
        avg_probe_ba_diff_list = np.subtract(*trajs1)
        cat_probes = model_groups[0][0].hub.probe_store.cat_probe_list_dict[cat]
        cat_probe_ids = [model_groups[0][0].hub.probe_store.probe_id_dict[cat_probe] for cat_probe in cat_probes]
        tups = [(avg_probe_ba_diff_list[probe_id], probe) for probe_id, probe in zip(cat_probe_ids, cat_probes)]
        sorted_tups = sorted(tups, key=itemgetter(0), reverse=True)[:max_num_probes]
        cat_avg_probe_ba_diffs, cat_probes = zip(*sorted_tups)
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        title = '\nvs.\n'.join(model_descs)
        plt.title(title)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off', bottom='off')
        ax.set_ylabel('Avg Probe Balanced Accuracy Diff', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xlabel('Probes in {}'.format(cat), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        num_sorted_tups = len(sorted_tups)
        xticks = np.add(range(num_sorted_tups), 0.5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(cat_probes, fontsize=FigsConfigs.AXLABEL_FONT_SIZE, rotation=90)
        ax.set_ylim([-ylim, ylim])
        # plot
        ax.plot(xticks, cat_avg_probe_ba_diffs, '.-', c='black', linewidth=FigsConfigs.LINEWIDTH)
        ax.axhline(0, linestyle='--', color='grey')
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    cats_sorted_by_ba_diff = make_cats_sorted_by_ba_diff(*model_groups)
    figs = [make_cat_probe_ba_diff_fig(cat) for cat in cats_sorted_by_ba_diff]
    return figs


def make_cat_ap_diff_figs(model_groups, model_descs, remove_punct=False):
    def make_avg_mats(cat_type):
        assert len(model_groups) == 2
        result = []
        for models, model_desc in zip(model_groups, model_descs):
            mats = []
            for model_id, model in enumerate(models):
                softmax_by_cat_mat = model.make_softmax_by_cat_mat(cat_type, remove_punct=remove_punct)
                mats.append(softmax_by_cat_mat)
            avg_mat = np.mean(np.array(mats), axis=0)
            result.append(avg_mat)
        assert result[0].shape[1:] == result[1].shape[1:]
        return result

    def sort_rows(mat):
        row_ids = ~np.all(mat == 0, axis=0)
        filtered_mat = mat[row_ids]
        result = filtered_mat[np.sum(np.abs(filtered_mat), axis=1).argsort()]
        return result

    def make_cat_ap_diff_fig(row, cat, cats, max_num_cats=50, ylim=0.2):  # TODO ylim
        start = time.time()
        # load data
        y, xticklabels = zip(*sorted(zip(row, cats), key=lambda t: t[0], reverse=True)[:max_num_cats])
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off', bottom='off')
        ax.set_ylabel('Softmax Diff', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xlabel('Correct Category: "{}"'.format(cat), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels(xticklabels, fontsize=FigsConfigs.AXLABEL_FONT_SIZE, rotation=90)
        ax.set_ylim([-ylim, ylim])
        # plot
        ax.plot(y, '.-', c='black', linewidth=FigsConfigs.LINEWIDTH)
        ax.axhline(0, linestyle='--', color='grey')
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_avg_mats_fig(mat, cats, mode, vmax=0.25):
        start = time.time()
        # mat
        if mode == 'improvement':
            mat_copy = np.copy(mat)
            #
            off_dig_ids = ~np.eye(mat.shape[0], dtype=bool)
            mat_copy[off_dig_ids] = -mat_copy[off_dig_ids]  # off-diagonal inverted
            mat_copy[mat_copy < 0] = 0
            heatmap_mat = mat_copy
            cbar_label = 'Softmax Improvement'
            title = '\nvs.\n'.join(model_descs)
        elif mode == 'worsening':
            mat_copy = np.copy(mat)
            #
            diag_ids = np.diag_indices(mat.shape[0])
            mat_copy[diag_ids] = -mat_copy[diag_ids]  # diagonal inverted
            mat_copy[mat_copy < 0] = 0
            heatmap_mat = mat_copy
            cbar_label = 'Softmax Worsening'
            title = '\nvs.\n'.join(model_descs)
        elif mode == 'difference':
            heatmap_mat = mat
            cbar_label = 'Softmax Difference'
            title = '\nvs.\n'.join(model_descs)
        elif mode in model_descs:
            heatmap_mat = mat
            cbar_label = 'Softmax'
            title = mode
        else:
            raise AttributeError('rnnlab: Invalid arg to "mode".')
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 5))
        plt.title(title)
        # plot
        sns.heatmap(heatmap_mat, ax=ax, square=True, annot=False,
                    annot_kws={"size": 6}, cbar_kws={"shrink": .5},
                    cmap='jet', fmt='d', vmin=-vmax, vmax=vmax)
        # colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_label(cbar_label)
        # ax (needs to be below plot for axes to be labeled)
        ax.set_xticks(np.arange(len(cats)) + 0.5)
        ax.set_yticks(np.arange(len(cats)) + 0.5)
        ax.set_yticklabels(cats, rotation=0)
        ax.set_xticklabels(cats, rotation=90)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    # laod data
    pos_avg_mats = make_avg_mats('pos')
    probes_avg_mats = make_avg_mats('probes')
    pos_avg_mats_diff = np.subtract(*pos_avg_mats)
    probes_avg_mats_diff = np.subtract(*probes_avg_mats)
    pos_cats = [cat + 's' for cat in sorted(GlobalConfigs.POS_TAGS_DICT.keys())]
    if remove_punct:
        pos_cats.remove('punctuations')
    # figs
    figs = [make_avg_mats_fig(pos_avg_mats[0], pos_cats, model_descs[0]),
            make_avg_mats_fig(pos_avg_mats[1], pos_cats, model_descs[1]),
            make_avg_mats_fig(probes_avg_mats_diff, model_groups[0][0].hub.probe_store.cats, mode='difference')]
    figs += [make_avg_mats_fig(pos_avg_mats_diff, pos_cats, mode=mode) for mode in ['improvement', 'worsening']]
    figs += [make_cat_ap_diff_fig(row, cat, pos_cats)
             for row, cat in zip(sort_rows(pos_avg_mats_diff), pos_cats)]
    return figs


def make_ap_by_cat_figs(model_groups, model_descs, remove_punct=False):
    def make_cat_expected_y_dict(cat_type):
        # cats
        if cat_type == 'probes':
            cats = model_groups[0][0].hub.probe_store.cats
            expected_y = [len(model_groups[0][0].hub.probe_store.cat_probe_list_dict[cat]) /
                          model_groups[0][0].hub.train_terms.num_types
                          for cat in cats]
        elif cat_type == 'pos':
            cats = [cat + 's' for cat in sorted(GlobalConfigs.POS_TAGS_DICT.keys())]
            if remove_punct:
                cats.remove('punctuations')
            expected_y = [len(getattr(model_groups[0][0].hub, cat)) /
                          model_groups[0][0].hub.train_terms.num_types
                          for cat in cats]
        else:
            raise AttributeError('rnnlab: Invalid arg to "cat_type".')
        # result
        result = {cat: y for cat, y in zip(cats, expected_y)}
        return result

    def make_cat_average_precision_list_fig(cat_type):
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        # load data
        cat_expected_y_dict = make_cat_expected_y_dict(cat_type)
        cats_to_sort_by, expected_y = zip(*sorted(cat_expected_y_dict.items(), key=lambda t: t[1]))
        num_cats = len(cats_to_sort_by)
        xys = []
        for models, model_desc in zip(model_groups, model_descs):
            cat_ap_lists = []
            for model_id, model in enumerate(models):
                cat_ap_lists.append(model.calc_sorted_cat_ap_list(cat_type, cats_to_sort_by, remove_punct=remove_punct))
            y = np.mean(np.asarray(cat_ap_lists), axis=0)
            xys.append((y, model_desc))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
        plt.title('cat_type="{}"'.format(cat_type))
        ax.set_ylabel('Average Precision', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xlabel('Category', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xticks(np.arange(num_cats))
        ax.set_xticklabels(cats_to_sort_by, rotation=90)
        # plot
        bar_width = 0.35
        for x, (y, model_desc) in zip([np.arange(num_cats) + (bar_width * i) for i in range(len(xys))], xys):
            c = next(palette)
            ax.axhline(y=np.mean(y), color=c, zorder=3, linestyle='-', alpha=0.75)
            ax.bar(x, expected_y, bar_width, zorder=3, color='grey')  # expected ap
            ax.bar(x, y, bar_width, color=c, zorder=2, label=model_desc)
        plt.legend(fontsize=FigsConfigs.LEG_FONTSIZE, frameon=False, loc='best')
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_cats_ap_trajs_fig():
        """
        Returns fig showing average precision trajectory
        """
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        # load data
        xys = []
        for models in model_groups:
            trajs1 = []
            trajs2 = []
            for model in models:
                traj1, traj2 = to_trajs(model, model.calc_cats_ap, ('pos', 'probes'))
                trajs1.append(traj1)
                trajs2.append(traj2)
            x = models[0].get_data_step_axis()
            traj_mat1 = np.asarray([traj[:len(x)] for traj in trajs1])
            traj_mat2 = np.asarray([traj[:len(x)] for traj in trajs2])
            y1 = np.mean(traj_mat1, axis=0)
            y2 = np.mean(traj_mat2, axis=0)
            sem1 = [stats.sem(row) for row in np.asarray(traj_mat1).T]
            sem2 = [stats.sem(row) for row in np.asarray(traj_mat2).T]
            xys.append((x, y1, y2, sem1, sem2))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        plt.title(model_groups[0][0].hub.mode)
        ax.set_ylabel('Mean Average Precision (+/-SEM)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        ax.set_ylim([0, 0.5])  # TODO improve
        # plot
        lines = []
        for n, (x, y1, y2, sem1, sem2) in enumerate(xys):
            color = next(palette)
            # lines
            l1, = ax.plot(x, y1, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-')
            l2, = ax.plot(x, y2, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='--')
            lines.append((copy.copy(l1), copy.copy(l2)))
            # sem
            ax.fill_between(x, np.add(y1, sem1), np.subtract(y1, sem1), alpha=FigsConfigs.FILL_ALPHA, color='grey')
            ax.fill_between(x, np.add(y2, sem2), np.subtract(y2, sem2), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        labels = ['part-of-speech', 'probes']
        add_double_legend(ax, lines, labels, model_descs)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_cat_ap_traj_fig(cat):
        """
        Returns fig showing average precision trajectory for "cat"
        """
        assert hasattr(model_groups[0][0].hub, cat +  's')  # must be pos class
        start = time.time()
        num_model_groups = len(model_groups)
        palette = cycle(sns.color_palette("hls", num_model_groups))
        # load data
        xys = []
        for models in model_groups:
            trajs1 = []
            for model in models:
                traj1 = to_traj(model, model.calc_pos_cat_ap, cat)  # TODO test
                trajs1.append(traj1)
            x = models[0].get_data_step_axis()
            traj_mat1 = np.asarray([traj[:len(x)] for traj in trajs1])
            y1 = np.mean(traj_mat1, axis=0)
            sem1 = [stats.sem(row) for row in np.asarray(traj_mat1).T]
            xys.append((x, y1,  sem1))
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 6))
        plt.title(cat)
        ax.set_ylabel('Average Precision (+/-SEM)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xlabel('Mini Batch', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(human_format))
        ax.yaxis.grid(True)
        # ax.set_ylim([0, 0.5])  # TODO improve
        # plot
        for n, (x, y1, sem1) in enumerate(xys):
            color = next(palette)
            ax.plot(x, y1, '-', linewidth=FigsConfigs.LINEWIDTH, color=color, linestyle='-')
            ax.fill_between(x, np.add(y1, sem1), np.subtract(y1, sem1), alpha=FigsConfigs.FILL_ALPHA, color='grey')
        plt.tight_layout()
        add_single_legend(ax, model_descs)
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    # figs
    figs = [make_cat_ap_traj_fig('noun'),
            # make_cat_average_precision_list_fig('pos'),
            # make_cat_average_precision_list_fig('probes'),
            make_cats_ap_trajs_fig(),
            ]
    return figs


def make_correlations_figs(model_groups, model_descs):
    def make_probe_corr_fig(x_name, y_name):
        """
        Return fig showing correlation between probe "x" and probe "
        """
        start = time.time()
        # load data
        xs = []
        ys = []
        for models in model_groups:
            model_xs = []
            model_ys = []
            for model in models:
                name_y_dict = {'avg_probe_fs_o': model.avg_probe_fs_o_list,
                               'avg_probe_fs_s': model.avg_probe_fs_s_list,
                               'avg_probe_fs_n': model.avg_probe_fs_n_list,
                               'avg_probe_pp': model.avg_probe_pp_list}
                name_x_dict = {'avg_probe_noun_sim_n': model.avg_probe_noun_sim_n_list,  # TODO calculates all each time
                               'avg_probe_verb_sim_n': model.avg_probe_verb_sim_n_list,
                               'avg_probe_sim': model.avg_probe_sim_list,
                               'median_probe_cg': model.median_probe_cg_list,
                               'avg_probe_cat_pred_goodness': model.avg_probe_cat_pred_goodness_list,
                               'avg_probe_loc': model.avg_probe_loc_list,
                               'avg_probe_loc_asymmetry': model.avg_probe_loc_asymmetry_list,
                               'probe_freq': model.probe_freq_list,
                               'probe_cat_freq': model.probe_cat_freq_list}
                model_xs.append(name_x_dict[x_name])
                model_ys.append(name_y_dict[y_name])
            xs.append(np.mean(model_xs, axis=0))
            ys.append(np.mean(model_ys, axis=0))
        y = np.subtract(*ys)
        y_name += ' difference'
        if x_name in ['avg_probe_loc', 'avg_probe_loc_asymmetry', 'probe_freq', 'probe_cat_freq']:
            x = xs[0]  # the above typically are the same between groups
        else:
            x = np.subtract(*xs)
            x_name += ' difference'
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_ylabel(y_name, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xlabel(x_name, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # plot
        ax.scatter(x, y, s=FigsConfigs.MARKERSIZE, facecolor='black', zorder=2)
        ax.axhline(y=0, linestyle='--', c='grey', zorder=1)
        ax.axvline(x=0, linestyle='--', c='grey', zorder=1)
        plot_best_fit_line(ax, zip(x, y), fontsize=12, x_pos=0.75, y_pos=0.1)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_cat_corr_fig(x_name, y_name):
        """
        Return fig showing correlation between cat "x" and cat "y"
        """
        start = time.time()
        # load data
        cats = model_groups[0][0].hub.probe_store.cats
        xs = []
        ys = []
        for models in model_groups:
            model_xs = []
            model_ys = []
            for model in models:
                name_y_dict = {'avg_cat_fs_o': [np.mean(l)
                                                for l in model.get_sorted_cat_avg_probe_ba_lists(cats, 'ordered')],
                               'avg_cat_fs_s': [np.mean(l)
                                                for l in model.get_sorted_cat_avg_probe_ba_lists(cats, 'shuffled')],
                               'avg_cat_fs_n': [np.mean(l)
                                                for l in model.get_sorted_cat_avg_probe_ba_lists(cats, 'none')]}
                if x_name == 'avg_cat_ap_probes':
                    model_xs.append(model.calc_sorted_cat_ap_list('probes', cats))
                elif x_name == 'avg_cat_context_cat_vs_item_goodness':
                    model_xs.append(model.calc_avg_cat_context_cat_vs_item_goodness_list())
                else:
                    raise AttributeError('rnnlab: Invalid arg to "x_name.')
                model_ys.append(name_y_dict[y_name])
            xs.append(np.mean(model_xs, axis=0))
            ys.append(np.mean(model_ys, axis=0))
        y = np.subtract(*ys)
        y_name += ' difference'
        x = np.subtract(*xs)
        x_name += ' difference'
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_ylabel(y_name, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xlabel(x_name, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # plot
        ax.scatter(x, y, s=FigsConfigs.MARKERSIZE, facecolor='black', zorder=2)
        ax.axhline(y=0, linestyle='--', c='grey', zorder=1)
        ax.axvline(x=0, linestyle='--', c='grey', zorder=1)
        plot_best_fit_line(ax, zip(x, y), fontsize=12, x_pos=0.75, y_pos=0.1)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    # cat_x_names = ['avg_cat_ap_probes', 'avg_cat_context_cat_vs_item_goodness']
    cat_x_names = ['avg_cat_context_cat_vs_item_goodness']
    # probe_x_names = ['avg_probe_noun_sim_n', 'avg_probe_verb_sim_n', 'avg_probe_sim', 'median_probe_cg',
    #                  'avg_probe_loc', 'avg_probe_loc_asymmetry', 'probe_freq', 'probe_cat_freq',
    #                  'avg_probe_cat_pred_goodness']
    probe_x_names = ['avg_probe_noun_sim_n', 'avg_probe_verb_sim_n']
    figs = []
    figs += [make_cat_corr_fig(x_name, 'avg_cat_fs_o') for x_name in cat_x_names]
    # figs += [make_cat_corr_fig(x_name, 'avg_cat_fs_s') for x_name in cat_x_names]
    # figs += [make_cat_corr_fig(x_name, 'avg_cat_fs_n') for x_name in cat_x_names]
    # figs += [make_probe_corr_fig(x_name, 'avg_probe_fs_o') for x_name in probe_x_names]
    # figs += [make_probe_corr_fig(x_name, 'avg_probe_fs_s') for x_name in probe_x_names]
    # figs += [make_probe_corr_fig(x_name, 'avg_probe_fs_n') for x_name in probe_x_names]
    # figs += [make_probe_corr_fig(x_name, 'avg_probe_pp') for x_name in probe_x_names]
    return figs


group_btn_name_figs_fn_dict = {k: globals()['make_{}_figs'.format(k)]
                               for k in chain(AppConfigs.MULTI_GROUP_BTN_NAME_INFO_DICT.keys(),
                                              AppConfigs.TWO_GROUP_BTN_NAME_INFO_DICT.keys())}
