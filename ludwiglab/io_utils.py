import csv
from operator import itemgetter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ludwiglab import config


def write_pca_loadings_files(model, is_terms=config.Figs.SVD_IS_TERMS, freq_thr=config.Figs.PCA_FREQ_THR):
    if is_terms:
        acts_mat = model.term_acts_mat
        terms = model.hub.train_terms.types
    else:
        acts_mat = model.get_multi_probe_prototype_acts_df().values
        terms = model.hub.probe_store.types
    pca_model = PCA(n_components=config.Figs.NUM_PCS)
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
    for pc_id in range(config.Figs.NUM_PCS):
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