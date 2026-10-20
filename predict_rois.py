from logging import info
from time import time

import numpy as np
from utils import Data


def fake_splitting_data(d: Data, sysfeature_scaler):
    d.split_data((0, 0))
    if sysfeature_scaler is not None:
        info("standardize data")
        d.standardize(
            other_descriptor_scaler=None,
            other_sysfeature_scaler=sysfeature_scaler,
            can_create_new_scaler=False,
        )
    (
        (train_graphs, train_x, train_sys, train_y),
        (val_graphs, val_x, val_sys, val_y),
        (test_graphs, test_x, test_sys, test_y),
    ) = d.get_split_data()
    X = np.concatenate((train_x, test_x, val_x)).astype(np.float32)
    X_sys = np.concatenate((train_sys, test_sys, val_sys)).astype(np.float32)
    Y = np.concatenate((train_y, test_y, val_y))
    graphs = np.concatenate((train_graphs, test_graphs, val_graphs))
    return graphs, X, X_sys, train_sys


def prediction_rois(d, model, graphs, train_sys, X_sys, X, batch_size, verbose):
    start_time = time()
    if hasattr(model, "add_sys_features") and model.add_sys_features:
        from utils_newbg import sysfeature_graph

        info("add system features to graphs")
        smiles_list = d.df.iloc[
            np.concatenate((d.train_indices, d.test_indices, d.val_indices))
        ]["smiles"].tolist()
        assert len(graphs) == len(smiles_list)
        from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim

        if model.add_sys_features_mode == "bond":
            set_extra_bond_fdim(train_sys.shape[1])
        elif model.add_sys_features_mode == "atom":
            set_extra_atom_fdim(train_sys.shape[1])
        for i in range(len(graphs)):
            graphs[i] = sysfeature_graph(
                smiles_list[i],
                graphs[i],
                X_sys[i],
                bond_or_atom=model.add_sys_features_mode,
            )
    t0_actual = time()
    preds = model.predict(
        graphs,
        X,
        X_sys,
        batch_size=batch_size,
        ret_features=False,
        prog_bar=verbose,
    )
    t1 = time() - start_time
    t1_actual = time() - t0_actual
    info(
        f"done predicting ROIs, took {t1:.1}({t1_actual:.1})s ({t1 / len(preds):.2}s per instance [{t1_actual / len(preds):.2}s], #={len(preds)}). predicting retention times..."
    )
    d.df["roi"] = preds[
        np.arange(len(d.df.rt))[  # restore correct order
            np.argsort(np.concatenate([d.train_indices, d.test_indices, d.val_indices]))
        ]
    ]
    # predict retention times right here
    d.df["roi2"] = d.df.roi**2  # for LAD model
    return d.df


def predict_rois(
    model, input_compounds_file, metadata, repo_root_folder, batch_size, verbose
):
    data_args = model.extra_storage["data_args"]
    data_args["repo_root_folder"] = repo_root_folder
    d = Data(**data_args)
    d.add_external_data(
        input_compounds_file,
        metadata=metadata,
        remove_nan_rts=False,
        tab_mode=True,
        isomeric=True,
        split_type="evaluate",
    )
    info("computing features")
    d.compute_features(mode=None, add_descs=False)

    info("computing graphs")
    d.compute_graphs()

    info("(fake) splitting data")
    graphs, X, X_sys, train_sys = fake_splitting_data(
        d, model.extra_storage["sysfeature_scaler"]
    )

    info(f"done preprocessing. predicting ROIs...")
    predicted_rois_df = prediction_rois(
        d,
        model,
        graphs,
        train_sys,
        X_sys,
        X,
        batch_size,
        verbose,
    )
    return predicted_rois_df
