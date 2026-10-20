"""Predict retention time by using anchors from a RepoRT dataset.

These are less reliable than using the anchors for your own exact setup, but can serve as a good way to select suitable setups for your experiment.
"""

import pickle
from logging import INFO, basicConfig, info, warning
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mapping import LADModel
from predict import load_metadata, load_model, metadata_incomplete
from predict_rois import predict_rois
from summarize_RepoRT_metadata import summarize_metadata
from tap import Tap
from tqdm import tqdm


def get_path_to_report_index_folder(report_root, report_index):
    # converts 24 to 0024
    report_index_4_numbers = str(report_index).zfill(4)

    report_root = Path(report_root)
    if not (report_root.exists and report_root.is_dir):
        raise ValueError("The given Report directory is not a directory")

    path_to_report_index_folder = (
        report_root / "processed_data" / report_index_4_numbers
    )
    if not (
        path_to_report_index_folder.exists() and path_to_report_index_folder.is_dir()
    ):
        raise ValueError(
            f"The path to the given report_index: {report_index} does not exist. {path_to_report_index_folder.absolute()}"
        )
    return path_to_report_index_folder


def get_files_from_report(report_root, report_index):
    path_to_report_index_folder = get_path_to_report_index_folder(
        report_root, report_index
    )

    input_compounds_path = (
        path_to_report_index_folder
        / f"{path_to_report_index_folder.name}_rtdata_canonical_success.tsv"
    )
    input_metadata_path = (
        path_to_report_index_folder
        / f"{path_to_report_index_folder.name}_metadata.yaml"
    )
    if not input_compounds_path.is_file():
        raise ValueError(f"The file {input_compounds_path} does not exist")
    if not input_metadata_path.is_file():
        raise ValueError(f"The file {input_metadata_path} does not exist")

    lad_model_path = (
        path_to_report_index_folder
        / f"{path_to_report_index_folder.name}_lad_model.pkl"
    )
    return input_compounds_path, input_metadata_path, lad_model_path


def train_lad_model(model, repo_root_folder, repo_index, save=False):
    input_compounds_path, input_metadata_path, model_save_location = (
        get_files_from_report(repo_root_folder, repo_index)
    )

    info("load input data...")
    metadata = load_metadata(input_metadata_path)
    if metadata_incomplete(metadata):
        warning(
            f"The metadata is not complete to train a reliable model for report index {repo_index}"
        )
        return None, None
    if model_save_location.is_file():
        info(f"Loading mapping model from {model_save_location}")
        with open(model_save_location, "rb") as out:
            mapping_model = pickle.load(out)
        return mapping_model, metadata
    predicted_rois_df = predict_rois(
        model, input_compounds_path, metadata, repo_root_folder, 256, False
    )

    # anchors are all data points with annotated retention time, discarding the void volume
    # Sometimes the ROI model outputs NaN the check for is finite removes those. Not sure why this happens, but I think it can happen if the metadata is not completely correct. 
    data_anchors = predicted_rois_df.loc[
        (predicted_rois_df.rt > metadata["column.t0"])
        & np.isfinite(predicted_rois_df.roi)
    ]

    if len(data_anchors) < 10: 
        warning(f"Too few valid anchors for report {repo_index}")
        return None, None
    
    info(f"building mapping using {len(data_anchors)} anchors")
    mapping_model = LADModel(
        data_anchors,
        ols_after=True,
        ols_discard_if_negative=True,
        ols_drop_mode="2*median",
    )
    if save:
        with open(model_save_location, "wb") as out:
            info(f"Saving mapping model in {model_save_location}")
            pickle.dump(mapping_model, out)
    # todo add some automatic testing as well, showing the accuracies for the input. Maybe a plot?
    return mapping_model, metadata


def predict_for_report_index(
    model,
    file_with_smiles_to_predict,
    repo_root_folder,
    report_index,
    add_metadata_summary,
):
    mapping_model, metadata = train_lad_model(
        model, repo_root_folder, report_index, False
    )
    if mapping_model is None:
        return None

    data_to_predict = predict_rois(
        model, file_with_smiles_to_predict, metadata, repo_root_folder, 256, False
    )

    data_to_predict["rt_pred"] = mapping_model.get_mapping(data_to_predict.roi)
    predictions = data_to_predict[["smiles.std", "rt_pred"]]
    predictions.set_index("smiles.std", inplace=True)
    if add_metadata_summary:
        report_dir = get_path_to_report_index_folder(repo_root_folder, report_index)
        metadata_summary = pd.DataFrame.from_dict(
            summarize_metadata(report_dir), orient="index", columns=["rt_pred"]
        )
        predictions = pd.concat([metadata_summary, predictions])
    return predictions


def predict_for_multiple_report_experiments(
    model,
    file_with_smiles_to_predict,
    repo_root_folder,
    report_indexes,
    add_metadata_summary,
):
    prediction_per_repo_index = []
    longest_index = 0
    index_order = None
    for repo_index in tqdm(report_indexes):
        out_df = predict_for_report_index(
            model,
            file_with_smiles_to_predict,
            repo_root_folder,
            repo_index,
            add_metadata_summary,
        )
        if out_df is None:
            continue
        prediction_per_repo_index.append(out_df)
        if out_df.shape[0] > longest_index:
            longest_index = out_df.shape[0]
            index_order = out_df.index.copy()
    if not prediction_per_repo_index:
        return None
    results = pd.concat(prediction_per_repo_index, axis=1)
    results = results.reindex(index_order)
    return results


class PredictRepoRTArgs(Tap):
    smiles_to_predict: str  # TSV file with `smiles` and `rt` columns
    repo_indexes: list[
        str
    ]  # yaml file with at least `column.name`, `eluent.A.pH`, and `column.t0` specified
    model: str = "./models/2-step0525.pt"  # model to load
    gpu: bool = False  # whether to use GPU for predictions
    out: str | None = (
        None  # where to write the output (TSV format). If not specified, output will be written to screen.
    )
    batch_size: int = 256  # adjust according to available VRAM
    repo_root_folder: str = (
        "../RepoRT/"  # location of RepoRT, needed for HSM/Tanaka database
    )
    verbose: bool = False  # more info on what is being done internally
    metadata_summary = (
        True  # adds extra rows with a metadata summary above the predictions.
    )


if __name__ == "__main__":
    args = PredictRepoRTArgs().parse_args()
    if args.verbose:
        basicConfig(level=INFO)
    if args.gpu:
        torch.set_default_device("cuda")
    # load model
    info("load model...")
    model = load_model(args.model, all_in_one=True)
    if isinstance(model, tuple):
        model = model[0]
    out_df = predict_for_multiple_report_experiments(
        model,
        args.smiles_to_predict,
        args.repo_root_folder,
        args.repo_indexes,
        args.metadata_summary,
    )
    if out_df is not None:
        if args.out is None:
            info(f"showing output.")
            print(out_df)
            # out_df.to_csv(sys.stdout, sep="\t")
        else:
            info(f"saving to {args.out}.")
            out_df.to_csv(args.out, sep="\t")
    else:
        warning("No predictions were made")
