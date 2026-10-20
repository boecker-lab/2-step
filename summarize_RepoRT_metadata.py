"""Creates a dictionary with a text based summary of a RepoRT dataset.

When making predictions for RepoRT

For dataset 0001 the output looks like:

RepoRT index :  0001
column :  Waters ACQUITY UPLC HSS T3, 150 x 2.1 mm, 1.8 µm particles, 40 °C
Eluent A :  100% h2o + 0.1% formic
Eluent B :  100% meoh + 0.1% formic
Gradient step 0 :  0.0 min: 100.0% A, 0.28 ml/min
Gradient step 1 :  1.0 min: 100.0% A, 0.28 ml/min
Gradient step 2 :  3.0 min: 90.0% A 10.0% B, 0.28 ml/min
Gradient step 3 :  18.0 min: 60.0% A 40.0% B, 0.28 ml/min
Gradient step 4 :  21.0 min: 100.0% B, 0.28 ml/min
Gradient step 5 :  25.5 min: 100.0% B, 0.28 ml/min
Gradient step 6 :  25.6 min: 100.0% A, 0.28 ml/min
Gradient step 7 :  28.0 min: 100.0% A, 0.28 ml/min

"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def summarize_column(row):
    column_summary = f"{row.get('column.name')}, {row['column.length']} x {row['column.id']} mm, {row.get('column.particle.size')} µm particles, {row.get('column.temperature')} °C"
    return column_summary


def summarize_eluents(row) -> dict[str, str]:
    def _num(value):
        """Converts to float if possible"""
        try:
            f = float(value)
        except ValueError:
            return value
        return f"{f:g}"

    ELUENT_RE = re.compile(r"^eluent\.([^.]+)\.(.+?)(\.unit)?$")
    # eluents[letter][component] = (value, unit)
    eluents = defaultdict(dict)
    for key, value in row.items():
        if value is None or value.strip() == "" or value == "0":
            continue
        m = ELUENT_RE.match(key)
        if not m or m.group(3):  # skip non-matching and *.unit columns
            continue
        letter, comp = m.group(1), m.group(2)
        unit = (row.get(f"eluent.{letter}.{comp}.unit") or "").strip()
        eluents[letter][comp] = (value.strip(), unit)
    eluents_summary = {}
    for letter in sorted(eluents):
        mix = []
        ph = eluents.get("pH", None)
        for comp, (value, unit) in eluents[letter].items():
            if comp.lower() == "ph":
                continue
            mix.append(f"{_num(value)}{unit or '%'} {comp}")
        text = " + ".join(mix)
        if ph:
            text += f", pH {ph}"
        eluents_summary["Eluent " + letter] = text
    return eluents_summary


def summarize_metadata(path_to_report_dataset) -> dict[str, str]:
    """Summarizes the column, the eluents and the gradient and combines it into one dict with summaries as strings."""
    path_to_report_dataset = Path(path_to_report_dataset)

    gradient_path = (
        path_to_report_dataset / f"{path_to_report_dataset.name}_gradient.tsv"
    )
    metadata_path = (
        path_to_report_dataset / f"{path_to_report_dataset.name}_metadata.tsv"
    )
    if not metadata_path.is_file():
        raise ValueError(f"The file {metadata_path} does not exist")
    if not gradient_path.is_file():
        raise ValueError(f"The file {gradient_path} does not exist")
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        first_row = reader.__next__()
    setup_summary = {
        "RepoRT index": path_to_report_dataset.name,
        "column": summarize_column(first_row),
    }
    return (
        setup_summary
        | summarize_eluents(first_row)
        | summarize_gradient_info(gradient_path)
    )


def summarize_gradient_info(gradient_info_file) -> dict[str, str]:
    """Summarizes the gradient info.
    E.g.:
    """
    gradient_info = pd.read_csv(
        gradient_info_file,
        sep="\t",
    )
    summary = {}
    for i, row in gradient_info.iterrows():
        row_summary = f"{row['t [min]']} min:"
        for eluent in ("A [%]", "B [%]", "C [%]", "D [%]"):
            if row[eluent]:
                row_summary += f" {row[eluent]}% {eluent[0]}"
        if row["flow rate [ml/min]"]:
            row_summary += f", {row['flow rate [ml/min]']} ml/min"
        summary[f"Gradient step {i}"] = row_summary
    return summary


if __name__ == "__main__":
    summary = summarize_metadata(
        "C:/Users/jonge094/Repos/dynamic_sops/rt_prediction/RepoRT/processed_data/0001/"
    )
    for key, value in summary.items():
        print(key, ": ", value)
