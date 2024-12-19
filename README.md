# Two-step retention time prediction

Use `predict.py` to first predict retention order indices and then map these to retention times using anchor compounds.

Two input files are needed:
- **Structures with retention times for anchor compounds** in TSV format:
  ```
  smiles	rt
  C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)OC4C(C(C(C(O4)CO)O)O)O
  C1=CC(=CC=C1C=CC(=O)OC(C(C(=O)O)O)C(=O)O)O
  CCC(C)CN
  C1=C(OC(=C1)C=O)CO
  C1=CC=C(C(=C1)C(=O)O)N
  COC1=C(C=CC(=C1)C=CC=O)O
  CC(C)CCN	6.9
  C1=CC=C2C(=C1)C(=CN2)CC(=O)C(=O)O	30.285
  C(CC(=O)N)C(C(=O)O)N	1.3
  C1=CC=C(C(=C1)C(=O)CC(C(=O)O)N)N	11.005
  ```
- **Information on the chromatographic setup** in YAML format, similar to what is used in [RepoRT](https://github.com/michaelwitting/RepoRT):
  ```yaml
  column:
    name: Waters ACQUITY UPLC HSS T3
    t0: 0.735
  eluent:
    A:
      pH: 3
   ```
   The column should have [HSM](https://github.com/michaelwitting/RepoRT/blob/master/resources/hsm_database/hsm_database.tsv) and [Tanaka](https://github.com/michaelwitting/RepoRT/blob/master/resources/tanaka_database/tanaka_database.tsv) parameters available. If provided, twice `t0` is used as the void threshold, anchors below will not be considered during mapping.

Consider standardizing compound structures before, using e.g., [standardizeUtils](https://github.com/boecker-lab/standardizeUtils).

To access HSM and Tanaka parameters, a copy of [RepoRT](https://github.com/michaelwitting/RepoRT) needs to be available.

Use `predict.py` like this:
```bash
python predict.py  --model models/twostep_everything_predready.pt  --repo_root_folder <path to RepoRT> \
       --input_compounds test/test_input.tsv --input_metadata test/test_metdata.yaml \
       --out test/test_output.tsv
```

With docker:
```bash
docker run -v $(pwd)/test:/app/test -v <path to RepoRT>:/RepoRT -it --rm ghcr.io/boecker-lab/twosteprt:latest \
       python predict.py --model models/twostep_everything_predready.pt --repo_root_folder /RepoRT \
       --input_compounds test/test_input.tsv --input_metadata test/test_metadata.yaml
```

An output with predicted retention times will be generated:
```
	smiles	rt_pred
0	C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)OC4C(C(C(C(O4)CO)O)O)O	26.192242
1	C1=CC(=CC=C1C=CC(=O)OC(C(C(=O)O)O)C(=O)O)O	11.60789
2	CCC(C)CN	19.052101
3	C1=C(OC(=C1)C=O)CO	20.10303
4	C1=CC=C(C(=C1)C(=O)O)N	16.751137
5	COC1=C(C=CC(=C1)C=CC=O)O	24.9707
```

## Dependencies

The following dependencies are required:
```
- python=3.12
- chemprop=1.6.1
- pulp
- pytorch
- rdkit
- statsmodels
- tensorboard
- tqdm
- yaml
- numpy<2
```

A conda/mamba environment is provided:
```bash
mamba env create -n twosteprt -f env.yaml
mamba activate twosteprt
```

For GPU support, the `pytorch-cuda`-package has to be added with the appropriate version, e.g., `pytorch-cuda=11.8`. See [env_cuda.yaml](env_cuda.yaml).
A [Dockerfile](Dockerfile) and [container](ghcr.io/boecker-lab/twosteprt:latest) is provided as well.


## Training ROI prediction models

```bash
python train.py --input <IDs of RepoRT datasets> --epsilon 10s \
       --run_name twosteproi --save_data  \
       --batch_size 512 --epochs 10 --sysinfo --columns_use_hsm --columns_use_tanaka --use_ph \
       --repo_root_folder <path to RepoRT> --clean_data \
       --encoder_size 512 --sizes 256 64 --sizes_sys 256 256 \
       --pair_step 1 --pair_stop None --sample --sampling_count 500_000 --no_group_weights \
       --mpn_no_residual_connections_encoder --no_standardize
```

Add `--gpu` to enable training on GPU.

Model training creates three files:
1. The model itself, `twosteproi.pt` (with option `--ep_save` files for every epoch are created: `twosteproi_ep1.pt` etc.)
2. Processed training data, `twosteproi_data.pkl`
3. A JSON file detailing the training configuration, `twosteproi_config.json`

To make the trained model ready for prediction, use the `repackage_model.py`-script:
```bash
python repackage_model.py twosteproi.pt twosteproi_predready.pt
```

This combines all information required for prediction into one file.

A model trained on 171 manually curated reversed-phase datasets from RepoRT (version 94f43c1b) is
provided in the `models` subdirectory (`models/twostep_everything_predready.pt`).

## Evaluation of retention order prediction accuracy

```bash
python evaluate.py --model <path to trained model> --test_sets <IDs of RepoRT datasets> \
       --repo_root_folder <path to RepoRT> --epsilon 10s
```
