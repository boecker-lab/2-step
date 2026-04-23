## Data splits for benchmark datasets

This folder contains data splits for the benchmark datasets:

- FEM long ([RepoRT](https://github.com/michaelwitting/RepoRT) dataset 0002) with 399 compounds and gradient length 60 min
- IPB Halle (0003) with 80 compounds and gradient length 20 min
- UniToyama Atlantis (0018) with 78 compounds and gradient length 40 min
- Eawag XBridgeC18 (0019) with 364 compounds and gradient length 30 min
- LIFE old (0055) with 172 compounds and gradient length 6 min
- LIFE new (0055) with 162 compounds and gradient length 7 min

As described in the publication, data was split into 10 folds, in two scenarios:
- uniform split (`_uniform_10cv`)
- realistic split: data was split in a way, that there is no compound in the train portion with [myopic MCES](https://github.com/AlBi-HHU/myopic-mces) smaller than 10, for all compounds in the test portion (`mces_10cv`). See the "Methods" section for details.

As previous methods benefit from including compounds eluting in the void volume in the *training data*, this was done for all splits in this folder (`_withvoid_`). Splits with these compounds completely removed are provided in the subfolder [additional/training_folds_without_void](additional/training_folds_without_void).

Due to the constraint of removing similar compounds from the train portion (MCES distance smaller than 10), folds from the "realistic split" scenario are partly unbalanced. Manually constructed splits with balanced folds are provided in the subfolder [additional/manual_mces_splits_balanced](additional/manual_mces_splits_balanced). Here, the MCES-constraint was softened for large clusters, resulting in folds where training and testing portions are more similar as in the realistic scenario. These splits are therefore "easier".
