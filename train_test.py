from utils import Data
from train import preprocess, TrainArgs
import json
import numpy as np
from random import randint

args = TrainArgs().from_dict(dict(
    input=[],                   # doesn't matter here
    verbose=True,
    standardize=True,
    val_split=0, test_split=0))

# initialize Data object
data = Data(use_compound_classes=False, # not used for now (only makes sense when not considering hierarchy)
            use_system_information=True, # always yes
            metadata_void_rt=True,       # no reason to disable
            custom_features=[], # maybe just logp or something
            use_hsm=True,       # yes
            use_tanaka=True,    # yes
            use_newonehot=True, # TODO:?
            use_ph=True, # not available for all datasets
            use_gradient=True, # not sure if good idea
            repo_root_folder='/home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/',
            custom_column_fields=['column.temperature'], # basically all others don't make much sense
            tanaka_match='best_match',              # 'exact' leads to losing many mappings
            graph_mode=True,
            smiles_for_graphs=False, # TODO: ?
            )

# add some datasets
data.add_dataset_id('0048')     # tanaka: no match with particle size, only with column
data.add_dataset_id('0001')     # match also with particle size
data.add_dataset_id('0382')     # test gradient concentrations


# preprocess
train_data, _, _ = preprocess(data, args)

graphs, x, x_sys, y = train_data

# check if hsm / tanaka params were chosen correctly (reverse scaling)
scaling = json.load(open('/home/fleming/Documents/Projects/RtPredTrainingData_mostcurrent/scaling.json'))
hsm = x_sys[0][:6]
tanaka = x_sys[-1][6:12]
hsm_means = np.array([scaling[f]['mean'] for f in data.hsm_fields])
hsm_std = np.array([scaling[f]['std'] for f in data.hsm_fields])
tanaka_means = np.array([scaling[f]['mean'] for f in data.tanaka_fields])
tanaka_std = np.array([scaling[f]['std'] for f in data.tanaka_fields])
hsm * hsm_std + hsm_means       # NOTE: works ☑
tanaka * tanaka_std + tanaka_means # NOTE: works ☑

# check the rankdataset
from utils_newbg import RankDataset
traindata = RankDataset(x_mols=graphs, x_extra=x, x_sys=x_sys,
                        x_ids=data.df.iloc[data.train_indices].smiles.tolist(),
                        y=y, x_sys_global_num=data.x_info_global_num,
                        dataset_info=data.df.dataset_id.iloc[data.train_indices].tolist(),
                        void_info=data.void_info,
                        pair_step=args.pair_step,
                        pair_stop=args.pair_stop, use_pair_weights=True,
                        use_group_weights=False,
                        cluster=args.cluster,
                        no_inter_pairs=True,
                        no_intra_pairs=False,
                        max_indices_size=args.max_pair_compounds,
                        weight_mid=args.weight_mid,
                        weight_steepness=args.weight_steep,
                        dynamic_weights=args.dynamic_weights)
traindata.preprocess_doublets()

# get function should work, so just get some random indices and check arrays
for c in range(5000):
    i = randint(0, len(traindata.x1_indices))
    # print(i)
    ds1 = traindata.dataset_info[traindata.x1_indices[i]]
    ds2 = traindata.dataset_info[traindata.x2_indices[i]]
    assert ds1 == ds2, (ds1, ds2, i)
    s1 = traindata.x_ids[traindata.x1_indices[i]]
    s2 = traindata.x_ids[traindata.x2_indices[i]]
    sys1 = traindata.x_sys[traindata.x1_indices[i]]
    sys2 = traindata.x_sys[traindata.x2_indices[i]]
    assert np.isclose(sys1[:traindata.x_sys_global_num], sys2[:traindata.x_sys_global_num], atol=1e-7, equal_nan=True).all(), i
    y = traindata.y_trans[i]
    w = traindata.weights[i]
    print(w)
    rel_data = data.df.loc[(data.df.dataset_id == ds1) & (data.df.smiles.isin([s1, s2]))]
    # (True, 1) or (False, 0/-1)
    order = (rel_data.loc[rel_data.smiles == s1, 'rt'].iloc[0] > rel_data.loc[rel_data.smiles == s2, 'rt'].iloc[0], y)
    print(order)
    assert order in [(True, 1), (False, 0), (False, -1)], (order, i, w)
    ((sys1[traindata.x_sys_global_num], rel_data.loc[rel_data.smiles == s1, [c for c in rel_data.columns if c.startswith('gradient_conc_')][0]]),
     (sys2[traindata.x_sys_global_num], rel_data.loc[rel_data.smiles == s2, [c for c in rel_data.columns if c.startswith('gradient_conc_')][0]]))
