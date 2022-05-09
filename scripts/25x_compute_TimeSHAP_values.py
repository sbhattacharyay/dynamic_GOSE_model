

    
testing_multihot = np.expand_dims(testing_multihot.numpy().copy(), axis=0)

# Calculate average event prediction sequence
avg_score_over_len = get_avg_score_with_avg_event(f_hs, AverageEvent, top=85)

# Define pruning parameters
pruning_dict = {'tol': 0.025}

# Define event-level prediction parameters
event_dict = {'rs': 42, 'nsamples': 32000}

# Define feature-level explanation parameters
plot_feats = dict(zip(token_labels, token_labels))
feature_dict = {'rs': 42, 'nsamples': 32000, 'feature_names': token_labels, 'plot_features': plot_feats}

# Define cell-level explanation parameters
cell_dict = {'rs': 42, 'nsamples': 32000, 'top_x_feats': 20, 'top_x_events': 20}

# Perform pruning
coal_plot_data, coal_prun_idx = tsx.local_pruning(f_hs, testing_multihot, pruning_dict, AverageEvent, entity_uuid=None, entity_col=None, verbose=True)

pruning_idx = testing_multihot.shape[1] + coal_prun_idx
plot_lim = max(abs(coal_prun_idx)+10, 40)
pruning_plot = tsp.plot_temp_coalition_pruning(coal_plot_data, coal_prun_idx, 10)

event_data = tsx.local_event(f_hs, testing_multihot, event_dict, entity_uuid=None, entity_col=None, baseline=AverageEvent, pruned_idx=pruning_idx)
event_plot = tsp.plot_event_heatmap(event_data)

feature_data = tsx.local_feat(f_hs, testing_multihot, feature_dict, entity_uuid=None, entity_col=None, baseline=AverageEvent, pruned_idx=pruning_idx)

if cell_dict:
    cell_data=tsx.local_cell_level(f_hs,testing_multihot,cell_dict,event_data,feature_data,entity_uuid=None,entity_col=None,baseline=AverageEvent,pruned_idx=pruning_idx)
    feat_names = list(feature_data['Feature'].values)[:-1] # exclude pruned events
    cell_plot = tsp.plot_cell_level(cell_data, feat_names, feature_dict.get('plot_features'))
    plot_report = (pruning_plot | event_plot | cell_plot).resolve_scale(color='independent')
else:
    plot_report = (pruning_plot | event_plot | feature_plot).resolve_scale(color='independent')


tsx.local_report(f_hs, testing_multihot, pruning_dict, event_dict, feature_dict, cell_dict=cell_dict, baseline=AverageEvent, verbose=True)



# Create vector of target column indices
class_indices = list(range(7))
gose_classes = np.sort(test_splits.GOSE.unique())



############################################################ 
# SAMPLE
data_directories = next(os.walk("/home/sb2406/python_venv/bin/timeshap/notebooks/AReM/AReM"))[1]

all_csvs = []
for folder in data_directories:
    if folder in ['bending1', 'bending2']:
        continue
    folder_csvs = next(os.walk(f"/home/sb2406/python_venv/bin/timeshap/notebooks/AReM/AReM/{folder}"))[2]
    for data_csv in folder_csvs:
        if data_csv == 'dataset8.csv' and folder == 'sitting':
            # this dataset only has 479 instances
            # it is possible to use it, but would require padding logic
            continue
        loaded_data = pd.read_csv(f"/home/sb2406/python_venv/bin/timeshap/notebooks/AReM/AReM/{folder}/{data_csv}", skiprows=4)
        print(f"{folder}/{data_csv} ------ {loaded_data.shape}")
        
        csv_id = re.findall(r'\d+', data_csv)[0]
        loaded_data['id'] = csv_id
        loaded_data['all_id'] = f"{folder}_{csv_id}"
        loaded_data['activity'] = folder
        all_csvs.append(loaded_data)

all_data = pd.concat(all_csvs)
raw_model_features = ['avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23']
all_data.columns = ['timestamp', 'avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23', 'id', 'all_id', 'activity']

# choose ids to use for test
ids_for_test = np.random.choice(all_data['id'].unique(), size=4, replace=False)

d_train =  all_data[~all_data['id'].isin(ids_for_test)]
d_test = all_data[all_data['id'].isin(ids_for_test)]

class NumericalNormalizer:
    def __init__(self, fields: list):
        self.metrics = {}
        self.fields = fields

    def fit(self, df: pd.DataFrame ) -> list:
        means = df[self.fields].mean()
        std = df[self.fields].std()
        for field in self.fields:
            field_mean = means[field]
            field_stddev = std[field]
            self.metrics[field] = {'mean': field_mean, 'std': field_stddev}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Transform to zero-mean and unit variance.
        for field in self.fields:
            f_mean = self.metrics[field]['mean']
            f_stddev = self.metrics[field]['std']
            # OUTLIER CLIPPING to [avg-3*std, avg+3*avg]
            df[field] = df[field].apply(lambda x: f_mean - 3 * f_stddev if x < f_mean - 3 * f_stddev else x)
            df[field] = df[field].apply(lambda x: f_mean + 3 * f_stddev if x > f_mean + 3 * f_stddev else x)
            if f_stddev > 1e-5:
                df[f'p_{field}_normalized'] = df[field].apply(lambda x: ((x - f_mean)/f_stddev))
            else:
                df[f'p_{field}_normalized'] = df[field].apply(lambda x: x * 0)
        return df
    
normalizor = NumericalNormalizer(raw_model_features)
normalizor.fit(d_train)
d_train_normalized = normalizor.transform(d_train)
d_test_normalized = normalizor.transform(d_test)

model_features = [f"p_{x}_normalized" for x in raw_model_features]
time_feat = 'timestamp'
label_feat = 'activity'
sequence_id_feat = 'all_id'

model_features = [f"p_{x}_normalized" for x in raw_model_features]
time_feat = 'timestamp'
label_feat = 'activity'
sequence_id_feat = 'all_id'

plot_feats = {
    'p_avg_rss12_normalized': "Mean Chest <-> Right Ankle",
    'p_var_rss12_normalized': "STD Chest <-> Right Ankle",
    'p_avg_rss13_normalized': "Mean Chest <-> Left Ankle",
    'p_var_rss13_normalized': "STD Chest <-> Left Ankle",
    'p_avg_rss23_normalized': "Mean Right Ankle <-> Left Ankle",
    'p_var_rss23_normalized': "STD Right Ankle <-> Left Ankle",
}

chosen_activity = 'cycling'

d_train_normalized['label'] = d_train_normalized['activity'].apply(lambda x: int(x == chosen_activity))
d_test_normalized['label'] = d_test_normalized['activity'].apply(lambda x: int(x == chosen_activity))

def df_to_Tensor(df, model_feats, label_feat, group_by_feat, timestamp_Feat):
    sequence_length = len(df[timestamp_Feat].unique())
    
    data_tensor = np.zeros((len(df[group_by_feat].unique()), sequence_length, len(model_feats)))
    labels_tensor = np.zeros((len(df[group_by_feat].unique()), 1))
    
    for i, name in enumerate(df[group_by_feat].unique()):
        name_data = df[df[group_by_feat] == name]
        sorted_data = name_data.sort_values(timestamp_Feat)
        
        data_x = sorted_data[model_feats].values
        labels = sorted_data[label_feat].values
        assert labels.sum() == 0 or labels.sum() == len(labels)
        data_tensor[i, :, :] = data_x
        labels_tensor[i, :] = labels[0]
    data_tensor = torch.from_numpy(data_tensor).type(torch.FloatTensor)
    labels_tensor = torch.from_numpy(labels_tensor).type(torch.FloatTensor)
    
    return data_tensor, labels_tensor

train_data, train_labels = df_to_Tensor(d_train_normalized, model_features, 'label', sequence_id_feat, time_feat)
test_data, test_labels = df_to_Tensor(d_test_normalized, model_features, 'label', sequence_id_feat, time_feat)

import torch.nn as nn
class ExplainedRNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 cfg: dict,
                 ):
        super(ExplainedRNN, self).__init__()
        self.hidden_dim = cfg.get('hidden_dim', 32)
        torch.manual_seed(cfg.get('random_seed', 42))

        self.recurrent_block = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            batch_first=True,
            num_layers=2,
            )
        
        self.classifier_block = nn.Linear(self.hidden_dim, 1)
        self.output_activation_func = nn.Sigmoid()

    def forward(self,
                x: torch.Tensor,
                hidden_states: tuple = None,
                ):
        
        print(x.shape)
        
        if hidden_states is None:
            output, hidden = self.recurrent_block(x)
        else:
            output, hidden = self.recurrent_block(x, hidden_states)

        # -1 on hidden, to select the last layer of the stacked gru
        assert torch.equal(output[:,-1,:], hidden[-1, :, :])
        
        y = self.classifier_block(hidden[-1, :, :])
        y = self.output_activation_func(y)
        return y, hidden
    
import torch.optim as optim

model = ExplainedRNN(len(model_features), {})
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

learning_rate = 0.005
EPOCHS = 8

import tqdm
import copy
for epoch in tqdm.trange(EPOCHS):
    train_data_local = copy.deepcopy(train_data)
    train_labels_local = copy.deepcopy(train_labels)
    
    y_pred, hidden_states = model(train_data_local)
    train_loss = loss_function(y_pred, train_labels_local)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    with torch.no_grad():
        test_data_local = copy.deepcopy(test_data)
        test_labels_local = copy.deepcopy(test_labels)
        test_preds, _ = model(test_data_local)
        test_loss = loss_function(test_preds, test_labels_local)
        print(f"Train loss: {train_loss.item()} --- Test loss {test_loss.item()} ")
        
from timeshap.wrappers import TorchModelWrapper
model_wrapped = TorchModelWrapper(model)
f_hs = lambda x, y=None: model_wrapped.predict_last_hs(x, y)

from timeshap.utils import calc_avg_event
average_event = calc_avg_event(d_train_normalized, numerical_feats=model_features, categorical_feats=[])

from timeshap.utils import get_avg_score_with_avg_event
avg_score_over_len = get_avg_score_with_avg_event(f_hs, average_event, top=480)

positive_sequence_id = f"cycling_{np.random.choice(ids_for_test)}"
pos_x_pd = d_test_normalized[d_test_normalized['all_id'] == positive_sequence_id]

# select model features only
pos_x_data = pos_x_pd[model_features]
# convert the instance to numpy so TimeSHAP receives it
pos_x_data = np.expand_dims(pos_x_data.to_numpy().copy(), axis=0)

pruning_dict = {'tol': 0.025}
event_dict = {'rs': 42, 'nsamples': 32000}
feature_dict = {'rs': 42, 'nsamples': 32000, 'feature_names': model_features, 'plot_features': plot_feats}
cell_dict = {'rs': 42, 'nsamples': 32000, 'top_x_feats': 2, 'top_x_events': 2}
tsx.local_report(f_hs, pos_x_data, pruning_dict, event_dict, feature_dict, cell_dict=cell_dict, entity_uuid=positive_sequence_id, entity_col='all_id', baseline=average_event)

pruning_dict = {'tol':0.025,}
coal_plot_data, coal_prun_idx = tsx.local_pruning(f_hs, pos_x_data, pruning_dict, average_event, positive_sequence_id, sequence_id_feat, False)
# coal_prun_idx is in negative terms
pruning_idx = pos_x_data.shape[1] + coal_prun_idx
pruning_plot = plot_temp_coalition_pruning(coal_plot_data, coal_prun_idx, plot_limit=40)
pruning_plot




event_dict = {'path': 'outputs/event_all.csv', 'rs': 42, 'nsamples': 32000}
event_data = event_explain_all(f_hs, pos_dataset, sequence_id_feat, average_event, event_dict, prun_indexes, model_features, time_feat)
event_global_plot = tsp.plot_global_event(event_data)
event_global_plot

feature_dict = {'path': 'outputs/feature_all.csv', 'rs': 42, 'nsamples': 32000, 'feature_names': model_features, 'plot_features': plot_feats, }
feat_data = feat_explain_all(f_hs, pos_dataset, sequence_id_feat, average_event, feature_dict, prun_indexes, model_features, time_feat)
feat_global_plot = tsp.plot_global_feat(feat_data, **feature_dict)
feat_global_plot
