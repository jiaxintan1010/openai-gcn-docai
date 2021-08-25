from graph import Grapher
import torch
import scipy.sparse
import torch_geometric.data
import networkx as nx
import numpy as np
import os 
import random

def from_networkx(G):
    """Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass 

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    return data

def get_data():
    """
    returns one big graph with unconnected graphs with the following:
    - x (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
    - edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]. (default: None)
    - edge_attr (Tensor, optional) – Edge feature matrix with shape [num_edges, num_edge_features]. (default: None)
    - y (Tensor, optional) – Graph or node targets with arbitrary shape. (default: None)
    - validation mask, training mask and testing mask 
    """
    path = "../../data/raw/box/"
    l=os.listdir(path)
    files=[x.split('.')[0] for x in l]
    files.sort()
    all_files = files[1:]

    list_of_graphs = []

    r"""to create train,test,val data"""
    files = all_files.copy()
    random.shuffle(files)

    r"""Resulting in 500 receipts for training, 63 receipts for validation, and 63 for testing."""
    training,testing,validating = files[:500],files[500:563],files[563:]
    for file in all_files:    
        connect = Grapher(file)
        G,_,_ = connect.graph_formation()
        if G is None:
            continue
        df = connect.relative_distance() 
        individual_data = from_networkx(G)
        feature_cols = ['rd_b', 'rd_r', 'rd_t', 'rd_l','line_number',\
                'n_upper', 'n_alpha', 'n_spaces', 'n_numeric','n_special']

        features = torch.tensor(df[feature_cols].values.astype(np.float32))

        for col in df.columns:
            try:
                df[col] = df[col].str.strip()
            except AttributeError:
                pass

        df['labels'] = df['labels'].fillna('undefined')
        df.loc[df['labels'] == 'company', 'num_labels'] = 1
        df.loc[df['labels'] == 'address', 'num_labels'] = 2
        df.loc[df['labels'] == 'invoice', 'num_labels'] = 3
        df.loc[df['labels'] == 'date', 'num_labels'] = 4
        df.loc[df['labels'] == 'total', 'num_labels'] = 5
        df.loc[df['labels'] == 'undefined', 'num_labels'] = 6
 
        assert df['num_labels'].isnull().values.any() == False, f'labeling error! Invalid label(s) present in {file}.csv'
        labels = torch.tensor(df['num_labels'].values.astype(np.int))
        text = df['Object'].values

        individual_data.x = features
        individual_data.y = labels
        individual_data.text = text

        r"""Create masks"""
        if file in training:
            individual_data.train_mask = torch.tensor([True] * df.shape[0])
            individual_data.val_mask = torch.tensor([False] * df.shape[0])
            individual_data.test_mask = torch.tensor([False] * df.shape[0])
        elif file in validating:
            individual_data.train_mask = torch.tensor([False] * df.shape[0])
            individual_data.val_mask = torch.tensor([True] * df.shape[0])
            individual_data.test_mask = torch.tensor([False] * df.shape[0])
        else:
            individual_data.train_mask = torch.tensor([False] * df.shape[0])
            individual_data.val_mask = torch.tensor([False] * df.shape[0])
            individual_data.test_mask = torch.tensor([True] * df.shape[0])
        print(f'{file} ---> Success')
        list_of_graphs.append(individual_data)

    data = torch_geometric.data.Batch.from_data_list(list_of_graphs)

    # printResultTable(data.text[0], data.y.numpy(), data.y.numpy())

    data.edge_attr = None 
    save_path = "../../data/processed/"  
    torch.save(data, save_path +'data_withtexts2.dataset')
    print('Data is saved!')

field_dict = {
    1 : "company",
    2 : "address",
    3 : "invoice",
    4 : "date",
    5: "total",
    6: "undefined"
}
def printResultTable(texts, labelNums, predictNums):
    labelledLabels = convertNumsToLabels(labelNums)
    predictedLabels = convertNumsToLabels(predictNums)

    print("-" * 100)
    printRow("Label Text", "Labelled Data", "Predicted Data")
    print("-" * 100)

    idx = 0
    while idx < len(texts) and idx < len(labelledLabels):
        text = texts[idx]
        labelledLabel = labelledLabels[idx]
        predictedLabel = predictedLabels[idx]


        printRow(text, labelledLabel, predictedLabel)
        idx += 1

    print("-" * 100)

def convertNumsToLabels(labelNums):
    result = []

    for num in labelNums:
        labelStr = field_dict[num]
        result.append(labelStr)
    
    return result

def printRow(text, data1, data2):
    charNum = 40
    data1Num = 20

    spacesToAdd = charNum - len(text)
    if spacesToAdd > 0:
        text += " " * spacesToAdd

    spacesToAdd = data1Num - len(data1)
    if spacesToAdd > 0:
        data1 += " " * spacesToAdd

    spacesToAdd = data1Num - len(data2)
    if spacesToAdd > 0:
        data2 += " " * spacesToAdd
    
    print(text, "   |   ", data1, "   |   ", data2)

if __name__ == "__main__":
    get_data()
