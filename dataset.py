import torch
from torch_geometric.data import Data, Dataset, DataLoader
import pandas as pd
import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore")


def data_processing(data):
    # AQI = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    AQI = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2',
           'PM25_AQI', 'PM10_AQI', 'NO2_AQI', 'CO_AQI', 'O3_AQI', 'SO2_AQI', 'AQI']
    station_list = data['station_id'].unique()
    dataset = np.zeros([len(station_list), len(AQI), 24 * 731])
    for i in range(len(AQI)):
        aqi = data[AQI[i]].values.reshape(len(station_list), -1)
        dataset[:, i, :] = copy.deepcopy(aqi)
    return dataset.transpose(2, 0, 1)

def get_city_label(city_num,total_city_num):
    label = np.zeros(total_city_num)
    label[city_num] = 1
    label = torch.tensor(label)
    label = label.to(torch.float32)
    return label
    
def get_data(dataset_path, history_len, prediction_len, is_src):
    # train,val,test数据集划分
    def dataset_split(data_x, data_y, is_src):
        if is_src == 1:
            tem_val = np.arange(int(data_x.shape[0] * 12 / 24), int(data_x.shape[0] * 14 / 24))
            tem_test = np.hstack((np.arange(int(data_x.shape[0] * 12 / 24), int(data_x.shape[0] * 13 / 24)),
                          np.arange(int(data_x.shape[0] * 15 / 24), int(data_x.shape[0] * 16 / 24)),
                          np.arange(int(data_x.shape[0] * 18 / 24), int(data_x.shape[0] * 19 / 24)),
                          np.arange(int(data_x.shape[0] * 21 / 24), int(data_x.shape[0] * 22 / 24))))
            tem_train = np.arange(int(data_x.shape[0] * 12 / 24))
            val_src, val_tar = data_x[tem_val], data_y[tem_val]
            test_src, test_tar = data_x[tem_test], data_y[tem_test]
            train_src, train_tar = data_x[tem_train], data_y[tem_train]

        else:
            tem_train = np.arange(int(data_x.shape[0] * 365 / 731), int(data_x.shape[0] * (365+10) / 731))
            tem_val = np.arange(int(data_x.shape[0] * 13 / 24), int(data_x.shape[0] * 14 / 24))
            tem_test = np.hstack((np.arange(int(data_x.shape[0] * 14 / 24), int(data_x.shape[0] * 15 / 24)),
                        np.arange(int(data_x.shape[0] * 17 / 24), int(data_x.shape[0] * 18 / 24)),
                        np.arange(int(data_x.shape[0] * 20 / 24), int(data_x.shape[0] * 21 / 24)),
                        np.arange(int(data_x.shape[0] * 23 / 24), int(data_x.shape[0] * 24 / 24))))
            val_src, val_tar = data_x[tem_val], data_y[tem_val]
            test_src, test_tar = data_x[tem_test], data_y[tem_test]
            train_src, train_tar = data_x[tem_train], data_y[tem_train]

        return train_src, train_tar, val_src, val_tar, test_src, test_tar

    dataset = pd.read_csv(dataset_path)
    dataset = data_processing(dataset)
    raw_dataset = copy.deepcopy(dataset[:, :, -1:])
    aqi_dataset = copy.deepcopy(dataset[:, :, -1:])
    raw_data = raw_dataset.transpose(1, 0, 2)
    aqi_data = aqi_dataset.transpose(1, 0, 2)

    data_x = np.zeros([raw_data.shape[1] - history_len - prediction_len, raw_data.shape[0], history_len, raw_data.shape[-1]])
    data_y = np.zeros([raw_data.shape[1] - history_len - prediction_len, raw_data.shape[0], prediction_len, raw_data.shape[-1]])
    for i in range(raw_data.shape[1] - history_len - prediction_len):
        data_x[i, :, :, :] = copy.deepcopy(raw_data[:, i:i + history_len, :])
        data_y[i, :, :, :] = copy.deepcopy(aqi_data[:, i + history_len:i + history_len + prediction_len, :])
    train_src, train_tar, val_src, val_tar, test_src, test_tar = dataset_split(data_x, data_y, is_src)
    # 数据归一化
    train_src_mean, train_src_std = np.mean(train_src, axis=(0, 1, 2)), np.std(train_src, axis=(0, 1, 2))
    val_src_mean, val_src_std = np.mean(val_src, axis=(0, 1, 2)), np.std(val_src, axis=(0, 1, 2))
    test_src_mean, test_src_std = np.mean(test_src, axis=(0, 1, 2)), np.std(test_src, axis=(0, 1, 2))

    train_src = train_src - train_src_mean.reshape(1, 1, 1, -1)
    train_src = train_src / train_src_std.reshape(1, 1, 1, -1)
    val_src = val_src - val_src_mean.reshape(1, 1, 1, -1)
    val_src = val_src / val_src_std.reshape(1, 1, 1, -1)
    test_src = test_src - test_src_mean.reshape(1, 1, 1, -1)
    test_src = test_src / test_src_std.reshape(1, 1, 1, -1)

    return train_src, train_tar, val_src, val_tar, test_src, test_tar


def edge_index_func(matrix_path):
    # print("In edge index function")
    a, b, c = [], [], []
    matrix = np.load(matrix_path)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # if(matrix[i][j] == 1 and i != j):
            if (matrix[i][j] > 0):
                a.append(i)
                b.append(j)
                c.append(matrix[i][j])
    edge = [a, b]
    edge_index = torch.tensor(edge, dtype=torch.long)
    edge_attr = torch.tensor(c, dtype=torch.float32)
    return edge_index, edge_attr.unsqueeze(1)


class myDataset(Dataset):
    def __init__(self, dataset_x, dataset_y, city_num_source):

        '''
        : data.x shape is [batch_size, node_num, his_num, message_dim]
        : data.y shape is [batch_size, node_num, pred_num]
        : data.edge_index constructed for torch_geometric
        : data.edge_attr  constructed for torch_geometric
        '''
        self.adjacency_matrix_path = r'/home/zzy6anl/cgy/DANN/edge/' + 'adj_mat_{}.npy'.format(city_num_source)
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.time_intervals, self.sensor_num, self.history_len, self.feature_dim = self.dataset_x.shape
        self.time_intervals, self.sensor_num, self.predict_len, self.feature_dim = self.dataset_y.shape
        self.edge_index, self.edge_attr = edge_index_func(self.adjacency_matrix_path)
        self.label = get_city_label(city_num_source,6)

    def __getitem__(self, index):
        '''
        data_source = Data(num_nodes=self.sensor_num)
        data_source.x = torch.tensor(self.dataset[index: index + self.history_len][np.newaxis, :], dtype=torch.float)
        # print('data.x shape is', data_source.x.shape)
        data_source.y = torch.tensor(
            self.dataset[index + self.history_len: index + self.history_len + self.predict_len][np.newaxis, :],
            dtype=torch.float)
        print('index is', index)
        '''
        data_source = Data(num_nodes=self.sensor_num)
        data_source.x = torch.tensor(self.dataset_x[index][np.newaxis, :], dtype=torch.float)
        data_source.y = torch.tensor(self.dataset_y[index][np.newaxis, :], dtype=torch.float)
        data_source.edge_index = self.edge_index
        data_source.edge_attr = self.edge_attr
        data_source.label = self.label
        return data_source

    def __len__(self):
        data_length = self.time_intervals
        return data_length

class myDataset_combined(Dataset):
    def __init__(self, dataset_x_source, dataset_y_source, city_num_source,
                        dataset_x_target, dataset_y_target, city_num_target):

        '''
        : data.x shape is [batch_size, node_num, his_num, message_dim]
        : data.y shape is [batch_size, node_num, pred_num]
        : data.edge_index constructed for torch_geometric
        : data.edge_attr  constructed for torch_geometric
        '''
        # big city ------------------------------------------------------------------------------
        self.adjacency_matrix_path_source = r'/home/zzy6anl/cgy/DANN/edge/' + 'adj_mat_{}.npy'.format(city_num_source)
        self.dataset_x_source = dataset_x_source
        self.dataset_y_source = dataset_y_source
        self.time_intervals_source, self.sensor_num_source, self.history_len, self.feature_dim = self.dataset_x_source.shape
        self.predict_len = self.dataset_y_source.shape[2]
        self.edge_index_source, self.edge_attr_source = edge_index_func(self.adjacency_matrix_path_source)
        self.label_source = get_city_label(0,2)
        # small city ----------------------------------------------------------------------------
        self.adjacency_matrix_path_target = r'/home/zzy6anl/cgy/DANN/edge/' + 'adj_mat_{}.npy'.format(city_num_target)
        self.dataset_x_target = dataset_x_target
        self.dataset_y_target = dataset_y_target
        self.time_intervals_target, self.sensor_num_target, self.history_len, self.feature_dim = self.dataset_x_target.shape
        self.predict_len = self.dataset_y_target.shape[2]
        self.edge_index_target, self.edge_attr_target = edge_index_func(self.adjacency_matrix_path_target)
        self.label_target = get_city_label(1,2)
        # dataset -------------------------------------------------------------------------------
        self.time_intervals = self.time_intervals_source + self.time_intervals_target

    def __getitem__(self, index):
        index_source = int(index/self.time_intervals * self.time_intervals_source)
        data_source = Data(num_nodes=self.sensor_num_source)
        data_source.x = torch.tensor(self.dataset_x_source[index_source][np.newaxis, :], dtype=torch.float)
        data_source.y = torch.tensor(self.dataset_y_source[index_source][np.newaxis, :], dtype=torch.float)
        data_source.edge_index = self.edge_index_source
        data_source.edge_attr = self.edge_attr_source
        data_source.label = self.label_source

        index_target = int(index/self.time_intervals * self.time_intervals_target)
        data_target = Data(num_nodes=self.sensor_num_target)
        data_target.x = torch.tensor(self.dataset_x_target[index_target][np.newaxis, :], dtype=torch.float)
        data_target.y = torch.tensor(self.dataset_y_target[index_target][np.newaxis, :], dtype=torch.float)
        data_target.edge_index = self.edge_index_target
        data_target.edge_attr = self.edge_attr_target
        data_target.label = self.label_target
        #data_source = Data()
        # data_source.x_source = torch.tensor(self.dataset_x_source[index_source][np.newaxis, :], dtype=torch.float)
        # data_source.y_source = torch.tensor(self.dataset_y_source[index_source][np.newaxis, :], dtype=torch.float)
        # data_source.edge_index_source = self.edge_index_source
        # data_source.edge_attr_source = self.edge_attr_source
        # data_source.label_source = self.label_source
    
        # data_source = Data(num_nodes=self.sensor_num_target)
        # index_target = int(index/self.time_intervals * self.time_intervals_target)
        # data_source.x_target = torch.tensor(self.dataset_x_target[index_target][np.newaxis, :], dtype=torch.float)
        # data_source.y_target = torch.tensor(self.dataset_y_target[index_target][np.newaxis, :], dtype=torch.float)
        # data_source.edge_index_target = self.edge_index_target
        # data_source.edge_attr_target = self.edge_attr_target
        # data_source.label_target = self.label_target
        
        return data_source, data_target

    def __len__(self):
        data_length = self.time_intervals
        return data_length



