
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from utils.utils import *
import pandas as pd


def get_Hugoton_Panoma_MultiViewData(path, args):
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)
    data_frame = data_frame[["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]

    X = data_frame.drop("Facies", axis=1).values
    y = data_frame["Facies"].values - 1
    y = torch.LongTensor(y)
    X_multiview_train, X_multiview_test, y_multiview_train, y_multiview_test = generate_multiview_data(X, y, args.test_size, args.random_state)

    raw_train_dataset = TensorDataset(X_multiview_train[0], y_multiview_train)
    raw_test_dataset = TensorDataset(X_multiview_test[0], y_multiview_test)

    multi1_train_dataset = MultiScale_Dataset(X_multiview_train[1], y_multiview_train)
    multi1_test_dataset = MultiScale_Dataset(X_multiview_test[1], y_multiview_test)

    multi2_train_dataset = MultiScale_Dataset(X_multiview_train[2], y_multiview_train)
    multi2_test_dataset = MultiScale_Dataset(X_multiview_test[2], y_multiview_test)

    multiview_train_dataset = MultiView_Dataset(raw_train_dataset, multi1_train_dataset, multi2_train_dataset)
    multiview_test_dataset = MultiView_Dataset(raw_test_dataset, multi1_test_dataset, multi2_test_dataset)

    train_dataloader = DataLoader(
        multiview_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_dataloader = DataLoader(
        multiview_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return X_multiview_train, X_multiview_test, train_dataloader, y_multiview_train, y_multiview_test, test_dataloader

def get_blind_HP_MultiViewData(path, blind_well1, blind_well2, args):


    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)


    train_frame = data_frame[(data_frame['Well Name'] != blind_well1) &
                             (data_frame['Well Name'] != blind_well2)]
    blind_frame = data_frame[(data_frame['Well Name'] == blind_well1) |
                            (data_frame['Well Name'] == blind_well2)]

    train_data = train_frame.loc[:,
                 ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]
    X_train = train_data.drop(labels='Facies', axis=1).values

    y_train = train_frame['Facies'].values - 1
    y_train = torch.LongTensor(y_train)


    blind_data = blind_frame.loc[:,
                 ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]]
    X_blind = blind_data.drop(labels='Facies', axis=1).values

    y_blind = blind_frame['Facies'].values - 1
    y_blind = torch.LongTensor(y_blind)



    X_multiview_train, y_multiview_train = generate_multiview_blind(X_train, y_train)
    X_multiview_blind, y_multiview_blind = generate_multiview_blind(X_blind, y_blind)

    raw_train_dataset = TensorDataset(X_multiview_train[0], y_multiview_train)
    raw_test_dataset = TensorDataset(X_multiview_blind[0], y_multiview_blind)

    multi1_train_dataset = MultiScale_Dataset(X_multiview_train[1], y_multiview_train)
    multi1_test_dataset = MultiScale_Dataset(X_multiview_blind[1], y_multiview_blind)

    multi2_train_dataset = MultiScale_Dataset(X_multiview_train[2], y_multiview_train)
    multi2_test_dataset = MultiScale_Dataset(X_multiview_blind[2], y_multiview_blind)

    multiview_train_dataset = MultiView_Dataset(raw_train_dataset, multi1_train_dataset, multi2_train_dataset)
    multiview_blind_dataset = MultiView_Dataset(raw_test_dataset, multi1_test_dataset, multi2_test_dataset)

    train_dataloader = DataLoader(
        multiview_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_dataloader = DataLoader(
        multiview_blind_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return X_multiview_train, X_multiview_blind, train_dataloader, y_multiview_train, y_multiview_blind, test_dataloader

def get_daqing_MultiViewData(path, args):
    data_frame = pd.read_excel(path)
    data_frame.dropna(inplace=True)
    data_frame = data_frame[["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "POR_index", "Ish", "Face"]]


    X = data_frame.drop("Face", axis=1).values
    y = data_frame["Face"].values
    y = torch.LongTensor(y)
    X_multiview_train, X_multiview_test, y_multiview_train, y_multiview_test = generate_multiview_data(X, y, args.test_size, args.random_state)

    raw_train_dataset = TensorDataset(X_multiview_train[0], y_multiview_train)
    raw_test_dataset = TensorDataset(X_multiview_test[0], y_multiview_test)

    multi1_train_dataset = MultiScale_Dataset(X_multiview_train[1], y_multiview_train)
    multi1_test_dataset = MultiScale_Dataset(X_multiview_test[1], y_multiview_test)

    multi2_train_dataset = MultiScale_Dataset(X_multiview_train[2], y_multiview_train)
    multi2_test_dataset = MultiScale_Dataset(X_multiview_test[2], y_multiview_test)

    multiview_train_dataset = MultiView_Dataset(raw_train_dataset, multi1_train_dataset, multi2_train_dataset)
    multiview_test_dataset = MultiView_Dataset(raw_test_dataset, multi1_test_dataset, multi2_test_dataset)

    train_dataloader = DataLoader(
        multiview_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_dataloader = DataLoader(
        multiview_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return X_multiview_train, X_multiview_test, train_dataloader, y_multiview_train, y_multiview_test, test_dataloader

def get_blind_daqing_MultiViewData(path1, path2, args):



    train_frame = pd.read_excel(path1)
    train_frame.dropna(inplace=True)
    blind_frame = pd.read_excel(path2)
    blind_frame.dropna(inplace=True)

    train_data = train_frame.loc[:,
                 ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "POR_index", "Ish", "Face"]]
    X_train = train_data.drop(labels='Face', axis=1).values

    y_train = train_frame['Face'].values
    y_train = torch.LongTensor(y_train)


    blind_data = blind_frame.loc[:,
                 ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "POR_index", "Ish", "Face"]]
    X_blind = blind_data.drop(labels='Face', axis=1).values

    y_blind = blind_frame['Face'].values
    y_blind = torch.LongTensor(y_blind)



    X_multiview_train, y_multiview_train = generate_multiview_blind(X_train, y_train)
    X_multiview_blind, y_multiview_blind = generate_multiview_blind(X_blind, y_blind)

    raw_train_dataset = TensorDataset(X_multiview_train[0], y_multiview_train)
    raw_test_dataset = TensorDataset(X_multiview_blind[0], y_multiview_blind)

    multi1_train_dataset = MultiScale_Dataset(X_multiview_train[1], y_multiview_train)
    multi1_test_dataset = MultiScale_Dataset(X_multiview_blind[1], y_multiview_blind)

    multi2_train_dataset = MultiScale_Dataset(X_multiview_train[2], y_multiview_train)
    multi2_test_dataset = MultiScale_Dataset(X_multiview_blind[2], y_multiview_blind)

    multiview_train_dataset = MultiView_Dataset(raw_train_dataset, multi1_train_dataset, multi2_train_dataset)
    multiview_blind_dataset = MultiView_Dataset(raw_test_dataset, multi1_test_dataset, multi2_test_dataset)

    train_dataloader = DataLoader(
        multiview_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_dataloader = DataLoader(
        multiview_blind_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return X_multiview_train, X_multiview_blind, train_dataloader, y_multiview_train, y_multiview_blind, test_dataloader


def generate_multiview_data(features, labels, test_size, random_state):

    X_multiview_train, X_multiview_test = [], []
    X_train1, X_test1, y_train1, y_test1 = [], [], [], []
    X_train2, X_test2, y_train2, y_test2 = [], [], [], []
    new_data_by_scale1 = prepare_multiscale1_data(features)
    new_data_by_scale2 = prepare_multiscale2_data(features)
    k = 1
    j = 5
    X_train, X_test, y_train, y_test = [], [], [], []
    train_test_data = []
    features_tensor = torch.tensor(features, dtype=torch.float32)
    X_raw_train, X_raw_test, y_raw_train, y_raw_test = (
        train_test_split(features_tensor, labels, test_size=test_size, random_state=random_state))
    X_multiview_train.append(X_raw_train)
    X_multiview_test.append(X_raw_test)
    y_multiview_train = y_raw_train
    y_multiview_test =y_raw_test

    for i in range(len(new_data_by_scale1)):
        reshaped_data = new_data_by_scale1[i] \
            .reshape(new_data_by_scale1[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data_reshaped = scaler.fit_transform(reshaped_data)
        scaled_data_reshaped = torch.FloatTensor(scaled_data_reshaped)
        if i == 0:
            new_data_by_scale1[i] = new_data_by_scale1[i].unsqueeze(1)
        else:
            new_data_by_scale1[i] = scaled_data_reshaped. \
                reshape(new_data_by_scale1[i].size(0), k, new_data_by_scale1[i].size(2))
        k = k + 2

        X_multiscale_train, X_multiscale_test, y_multiscale_train, y_multiscale_test = (
            train_test_split(new_data_by_scale1[i], labels, test_size=test_size, random_state=random_state))
        X_train1.append(X_multiscale_train)
        X_test1.append(X_multiscale_test)
        y_train1.append(y_multiscale_train)
        y_test1.append(y_multiscale_test)
    X_multiview_train.append(X_train1)
    X_multiview_test.append(X_test1)


    for i in range(len(new_data_by_scale2)):
        reshaped_data = new_data_by_scale2[i] \
            .reshape(new_data_by_scale2[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data_reshaped = scaler.fit_transform(reshaped_data)
        scaled_data_reshaped = torch.FloatTensor(scaled_data_reshaped)

        new_data_by_scale2[i] = scaled_data_reshaped. \
                reshape(new_data_by_scale2[i].size(0), j, new_data_by_scale2[i].size(2))
        j = j + 2

        X_multiscale_train, X_multiscale_test, y_multiscale_train, y_multiscale_test = (
            train_test_split(new_data_by_scale2[i], labels, test_size=test_size, random_state=random_state)
        )
        X_train2.append(X_multiscale_train)
        X_test2.append(X_multiscale_test)
        y_train2.append(y_multiscale_train)
        y_test2.append(y_multiscale_test)
    X_multiview_train.append(X_train2)
    X_multiview_test.append(X_test2)


    return X_multiview_train, X_multiview_test, y_multiview_train, y_multiview_test

def generate_multiview_blind(features, labels):

    new_data_by_scale1 = prepare_multiscale1_data(features)
    new_data_by_scale2 = prepare_multiscale2_data(features)

    multiview_data = []

    scaler_raw = preprocessing.StandardScaler()
    X_raw = scaler_raw.fit_transform(features)
    X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32)
    multiview_data.append(X_raw_tensor)

    view2_data = []
    k = 1  # 初始通道数
    for i in range(len(new_data_by_scale1)):
        reshaped_data = new_data_by_scale1[i].reshape(new_data_by_scale1[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(reshaped_data)

        scaled_tensor = torch.FloatTensor(scaled_data)
        if i == 0:
            scaled_tensor = scaled_tensor.unsqueeze(1)
        else:
            scaled_tensor = scaled_tensor.reshape(
                new_data_by_scale1[i].size(0),
                k,
                new_data_by_scale1[i].size(2)
            )
        k += 2

        view2_data.append(scaled_tensor)
    multiview_data.append(view2_data)

    view3_data = []
    j = 5  # 初始通道数
    for i in range(len(new_data_by_scale2)):
        reshaped_data = new_data_by_scale2[i].reshape(new_data_by_scale2[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(reshaped_data)

        scaled_tensor = torch.FloatTensor(scaled_data).reshape(
            new_data_by_scale2[i].size(0),
            j,
            new_data_by_scale2[i].size(2)
        )
        j += 2

        view3_data.append(scaled_tensor)
    multiview_data.append(view3_data)

    return multiview_data, labels


def prepare_multiscale1_data(features):
    
    scales = [1, 3]
    new_data_scale_features = [[] for _ in scales]
    for i in range(len(features)):
        new_data_scale_features[0].append(features[i])

        if i == 0:
            new_data_scale_features[1].append(np.vstack([np.zeros(features.shape[1]), features[i], features[min(i + 1, len(features) - 1)]]))
        elif i == len(features) - 1:
            new_data_scale_features[1].append(np.vstack([features[max(i - 1, 0)], features[i], np.zeros(features.shape[1])]))
        else:
            new_data_scale_features[1].append(np.vstack([features[i - 1], features[i], features[i + 1]]))


    new_data_by_scale = convert_to_tensor(new_data_scale_features)
    return new_data_by_scale

def prepare_multiscale2_data(features):
    
    scales = [5, 7]
    new_data_scale_features = [[] for _ in scales]
    num_samples, num_features = features.shape

    for i in range(num_samples):

        left2 = features[i-2] if i-2 >=0 else np.zeros(num_features)
        left1 = features[i-1] if i-1 >=0 else np.zeros(num_features)
        right1 = features[i+1] if i+1 < num_samples else np.zeros(num_features)
        right2 = features[i+2] if i+2 < num_samples else np.zeros(num_features)
        scale5_data = np.vstack([left2, left1, features[i], right1, right2])
        new_data_scale_features[0].append(scale5_data)

        left3 = features[i-3] if i-3 >=0 else np.zeros(num_features)
        left2 = features[i-2] if i-2 >=0 else np.zeros(num_features)
        left1 = features[i-1] if i-1 >=0 else np.zeros(num_features)
        right1 = features[i+1] if i+1 < num_samples else np.zeros(num_features)
        right2 = features[i+2] if i+2 < num_samples else np.zeros(num_features)
        right3 = features[i+3] if i+3 < num_samples else np.zeros(num_features)
        scale7_data = np.vstack([
            left3, left2, left1,
            features[i],
            right1, right2, right3
        ])
        new_data_scale_features[1].append(scale7_data)

    new_data_by_scale = convert_to_tensor(new_data_scale_features)
    return new_data_by_scale