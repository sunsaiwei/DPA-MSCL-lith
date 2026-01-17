import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt


class MultiView_Dataset(Dataset):
    def __init__(self, dataset_view1, dataset_view2, dataset_view3):

        assert len(dataset_view1) == len(dataset_view2) == len(dataset_view3), \
            "not equals"
        self.dataset1 = dataset_view1
        self.dataset2 = dataset_view2
        self.dataset3 = dataset_view3

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        data1, label1 = self.dataset1[idx]
        data2, label2 = self.dataset2[idx]
        data3, label3 = self.dataset3[idx]

        label = label1

        return (data1, data2, data3), label

class MultiScaleDataset(Dataset):

    def __init__(self, data_scale_features, labels):
        self.data_by_scale = data_scale_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return tuple(data[index] for data in self.data_by_scale), self.labels[index]


def convert_to_tensor(data_list):
    return [torch.tensor(data, dtype=torch.float32) for data in data_list]



class MultiScale_Dataset(Dataset):
    def __init__(self, data_scale_features, labels):
        self.data_by_scale = data_scale_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return tuple(data[index] for data in self.data_by_scale), self.labels[index]

def prepare_multiscale1_data(features):

    scales = [1, 3, 5]
    new_data_scale_features = [[] for _ in scales]
    for i in range(len(features)):
        new_data_scale_features[0].append(features[i])


        if i == 0:
            new_data_scale_features[1].append(np.vstack([np.zeros(features.shape[1]), features[i], features[min(i + 1, len(features) - 1)]]))
        elif i == len(features) - 1:
            new_data_scale_features[1].append(np.vstack([features[max(i - 1, 0)], features[i], np.zeros(features.shape[1])]))
        else:
            new_data_scale_features[1].append(np.vstack([features[i - 1], features[i], features[i + 1]]))


        left_padding_2 = np.zeros(features.shape[1]) if i - 2 < 0 else features[i - 2]
        left_padding_1 = np.zeros(features.shape[1]) if i - 1 < 0 else features[i - 1]
        right_padding_1 = np.zeros(features.shape[1]) if i + 1 >= len(features) else features[i + 1]
        right_padding_2 = np.zeros(features.shape[1]) if i + 2 >= len(features) else features[i + 2]
        new_data_scale_features[2].append(
            np.vstack([left_padding_2, left_padding_1, features[i], right_padding_1, right_padding_2]))

    new_data_by_scale = convert_to_tensor(new_data_scale_features)
    return new_data_by_scale

def prepare_multiscale2_data(features):
    scales = [3, 5, 7]
    new_data_scale_features = [[] for _ in scales]
    num_samples, num_features = features.shape

    for i in range(num_samples):

        if i == 0:
            scale3_data = np.vstack([
                np.zeros(num_features),
                features[i],
                features[min(i+1, num_samples-1)]
            ])
        elif i == num_samples - 1:
            scale3_data = np.vstack([
                features[max(i-1, 0)],
                features[i],
                np.zeros(num_features)
            ])
        else:
            scale3_data = np.vstack([
                features[i-1],
                features[i],
                features[i+1]
            ])
        new_data_scale_features[0].append(scale3_data)


        left2 = features[i-2] if i-2 >=0 else np.zeros(num_features)
        left1 = features[i-1] if i-1 >=0 else np.zeros(num_features)
        right1 = features[i+1] if i+1 < num_samples else np.zeros(num_features)
        right2 = features[i+2] if i+2 < num_samples else np.zeros(num_features)
        scale5_data = np.vstack([left2, left1, features[i], right1, right2])
        new_data_scale_features[1].append(scale5_data)


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
        new_data_scale_features[2].append(scale7_data)

    new_data_by_scale = convert_to_tensor(new_data_scale_features)
    return new_data_by_scale

def generate_multiview_data(features, labels, test_size, random_state):
    X_multiview_train, X_multiview_test = [], []
    X_train1, X_test1, y_train1, y_test1 = [], [], [], []
    X_train2, X_test2, y_train2, y_test2 = [], [], [], []
    new_data_by_scale1 = prepare_multiscale1_data(features)
    new_data_by_scale2 = prepare_multiscale2_data(features)
    k = 1
    j = 3
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
    k = 1
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
    j = 3
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


def generate_multiscale_data(data, labels, args):
    new_data_by_scale = prepare_multiscale_data(data)
    j = 1
    X_train, X_test, y_train, y_test = [], [], [], []
    train_test_data = []
    # 将输入的多尺度数据进行预处理，并划分成训练集和测试集
    for i in range(len(new_data_by_scale)):
        # 把每个尺度的数据重塑为一个二维数组以便于标准化
        reshaped_data = new_data_by_scale[i] \
            .reshape(new_data_by_scale[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data_reshaped = scaler.fit_transform(reshaped_data)
        scaled_data_reshaped = torch.FloatTensor(scaled_data_reshaped)
        # 将每个尺度的数据变换为维度为(N,C,L)的张量，其中N为样本数，C为通道数，L为特征长度
        if i == 0:
            new_data_by_scale[i] = new_data_by_scale[i].unsqueeze(
                1)  # 对于尺度1在第二个维度上增加一个大小为1的维度，因为unsqueeze()函数新增维度的大小固定为1
        else:
            new_data_by_scale[i] = scaled_data_reshaped. \
                reshape(new_data_by_scale[i].size(0), j, new_data_by_scale[i].size(2))
        j = j + 2
        # 将每个尺度的数据划分为训练集和测试集
        X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(new_data_by_scale[i], labels,
                                                                                test_size=args.test_size,
                                                                                random_state=args.random_state)
        X_train.append(X_train_data)
        X_test.append(X_test_data)
        y_train.append(y_train_data)
        y_test.append(y_test_data)
    train_test_data.append(X_train)
    train_test_data.append(X_test)
    train_test_data.append(y_train)
    train_test_data.append(y_test)
    return train_test_data

def prepare_multiscale_data(features):

    scales = [1, 3, 5]

    new_data_scale_features = [[] for _ in scales]
    for i in range(len(features)):

        new_data_scale_features[0].append(features[i])

        if i == 0:
            new_data_scale_features[1].append(np.vstack([np.zeros(features.shape[1]), features[i], features[min(i + 1, len(features) - 1)]]))
        elif i == len(features) - 1:
            new_data_scale_features[1].append(np.vstack([features[max(i - 1, 0)], features[i], np.zeros(features.shape[1])]))
        else:
            new_data_scale_features[1].append(np.vstack([features[i - 1], features[i], features[i + 1]]))

        left_padding_2 = np.zeros(features.shape[1]) if i - 2 < 0 else features[i - 2]
        left_padding_1 = np.zeros(features.shape[1]) if i - 1 < 0 else features[i - 1]
        right_padding_1 = np.zeros(features.shape[1]) if i + 1 >= len(features) else features[i + 1]
        right_padding_2 = np.zeros(features.shape[1]) if i + 2 >= len(features) else features[i + 2]
        new_data_scale_features[2].append(
            np.vstack([left_padding_2, left_padding_1, features[i], right_padding_1, right_padding_2]))

    new_data_by_scale = convert_to_tensor(new_data_scale_features)
    return new_data_by_scale


def generate_multiscale_blind(features, labels):
    new_data_by_scale = prepare_multiscale_data(features)
    j=1
    for i in range(len(new_data_by_scale)):
        reshaped_data = new_data_by_scale[i] \
            .reshape(new_data_by_scale[i].size(0), -1)
        scaler = preprocessing.StandardScaler()
        scaled_data_reshaped = scaler.fit_transform(reshaped_data)
        scaled_data_reshaped = torch.FloatTensor(scaled_data_reshaped)
        if i == 0:
            new_data_by_scale[i] = new_data_by_scale[i].unsqueeze(1)
        else:
            new_data_by_scale[i] = scaled_data_reshaped. \
                reshape(new_data_by_scale[i].size(0), j, new_data_by_scale[i].size(2))
        j = j + 2
    return new_data_by_scale, labels





def get_cls_num_list(y_train, num_classes):
    class_data = [[] for _ in range(num_classes)]

    for i in range(len(y_train)):
        y = y_train[i]
        class_data[y].append(i)

    cls_num_list = [len(class_data[i]) for i in range(num_classes)]
    return cls_num_list

def warmup_lr_scheduler(args, optimizer, current_epoch):
    if current_epoch <= args.warm_epochs:
        lr = args.warmup_from + (args.warmup_to - args.warmup_from) * \
             current_epoch / args.warm_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer


def write_file(file_path,predicted):

    with open(file_path, "w") as file:
        for prediction in predicted:
            file.write(f"{prediction.item()}\n")
        file.close()

def get_confusion_matrix(trues, preds):

    conf_matrix = confusion_matrix(trues, preds)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_matrix_normalized = conf_matrix / row_sums
    return np.round(conf_matrix_normalized * 100, 2)

def save_metrics_plot(args, accuracy, precision, recall, f1, save_dir, model_name):

    plt.figure(figsize=(8, 4))
    ax = plt.subplot(111)

    ax.axis('off')

    cell_text = [
        [f"{accuracy:.4f}"],
        [f"{precision:.4f}"],
        [f"{recall:.4f}"],
        [f"{f1:.4f}"]
    ]

    table = ax.table(
        cellText=cell_text,
        rowLabels=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        colLabels=['Value'],
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    plt.title(f'Evaluation Metrics - {model_name} on {args.dataset}', fontsize=14, pad=20)

    plt.tight_layout()
    save_path = save_dir + f'{model_name}.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()







class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def warmup_lr_scheduler(args, optimizer, current_epoch):
    if current_epoch <= args.warm_epochs:
        lr = args.warmup_from + (args.warmup_to - args.warmup_from) * \
             current_epoch / args.warm_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer
