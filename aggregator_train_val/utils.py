import numpy as np
import yaml
from addict import Dict
import torch
import torch.nn.functional as F
from pathlib import Path
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def load_loggers(cfg):
    log_path = cfg.General.log_path
    Path(log_path).mkdir(exist_ok=True, parents=True)
    cfg.log_path = Path(log_path) / f'{cfg.Model.exp_name}' 
    print(f'---->Log dir: {cfg.log_path}')

    tb_logger = pl_loggers.TensorBoardLogger(cfg.log_path, name='log', log_graph = True, default_hp_metric=False)
    csv_logger = pl_loggers.CSVLogger(cfg.log_path, name='log')
    return [tb_logger, csv_logger]


def load_callbacks(cfg):
    Mycallbacks = []
    # Make output path
    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    early_stop_callback = EarlyStopping(
        monitor='multi_acc',
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='max'
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train' :
        Mycallbacks.append(ModelCheckpoint(monitor = 'multi_acc',
                                         dirpath = str(cfg.log_path),
                                         filename = '{epoch:02d}-{multi_acc:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = 'max',
                                         save_weights_only = True))
    return Mycallbacks


def read_yaml(fpath: str)->dict:
    """
    Read yaml file.

    Parameters
    ----------
    fpath : string
        The path to the yaml file.

    Returns
    -------
    dict
        Dict storing the yaml file.
    """

    with open(fpath, "r") as file:
        yml = yaml.safe_load(file)
        
    return Dict(yml)


def label_vec_generator(label0, soft_labels, cls_count, label1=None, w0=1.0):
    aug_label = torch.zeros(cls_count)
    if soft_labels:
        label = w0 * label0 + (1 - w0) * label1 if label1 is not None else label0
        aug_label[:len(label)] = label
    else:
        aug_label[label0] = w0
        if label1 is not None:
            aug_label[label1] = 1 - w0
    return aug_label


def get_loc_representation(loc, loc_dict):
    location = torch.zeros(len(loc_dict))
    if loc in list(loc_dict.keys()):
        location[loc_dict[loc]] = 1
    else:
        pass
    return location


def get_positional_encoding(age, d_model):
    """
    获取年龄的正弦和余弦位置编码。
    
    参数：
    age: 年龄（标量）
    d_model: 编码的维度
    
    返回：
    年龄的d_model维度的编码向量
    """
    if isinstance(age, str):
        if age == 'unknown' or not age.isnumeric():
            pos_encoding = np.zeros(d_model)
            return torch.tensor(pos_encoding)
        else:
            age = float(age)
    elif isinstance(age, (int, float)):
        if np.isnan(age):
            pos_encoding = np.zeros(d_model)
            return torch.tensor(pos_encoding)
        else:
            age = float(age)

    position = np.arange(0, d_model, 2)
    div_term = np.exp(position * -np.log(10000.0) / d_model)
    pos_encoding = np.zeros(d_model)
    pos_encoding[0::2] = np.sin(age * div_term)
    pos_encoding[1::2] = np.cos(age * div_term)
    pos_encoding = torch.tensor(pos_encoding)
    return pos_encoding


def age_augmentation(age0, age1, w0, drop_prob, dim_age_embed):
    age0 = age0 if np.random.rand() < drop_prob else 'unknown'
    age1 = age1 if np.random.rand() < drop_prob else 'unknown'
    age0_representation = get_positional_encoding(age0, dim_age_embed)
    age1_representation = get_positional_encoding(age1, dim_age_embed)
    aug_age = w0 * age0_representation + (1 - w0) * age1_representation
    return aug_age.to(torch.float32)


def loc_augmentation(loc0, loc1, w0, drop_prob, loc_dict):
    loc0 = get_loc_representation(loc0, loc_dict)
    loc1 = get_loc_representation(loc1, loc_dict)
    loc0 = loc0 if np.random.rand() < drop_prob else torch.zeros_like(loc0)
    loc1 = loc1 if np.random.rand() < drop_prob else torch.zeros_like(loc1)
    aug_loc = w0 * loc0 + (1 - w0) * loc1
    return aug_loc.to(torch.float32)



def update_ema_variables(old_params, new_params, current_epoch):
    if current_epoch >=0 and current_epoch < 3:
        alpha = 0.5
    elif current_epoch >=3 and current_epoch < 9:
        alpha = 0.75
    elif current_epoch >=9:
        alpha = 0.99
    else:
        raise ValueError('Invalid epoch number.')
    # alpha = min(1 - 1 / (1+global_step), alpha)
    new_params = old_params * alpha + new_params * (1 - alpha)
    return new_params


def cross_entropy_torch(x, y):
    x_softmax = [F.softmax(x[i]) for i in range(len(x))]
    x_log = torch.tensor([torch.log(x_softmax[i][y[i]]) for i in range(len(y))])
    loss = - torch.sum(x_log) / len(y)
    return loss