import torch
import numpy as np
import scipy.sparse as sp
import random

from recbole.sampler import KGSampler
from recbole.utils import ModelType, set_color
from recbole.data import save_split_dataloaders, get_dataloader, create_samplers, getLogger
from recbole.data.dataloader import *


def generate_data(train_dataset, valid_dataset, test_dataset, config):
    if config['ptb_strategy']:
        train_dataset = generate_perturbed_dataset(train_dataset, strategy=config['ptb_strategy'], prop=config['ptb_prop'])
        valid_dataset = generate_perturbed_dataset(valid_dataset, strategy=config['ptb_strategy'], prop=config['ptb_prop'])

    return train_dataset, valid_dataset, test_dataset


def customized_data_preparation(config, dataset, save=False):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        save (bool, optional): If ``True``, it will call :func:`save_datasets` to save split dataset.
            Defaults to ``False``.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    model_type = config['MODEL_TYPE']
    built_datasets = dataset.build()

    train_dataset, valid_dataset, test_dataset = built_datasets
    train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

    # here
    train_dataset, valid_dataset, test_dataset = generate_data(train_dataset, valid_dataset, test_dataset, config)

    if model_type != ModelType.KNOWLEDGE:
        train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
    else:
        kg_sampler = KGSampler(dataset, config['train_neg_sample_args']['distribution'])
        train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, kg_sampler, shuffle=True)

    valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
    test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)

    if save:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    # here
    logger = getLogger()
    logger.info(dataset_info(train_data.dataset, 'train_data'))
    logger.info(dataset_info(valid_data.dataset, 'valid_data'))
    logger.info(dataset_info(test_data.dataset, 'test_data'))

    return train_data, valid_data, test_data


def dataset_info(dataset, name):
    info = [set_color(dataset.dataset_name + '\t' + name, 'pink')]
    if dataset.uid_field:
        info.extend([set_color('\nAverage actions of users', 'blue') + f': {dataset.avg_actions_of_users}'])
    if dataset.iid_field:
        info.extend([set_color('Average actions of items', 'blue') + f': {dataset.avg_actions_of_items}'])
    info.append(set_color('\nThe number of inters', 'blue') + f': {dataset.inter_num}')
    if dataset.uid_field and dataset.iid_field:
        info.append(set_color('The sparsity of the dataset', 'blue') + f': {dataset.sparsity * 100}%')
    return '\t'.join(info)


def sp2tensor(L):
    # covert norm_adj matrix to tensor
    L = sp.coo_matrix(L)
    row = L.row
    col = L.col
    i = torch.LongTensor([row, col])
    data = torch.FloatTensor(L.data)
    SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape)).coalesce()
    return SparseL


def del_tensor_ele(tens, index_list):
    k_array = tens.numpy()
    k_array = np.delete(k_array, index_list)
    return torch.from_numpy(k_array)


def generate_perturbed_dataset(dataset, strategy='replace', prop=0.05):
    inter_feat = dataset.inter_feat
    interaction_length = inter_feat.length
    interaction_dict = inter_feat.interaction
    interaction_dict_keys = list(interaction_dict.keys())

    user_id = interaction_dict_keys[0]
    item_id = interaction_dict_keys[1]

    userID_max = max(interaction_dict[user_id])
    itemID_max = max(interaction_dict[item_id])

    user_history_dict = {}
    item_history_dict = {}
    inter_list_user = interaction_dict[user_id].tolist()
    inter_list_item = interaction_dict[item_id].tolist()
    for u, i in zip(inter_list_user, inter_list_item):
        if u not in user_history_dict:
            user_history_dict[u] = [i]
        else:
            user_history_dict[u].append(i)
        if i not in item_history_dict:
            item_history_dict[i] = [u]
        else:
            item_history_dict[i].append(u)

    if strategy == "replace":  # random replace edges
        fake_data = []
        while len(fake_data) < interaction_length * prop:  # get tetrad
            random_user = random.randint(1, userID_max)
            random_item = random.randint(1, itemID_max)
            if random_item in user_history_dict.get(random_user, []):
                continue
            random_rating = -1
            fake_data.append([random_user, random_item, float(random_rating), 0.0])

        random_drop_index = random.sample(range(interaction_length), int(interaction_length * prop))
        for k in interaction_dict.keys():
            interaction_dict[k] = del_tensor_ele(interaction_dict[k], random_drop_index)

        if len(fake_data) > 0:  # combine with original inter
            fake_data = np.array(fake_data)
            for i, k in enumerate(interaction_dict.keys()):
                fake_data_filed = torch.tensor(fake_data[:, i], dtype=interaction_dict[k].dtype).reshape(-1)
                interaction_dict[k] = torch.cat((interaction_dict[k], fake_data_filed), 0)
    else:  # other perturbation
        print('ERROR: No implement!')

    dataset.inter_feat = Interaction(interaction_dict)
    return dataset
