from logging import getLogger

from recbole.utils import init_logger, init_seed, set_color
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

from trainer import customized_Trainer
from utils import customized_data_preparation
from rgcf import RGCF


def run(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
        saved (bool): whether to save the model
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset,
                    config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    if config['ptb_strategy'] != 'None':
        logger.info(set_color('inject some polluted interactions ! ', 'red'))
        train_data, valid_data, test_data = customized_data_preparation(
            config, dataset)
    else:
        train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = model(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = customized_Trainer(config, model)

    # model training
    _, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress'])

    # path = None
    test_result = trainer.evaluate(test_data,
                                   load_best_model=saved,
                                   show_progress=config['show_progress'],
                                   item_feat=None,
                                   model_file=None)

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    count = 0
    info = '\nbest valid score:'
    for i in best_valid_result.keys():
        if count == 0:
            info += '\n'
        count = (count + 1) % 3
        info += "{:15}:{:<10}    ".format(i, best_valid_result[i])
    logger.info(info)

    count = 0
    info = '\ntest result:'
    for i in test_result.keys():
        if count == 0:
            info += '\n'
        count = (count + 1) % 3
        info += "{:15}:{:<10}    ".format(i, test_result[i])
    logger.info(info)


if __name__ == '__main__':
    my_model = RGCF

    run(model=my_model, config_file_list=[
        './config/data.yaml', './config/model-rgcf.yaml'])
