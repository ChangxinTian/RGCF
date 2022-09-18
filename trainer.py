import numpy
import torch
from tqdm import tqdm
from time import time

from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.utils import early_stopping, dict2str, EvaluatorType, set_color, get_gpu_usage
from recbole.trainer import Trainer


class customized_Trainer(Trainer):
    def __init__(self, config, model):
        super(customized_Trainer, self).__init__(config, model)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (tqdm(
            enumerate(train_data),
            total=len(train_data),
            desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
        ) if show_progress else enumerate(train_data))
        for batch_idx, interaction in iter_data:
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()

            loss = self.model.calculate_loss(interaction, epoch_idx=epoch_idx, tensorboard=self.tensorboard)

            total_loss = loss.item() if total_loss is None else total_loss + loss.item()
            self._check_nan(loss)
            loss.backward()

            self.optimizer.step()
        return total_loss

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        self.eval_collector.data_collect(train_data)

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color('Saving current', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(valid_score,
                                                                                              self.best_valid_score,
                                                                                              self.cur_step,
                                                                                              max_step=self.stopping_step,
                                                                                              bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False, item_feat=None):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        iter_data = (tqdm(
            eval_data,
            total=len(eval_data),
            ncols=100,
            desc=set_color(f"Evaluate   ", 'pink'),
        ) if show_progress else eval_data)
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)

        if item_feat is not None:
            rec_items = struct.get('rec.items')
            user_num = rec_items.shape[0]
            diversity_res = 'diversity_res: \n'

            k_list = [10, 20, 50]
            if 'genre' in item_feat:
                item_genre_map = item_feat['genre'].values.tolist()
            elif 'categories' in item_feat:
                item_genre_map = item_feat['categories'].values.tolist()
            else:
                print('No item_genre!')
            genres = set()
            for ge in item_genre_map:
                genres = genres | set(ge)
            genres_size = len(genres)
            print("genres_size:", genres_size)
            for k in k_list:
                user_coverage = 0
                user_entropy = 0
                for u in range(user_num):
                    rec_genre_list = []
                    u_rec_k = rec_items[u, :k]
                    for i in u_rec_k:
                        rec_genre_list += list(item_genre_map[i])
                    rec_genre_list = numpy.array(rec_genre_list)
                    unique, counts = numpy.unique(rec_genre_list, return_counts=True)

                    for genre_count in counts:
                        p = genre_count / k
                        user_entropy += -p * numpy.log(p)
                    user_coverage += len(unique) / genres_size
                user_coverage /= user_num
                user_entropy /= user_num

                diversity_res += 'coverage@{0}: {1} \t entropy@{0}: {2} \n'.format(k, user_coverage, user_entropy)
            self.logger.info(diversity_res)
            print(diversity_res)

        return result
