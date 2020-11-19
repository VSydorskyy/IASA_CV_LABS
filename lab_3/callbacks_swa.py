from os.path import join as pjoin
from copy import deepcopy
from typing import Optional, List
from collections import OrderedDict

import torch
import numpy as np

from catalyst.dl import Callback, CallbackOrder

class SWACallback(Callback):
    def __init__(
        self,
        num_of_swa_models: int,
        maximize: bool,
        logging_metric: str,
        verbose: bool
    ):
        super().__init__(CallbackOrder.External)
        self.model_checkpoints = [None]*num_of_swa_models
        self.scores = [-np.inf]*num_of_swa_models if maximize else [np.inf]*num_of_swa_models
        
        self.maximize = maximize
        self.logging_metric = logging_metric
        self.verbose = verbose
        

    def on_stage_end(self, state):
        path_name = pjoin(state.logdir,'checkpoints',f'swa_models_{self.logging_metric}.pt')
        save_dict = {
            i:(s,m) for i, (s,m) in enumerate(zip(self.scores, self.model_checkpoints))
        }
        torch.save(save_dict, path_name)

    def _put_state_dict_on_cpu(self, sd):
        sd_copy = deepcopy(sd)
        for k, v in sd_copy.items():
            sd_copy[k] = v.cpu()
        return sd_copy

    def on_epoch_end(self, state):
        epoch_metric = state.loader_metrics[self.logging_metric]
        
        if self.maximize:
            if np.min(self.scores) < epoch_metric:
                min_value_index = np.argmin(self.scores)
                self.scores[min_value_index] = epoch_metric
                self.model_checkpoints[min_value_index] = self._put_state_dict_on_cpu(state.model.state_dict())
        else:
            if np.max(self.scores) > epoch_metric:
                max_value_index = np.argmax(self.scores)
                self.scores[max_value_index] = epoch_metric
                self.model_checkpoints[max_value_index] = self._put_state_dict_on_cpu(state.model.state_dict())
                
        if self.verbose:
            print('Best models scores by {} : {}'.format(self.logging_metric,self.scores))


def avarage_weights(
    path_to_chkp : str,
    save_path: str,
    delete_module : bool = False,
    take_best: Optional[bool] = None,
    sort_ascending: bool = True,
    verbose: bool = True
):
    nn_weights = torch.load(path_to_chkp, map_location='cpu')
    nn_weights = [el for _,el in nn_weights.items()]
    nn_weights = sorted(nn_weights, key=lambda x: x[0] if sort_ascending else -x[0])
    
    nn_scores = [el[0] for el in nn_weights]
    nn_weights = [el[1] for el in nn_weights]
    
    if verbose: print(f'SWA score: {nn_scores}')
    
    if take_best is not None:
        if verbose: print('Solo')
        avaraged_dict = OrderedDict()
        for k in nn_weights[take_best].keys():
            if delete_module:
                new_k = k[len('module.'):]
            else:
                new_k = k
                
            avaraged_dict[new_k] = nn_weights[take_best][k]
    else:
        if verbose: print('SWA')
        n_nns = len(nn_weights)
        if n_nns < 2:
            raise RuntimeError('Please provide more then 2 checkpoints')

        avaraged_dict = OrderedDict()
        for k in nn_weights[0].keys():
            if delete_module:
                new_k = k[len('module.'):]
            else:
                new_k = k

            avaraged_dict[new_k] = sum(nn_weights[i][k] for i in range(n_nns)) / float(n_nns)

    torch.save({'model_state_dict':avaraged_dict}, save_path)
        