from typing import Optional

import torch
import numpy as np

from catalyst.dl import Callback, CallbackOrder

class AccuracyCallback(Callback):
    def __init__(
        self,
        overall_accuracy: bool,
        class_accuracy: Optional[int]
    ):
        super().__init__(CallbackOrder.Metric)

        if not overall_accuracy and class_accuracy is None:
            raise RuntimeError(
                "Choose `overall_accuracy` "
                "or provide `class_accuracy`"
            )

        self.overall_accuracy = overall_accuracy
        self.class_accuracy = class_accuracy
        
        self.running_preds = []
        self.running_targets = []
        
    def on_batch_end(self, state):
        y_hat = state.output['logits'].detach().cpu().numpy().argmax(-1)
        y = state.input['targets'].detach().cpu().numpy()

        if len(y_hat.shape) < 1:
            y_hat = np.expand_dims(y_hat, axis=0)  

        if len(y.shape) < 1:
            y = np.expand_dims(y, axis=0)

        self.running_preds.append(y_hat)
        self.running_targets.append(y)
            
    def on_loader_end(self, state):
        y_true = np.concatenate(self.running_targets)
        y_pred = np.concatenate(self.running_preds)
        
        if self.overall_accuracy:
            score = (y_true == y_pred).sum() / y_true.shape[0]
            state.loader_metrics['overall_accuracy'] = score
        else:
            y_true = (y_true == self.class_accuracy)
            y_pred = (y_pred == self.class_accuracy)
            score = (y_true == y_pred).sum() / y_true.shape[0]
            state.loader_metrics[f'{self.class_accuracy}_accuracy'] = score

        self.running_preds = []
        self.running_targets = []