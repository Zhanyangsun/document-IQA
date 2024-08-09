from ignite.metrics import Metric
import torch
import numpy as np
from scipy import stats

class DIQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, PLCC, and MSE.

    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self.label_pred = []
        self.label = []

    def update(self, output):
        y_pred, y = output
        self.label.append(y)
        self.label_pred.append(torch.mean(y_pred, dim=0))  # Make sure to take the mean across the correct dimension

    def compute(self):
        # Move tensors to CPU only when converting to numpy
        sq_std = np.reshape(np.asarray([t.cpu().numpy() for t in self.label]), (-1,))
        sq_pred = np.reshape(np.asarray([t.cpu().numpy() for t in self.label_pred]), (-1,))

        # Compute SROCC and PLCC (commented out)
        # srocc = stats.spearmanr(sq_std, sq_pred)[0]
        # plcc = stats.pearsonr(sq_std, sq_pred)[0]

        # Compute MSE
        mse = np.mean((sq_std - sq_pred) ** 2)

        # Return the metrics
        # return srocc, plcc
        return mse