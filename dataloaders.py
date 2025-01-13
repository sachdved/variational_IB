import torch
import pandas as pd
import numpy as np
import scipy as sp
import typing

class Ecology_Dataset(torch.utils.data.Dataset):
    """
    Creates a class that will be used for returning
    the X, Y pairs that correspond to presence, and
    the corresponding abundance data.
    I will log transform the relative abundance data
    because I'll assume the relative abunances is basically
    given as:
        relAb_i = exp(-s_i * \ Delta t) / \sum_{j} exp(-s_j * \ Delta t).
    """
    def __init__(
        self,
        presence_data: pd.DataFrame,
        relAbs_data: pd.DataFrame
    ):
        self.X = presence_data
        self.Y = relAbs_data

        self.data_dim = self.X.shape[1]

    def __len__(
        self
    ):
        """
        Return the number of data points in the dataset.
        """
        assert len(self.X) == len(self.Y)
        return len(self.X)

    def __getitem__(
        self,
        i
    ) -> typing.Dict:
        """
        Get the ith data point pair
        from the dataset.
        """
        X = torch.Tensor(self.X.iloc[i].values)

        # Adding a small number to deal with log of 0.
        Y = torch.log(
            torch.Tensor(
                abs(self.Y.iloc[i].values) + 1e-9
            )
        )
            

        batch = {}
        batch['X'] = X
        batch['Y'] = Y

        return batch
