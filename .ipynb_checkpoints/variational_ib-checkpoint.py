import torch
import numpy as np
import scipy as sp
import typing

class VariationalIB(torch.nn.Module):
    """
    Implementing the Variational IB module as a class, with a variety of sub-methods.
    The most important function is computing the loss of the encoded coordinates 
    against the prior. We assume here that the prior is a gaussian with unit covariance
    matrix. Other priors can be tested, but analytic forms of the loss may not be easily
    obtained.
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_dim: int,
        input_dim: int
    ):
        """
        Initialize the class. 
        Args:
            encoder: a torch module specifying an encoder. It must take in a dictionary and return the same dictionary, with the following keys added: 
                z_mu: the mean embedding. Dimensions latent_dim x 1
                z_log_var: the variance of each coordinate of hte embedding. Dimensions latent_dim x 1.
            decoder: a torch module specifying a decoder. It must take in a dictionary with minimally input key and return a dictionary with some output keys:
                input keys:
                    z: the embedding of the object. Latent dim x 1
                output keys:
                    Yhat: The class label of the object. input dim x 1
            latent_dim: integer specifying how many latent dimensions.
            input_dim: integer specifying the input size
        """

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def reparametrization(
        self,
        z_mu,
        z_log_var
    ) -> torch.Tensor:
        """
        Using the specified mean and log variance, we sample from a gaussian distribution with a unit variance
        to return the stochastic embedding.

        Args:
            z_mu: A torch.Tensor specifying the mean embedding.
            z_log_var: A torch.Tensor specifying the log variances of each coordinate of the embedding

        Returns:
            z: A torch.Tensor with the stochastic embedding
        """

        z = z_mu + torch.randn_like(z_mu) * torch.exp(z_log_var/2.)
        return z

    @staticmethod
    def prior_loss(
        batch: typing.Dict
    ) -> torch.Tensor:
        """
        Computing the prior loss, assuming prior is a gaussian with 0 mean, covariance 1.
        """
        z_mu = batch['z_mu']
        z_log_var = batch['z_log_var']
        return ( - 0.5 * (1. + z_log_var - z_mu ** 2 - torch.exp(z_log_var))).sum(dim=-1)

    def mse_loss(
        self,
        batch: typing.Dict
    ) -> torch.Tensor:
        """
        Computing MSE loss.
        """
        Y = batch['Y']
        Yhat = batch['Yhat']
        return ((Yhat - Y) ** 2).sum()/self.input_dim

    def forward(
        self,
        batch: typing.Dict
    ) -> typing.Dict:
        """
        Forward pass of the model.
        Takes in a dictionary minimally with input keys and adds keys to that dictionary as specified:
            input keys:
                X: torch.Tensor, of size input_dim x batch_size.
            output keys:
                z_mu: The mean embedding into the latent space.
                z_log_var: The log variance embedding into the latent space.
                z: The embedding into the latent space after reparametrization.
                Yhat: The prediction based on the embedding.
        """
        batch = self.encoder(batch)

        batch['z'] = self.reparametrization(batch['z_mu'], batch['z_log_var'])
        
        batch = self.decoder(batch)

        return batch

class Encoder(torch.nn.Module):
    """
    A simple MLP encoder class.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: typing.List[int],
        latent_dim: int,
        activation = torch.nn.ReLU()
    ):
        """
        Initialize the MLP encoder.
        """
        
        super().__init__()
    
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.activation = activation
        
        self.project_from_input = torch.nn.Linear(self.input_dim, self.hidden_dims[0])

        self.MLP = torch.nn.ModuleList()
        
        for i in range(1, len(self.hidden_dims)):
            self.MLP.append(torch.nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
            if self.activation is not None:
                self.MLP.append(self.activation)

        self.MLP = torch.nn.Sequential(*self.MLP)

        self.z_mu = torch.nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.z_log_var = torch.nn.Linear(self.hidden_dims[-1], self.latent_dim)

    def forward(
        self,
        batch: typing.Dict
    ) -> typing.Dict:
        """
        Adds keys for z_mu and z_log_var to batch, using the MLP method definied in __init__
        """
        
        X = batch['X']
        h = self.project_from_input(X)
        h = self.MLP(h)
        batch['z_mu'] = self.z_mu(h)
        batch['z_log_var'] = self.z_log_var(h)

        return batch

class Decoder(torch.nn.Module):
    """
    A simple MLP Decoder class.
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: typing.List[int],
        output_dim: int,
        activation = torch.nn.ReLU()
    ):
        """
        Initialize the MLP encoder.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.activation = activation
        
        self.project_from_latent = torch.nn.Linear(self.latent_dim, self.hidden_dims[0])

        self.MLP = torch.nn.ModuleList()
        
        for i in range(1, len(self.hidden_dims)):
            self.MLP.append(torch.nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
            if self.activation is not None:
                self.MLP.append(self.activation)

        self.MLP = torch.nn.Sequential(*self.MLP)

        self.Yhat = torch.nn.Linear(self.hidden_dims[-1], self.output_dim)

    def forward(
        self,
        batch: typing.Dict
    ) -> typing.Dict:
        """
        Adds keys Yhat, based on key z to batch.
        """
        
        z = batch['z']
        h = self.project_from_latent(z)
        h = self.MLP(h)
        batch['Yhat'] = self.Yhat(h)

        return batch
