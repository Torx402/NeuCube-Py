import torch

# the main question here is why are these hooks unimplemented?

class STDP():
    def __init__(self, alpha=0.0001, alpha_neg=0.01, t_constant=3):
        """
        Initializes the STDP object.

        Parameters:
            alpha (float): Synaptic adjustment rate.
            t_constant (int): Pre- and post-synaptic time interval.
        """
        self.alpha = alpha
        self.alpha_neg = alpha_neg
        self.t_constant = t_constant

    def setup(self, device, n_neurons):
        """
        Hook for performing setup tasks.

        Parameters:
            device (torch.device): The torch device.
            n_neurons (int): Number of neurons in the cube.
        """
        self.device = device
        self.neurons = n_neurons

    def per_sample(self, s):
        """
        Hook for performing per sample tasks.

        Parameters:
            s (int): The sample number.
        """
        pass

    def per_time_slice(self, s, k):
        """
        Hook for performing per time-slice tasks.

        Parameters:
            s (int): The sample number.
            k (int): The time-slice number.
        """
        pass

    def train(self, aux, w_latent, spike_latent):
        """
        Learning rule as per the original NeuCube-Py implementation.

        Parameters:
            aux (torch.Tensor): Element-wise time since last spike.
            w_latent (torch.Tensor): Output weights.
            spike_latent (torch.Tensor): Output spikes.
        Returns:
            pre_updates (torch.Tensor): Pre-synaptic updates.
            pos_updates (torch.Tensor): Post-synaptic updates.
        """
        pre_w = self.alpha * torch.exp(-aux / self.t_constant)*torch.gt(aux,0).int()
        pos_w = -self.alpha_neg * torch.exp(-aux / self.t_constant)*torch.gt(aux,0).int()
        pre_updates = pre_w * torch.gt((w_latent.T * spike_latent).T, 0).int()
        pos_updates = pos_w * torch.gt(w_latent * spike_latent, 0).int()
        return pre_updates, pos_updates

    def reset(self):
        """
        Hook for performing reset tasks.
        """
        pass

class STDPn():
    def __init__(self, alpha=0.01):
        """
        Initializes the STDP object.

        Parameters:
            alpha (float): Synaptic adjustment rate.
        """
        self.alpha = alpha

    def setup(self, device, n_neurons):
        """
        Hook for performing setup tasks.

        Parameters:
            device (torch.device): The torch device.
            n_neurons (int): Number of neurons in the cube.
        """
        self.device = device
        self.neurons = n_neurons

    def per_sample(self, s):
        """
        Hook for performing per sample tasks.

        Parameters:
            s (int): The sample number.
        """
        pass

    def per_time_slice(self, s, k):
        """
        Hook for performing per time-slice tasks.

        Parameters:
            s (int): The sample number.
            k (int): The time-slice number.
        """
        pass

    def train(self, aux, w_latent, spike_latent):
        """
        Non-exponential STDP Learning Rule.

        Parameters:
            aux (torch.Tensor): Element-wise time since last spike.
            w_latent (torch.Tensor): Output weights.
            spike_latent (torch.Tensor): Output spikes.
        Returns:
            pre_updates (torch.Tensor): Pre-synaptic updates.
            pos_updates (torch.Tensor): Post-synaptic updates.
        """
        pre_w = self.alpha * aux * torch.gt(aux,0).int()
        pos_w = -self.alpha * aux * torch.gt(aux,0).int()
        pre_updates = pre_w * torch.gt((w_latent.T * spike_latent).T, 0).int()
        pos_updates = pos_w * torch.gt(w_latent * spike_latent, 0).int()
        return pre_updates, pos_updates

    def reset(self):
        """
        Hook for performing reset tasks.
        """
        pass
