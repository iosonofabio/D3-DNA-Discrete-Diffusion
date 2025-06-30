import abc
import torch
import torch.nn.functional as F
from scripts.catsample import sample_categorical
from scripts.ddim_sampling import get_ddim_sampler

from utils.utils import get_score_fn

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, labels, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, labels, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, labels, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        # print (stag_score.shape)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, labels, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, labels, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model, labels):
        sampling_score_fn = get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, labels, t, dt)
            # print(x)
            

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, labels, t)
            
        return x
    
    return pc_sampler


def get_ddim_sampler_wrapper(graph, noise, batch_dims, num_inference_steps=20, eta=0.0, temperature=1.0, device=torch.device('cpu')):
    """
    Wrapper for DDIM sampler to match the interface of get_pc_sampler.
    
    Args:
        graph: Graph object
        noise: Noise schedule
        batch_dims: Tuple of (batch_size, seq_length)
        num_inference_steps: Number of DDIM steps
        eta: Stochasticity parameter
        temperature: Sampling temperature
        device: Device to run on
    
    Returns:
        Sampling function that takes (model, labels) and returns samples
    """
    if graph.absorb:
        raise ValueError("DDIM sampler currently only supports uniform (non-absorbing) graphs")
    
    batch_size, seq_length = batch_dims
    ddim_sampler_fn = get_ddim_sampler(graph, noise, device)
    
    @torch.no_grad()
    def ddim_wrapper(model, labels):
        # Create DDIM sampler with the actual model
        from scripts.ddim_sampling import UniformDDIMSampler
        
        # Get model in correct format for DDIM sampler
        sampling_score_fn = get_score_fn(model, train=False, sampling=True)
        
        # Create a wrapped model that matches DDIM interface
        def wrapped_model(x, sigma):
            # labels are not used in current DDIM implementation, but kept for compatibility
            return sampling_score_fn(x, labels, sigma)
        
        # Create DDIM sampler instance directly
        sampler = UniformDDIMSampler(wrapped_model, graph, noise, device, original_model=model)
        
        return sampler.sample_ddim(
            batch_size=batch_size,
            seq_length=seq_length,
            num_inference_steps=num_inference_steps,
            eta=eta,
            temperature=temperature
        )
    
    return ddim_wrapper

