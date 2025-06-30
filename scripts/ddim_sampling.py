import torch
import torch.nn.functional as F
from tqdm import tqdm
from scripts.catsample import sample_categorical


class UniformDDIMSampler:
    """
    DDIM-style sampler for uniform discrete diffusion models.
    Enables faster sampling with fewer steps while preserving quality.
    """

    def __init__(self, model, graph, noise, device, original_model=None):
        self.model = model  # This could be a function or the actual model
        self.original_model = original_model  # Store original model for eval()
        self.graph = graph
        self.noise = noise
        self.device = device
        
        # Ensure we're working with uniform graph
        if graph.absorb:
            raise ValueError("UniformDDIMSampler only works with non-absorbing (uniform) graphs")

    def sample_from_scores(self, scores, temperature=1.0, method='gumbel'):
        """
        Sample from score-based distribution with temperature control.
        
        Args:
            scores: Model scores [batch_size, seq_len, vocab_size]
            temperature: Sampling temperature (lower = more deterministic)
            method: 'gumbel' for Gumbel sampling, 'multinomial' for standard sampling
        
        Returns:
            Sampled tokens [batch_size, seq_len]
        """
        logits = scores / temperature

        if method == 'gumbel':
            # Gumbel sampling for better diversity
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            return (logits + gumbel_noise).argmax(dim=-1)
        else:
            # Standard multinomial sampling
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.shape[:-1])

    def ddim_step_uniform(self, x_t, t_current, t_next, eta=0.0, temperature=1.0):
        """
        Single DDIM step for uniform graph.
        
        Args:
            x_t: Current state [batch_size, seq_len]
            t_current: Current timestep
            t_next: Next timestep  
            eta: Stochasticity parameter (0=deterministic, 1=stochastic like DDPM)
            temperature: Sampling temperature
        
        Returns:
            x_next: Next state [batch_size, seq_len]
        """
        batch_size = x_t.shape[0]

        # Get noise parameters - need to reshape for proper broadcasting
        t_current_tensor = torch.full((batch_size, 1), t_current, device=self.device)
        sigma_current, _ = self.noise(t_current_tensor)

        if t_next > 0:
            t_next_tensor = torch.full((batch_size, 1), t_next, device=self.device)
            sigma_next, _ = self.noise(t_next_tensor)
        else:
            sigma_next = torch.zeros_like(sigma_current)

        # Get model scores
        with torch.no_grad():
            scores = self.model(x_t, sigma_current)

        # For final step, use more deterministic sampling
        if t_next == 0:
            final_temperature = min(temperature, 0.5)  # Lower temperature for final step
            return self.sample_from_scores(scores, final_temperature, 'gumbel')

        # Calculate step parameters
        step_size = t_current - t_next
        guidance_strength = step_size * 2.0  # Stronger guidance for larger steps
        
        # Adaptive temperature based on step size and eta
        adaptive_temperature = max(0.1, temperature * (1.0 - guidance_strength + eta))
        
        # DDIM-style deterministic update with optional stochasticity
        if eta == 0.0:
            # Pure deterministic DDIM
            # Use score to guide transition more strongly
            enhanced_scores = scores * (1.0 + guidance_strength)
            x_next = self.sample_from_scores(enhanced_scores, adaptive_temperature, 'gumbel')
        else:
            # Stochastic DDIM (interpolation between deterministic and random)
            # Deterministic component
            det_scores = scores * (1.0 + guidance_strength * (1.0 - eta))
            det_sample = self.sample_from_scores(det_scores, adaptive_temperature, 'gumbel')
            
            # Stochastic component
            if eta > 0:
                # Add noise proportional to eta
                noise_component = torch.randint_like(x_t, self.graph.dim)
                transition_prob = eta * step_size * 0.5  # Scale stochasticity
                should_use_noise = torch.rand_like(x_t, dtype=torch.float) < transition_prob
                x_next = torch.where(should_use_noise, noise_component, det_sample)
            else:
                x_next = det_sample

        return x_next

    def sample_ddim(self, batch_size, seq_length, num_inference_steps=20, eta=0.0, 
                   temperature=1.0, return_trajectory=False):
        """
        DDIM sampling for uniform discrete diffusion.
        
        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of each sequence
            num_inference_steps: Number of denoising steps (much fewer than DDPM)
            eta: Stochasticity parameter (0=deterministic, 1=stochastic)
            temperature: Sampling temperature
            return_trajectory: Whether to return full sampling trajectory
        
        Returns:
            Generated sequences [batch_size, seq_length]
            Optional: Full trajectory if return_trajectory=True
        """
        # Set model to eval mode if we have the original model
        if self.original_model is not None and hasattr(self.original_model, 'eval'):
            self.original_model.eval()
        elif hasattr(self.model, 'eval'):
            self.model.eval()

        # Create timestep schedule - linear for now, could be customized
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device)

        # Start from uniform random state (limiting distribution)
        x = self.graph.sample_limit(batch_size, seq_length).to(self.device)

        trajectory = [x.clone()] if return_trajectory else None

        with torch.no_grad():
            for i in tqdm(range(num_inference_steps), desc=f"DDIM sampling ({num_inference_steps} steps)"):
                t_current = timesteps[i].item()
                t_next = timesteps[i + 1].item()

                x = self.ddim_step_uniform(x, t_current, t_next, eta=eta, temperature=temperature)

                if return_trajectory:
                    trajectory.append(x.clone())

        if return_trajectory:
            return x, torch.stack(trajectory)
        return x


def get_ddim_sampler(graph, noise, device):
    """
    Factory function to create DDIM sampler.
    
    Args:
        graph: Graph object (must be uniform/non-absorbing)
        noise: Noise schedule object
        device: Device to run on
    
    Returns:
        Function that takes (model, batch_size, seq_length, **kwargs) and returns samples
    """
    def ddim_sampling_fn(model, batch_size, seq_length, num_inference_steps=20, 
                        eta=0.0, temperature=1.0, return_trajectory=False):
        sampler = UniformDDIMSampler(model, graph, noise, device)
        return sampler.sample_ddim(
            batch_size=batch_size,
            seq_length=seq_length, 
            num_inference_steps=num_inference_steps,
            eta=eta,
            temperature=temperature,
            return_trajectory=return_trajectory
        )
    
    return ddim_sampling_fn