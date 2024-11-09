import torch
from torch.distributions import Independent, Normal, Bernoulli

d, nh, D = 32, 200, 28 * 28

def loss_vae(
        x: torch.Tensor,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module
) -> float:
    """
    returns
    1. the average value of negative ELBO across the minibatch x
    2. and the output of the decoder
    """
    batch_size = x.size(0)
    encoder_output = encoder(x)
    
    mu, log_var = encoder_output[:, :d], encoder_output[:, d:]
    std = torch.exp(0.5 * log_var)
    
    pz = Independent(Normal(loc=torch.zeros(batch_size, d),
                            scale=torch.ones(batch_size, d)),
                     reinterpreted_batch_ndims=1)
    qz_x = Independent(Normal(loc=mu, scale=std),
                       reinterpreted_batch_ndims=1)
    
    z = qz_x.rsample()
    decoder_output = decoder(z)
    px_z = Independent(Bernoulli(logits=decoder_output),
                       reinterpreted_batch_ndims=1)

    elbo = -(px_z.log_prob(x) + pz.log_prob(z) - qz_x.log_prob(z)).mean()
    
    return elbo, decoder_output
