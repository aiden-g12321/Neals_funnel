'''Simulate time-domain data whose frequency representation obeys a power law.'''


import numpy as np
from jax import jit
import jax.numpy as jnp
import jax.random as jr


# time samples
Nt = 100
times = jnp.linspace(0., 1., Nt)
Tspan = times[-1] - times[0]

# frequency bins
Nf = 30
Na = 2 * Nf
freqs = jnp.arange(1, Nf + 1) / Tspan
log_freqs = jnp.log10(freqs)
f_ref = freqs[0]

# Fourier design matrix
F = jnp.zeros((Nt, Na))
for j in range(Nf):
    F = F.at[:, 2 * j].set(jnp.sin(2. * jnp.pi * freqs[j] * times))
    F = F.at[:, 2 * j + 1].set(jnp.cos(2. * jnp.pi * freqs[j] * times))

# get PSD given hyper-parameters
@jit
def get_rho(log_amp, gamma):
    amp = 10. ** log_amp
    rho = amp * (freqs / f_ref) ** (-gamma)
    return jnp.repeat(rho, 2)

# injected hyper-parameters
log_amp_inj = 1.2
gamma_inj = 3.2
hypers_inj = jnp.array([log_amp_inj, gamma_inj])
rho_inj = get_rho(log_amp_inj, gamma_inj)

# simulate Gaussian noise
sigma_inj = 1.2
noise_seed = 0
noise = jr.normal(jr.key(noise_seed), (Nt,)) * sigma_inj

# draw Fourier coefficients for signal realization
signal_seed = 2
a_inj = jr.multivariate_normal(jr.key(signal_seed), mean=jnp.zeros(Na), cov=jnp.diag(rho_inj))
signal_inj = F @ a_inj
data = signal_inj + noise


# save data in dictionary
data_dict = {}
data_dict['times'] = times
data_dict['freqs'] = freqs
data_dict['data'] = data
data_dict['signal_inj'] = signal_inj
data_dict['hypers_inj'] = hypers_inj
data_dict['sigma_inj'] = sigma_inj
data_dict['a_inj'] = a_inj
data_dict['F'] = F

np.savez_compressed("data.npz", **data_dict)



