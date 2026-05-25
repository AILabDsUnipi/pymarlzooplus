import numpy as np
import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

# TODO: Use different network structure other than MLP
# ref: https://github.com/pyro-ppl/pyro/blob/dev/examples/cvae/cvae.py


class StatePredBL(nn.Module):
    def __init__(self, args):
        super().__init__()

        hidden_dim = args.hidden_dim
        embedding_dim = args.embedding_dim
        n_actions = args.n_actions
        state_dim = int(np.prod(args.state_shape))
        self.n_agents = args.n_agents

        # 0 is no embedding!!!
        self.embedding = nn.Embedding(n_actions + 1, embedding_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + embedding_dim * self.n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.non_linear = nn.Tanh()

    def forward(self, s, a):
        """_summary_
        Args:
            s (_type_): (bs, state_dim)
            a (_type_): (bs, n_agents)
        """
        assert len(a.shape) == 2
        bs, n_agents = a.shape
        assert n_agents == self.n_agents
        # avoid zeros, (bs, n_agents) range(1, n_actions)
        a = a + 1
        a_embedding = self.embedding(a).reshape((bs, -1))
        x = torch.cat([s, a_embedding], dim=-1)

        out = self.mlp(x)
        # not 100% correct because the z axis is unbounded, however, otherwise numerical stability can not be ensured
        out = self.non_linear(out)

        return out


class Encoder(nn.Module):
    def __init__(
        self, state_dim, embedding_dim, n_agents, n_actions, z_dim, hidden_dim=64
    ):
        super().__init__()
        self.n_agents = n_agents

        # 0 is no embedding!!!
        self.embedding = nn.Embedding(n_actions + 1, embedding_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(2 * state_dim + embedding_dim * n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_dim),
        )

    def forward(self, x_s, x_a, y_s_prime):
        """_summary_
        Args:
            x_a (_type_): (bs, n_agents)
            x_s (_type_): (bs, state_dim)
            y_s_prime (_type_): (bs, state_dim)
        Returns:
            _type_: _description_
        """
        assert len(x_a.shape) == 2
        bs, n_agents = x_a.shape
        assert n_agents == self.n_agents
        # avoid zeros, (bs, (n_agents * n_agents + 1))
        x_a = x_a + 1
        x_a_embedding = self.embedding(x_a).reshape((bs, -1))

        xy = torch.cat([x_s, x_a_embedding, y_s_prime], dim=-1)
        z_param = self.mlp(xy)
        z_mu, z_sigma = torch.chunk(z_param, 2, dim=-1)
        z_sigma = torch.exp(z_sigma)

        return z_mu, z_sigma


class Decoder(nn.Module):
    """Modified from the original codes
    Args:
        nn (_type_): _description_
    """

    def __init__(self, state_dim, z_dim, hidden_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * state_dim),
        )
        self.non_linear = nn.Tanh()

    def forward(self, z):
        y_param = self.mlp(z)
        # y_param = self.non_linear(x)
        y_mu, y_sigma = torch.chunk(y_param, 2, dim=-1)
        # not 100% correct because the z axis is unbounded, however, otherwise numerical stability can not be ensured
        y_mu = self.non_linear(y_mu)
        y_sigma = torch.exp(y_sigma)

        return y_mu, y_sigma


class StatePredCVAE(nn.Module):
    def __init__(self, args, baseline_net_global, baseline_net_local):
        super().__init__()

        z_dim = args.z_dim
        hidden_dim = args.hidden_dim
        embedding_dim = args.embedding_dim
        n_actions = args.n_actions
        n_agents = args.n_agents
        state_dim = int(np.prod(args.state_shape))

        self.baseline_net_global = baseline_net_global
        self.baseline_net_local = baseline_net_local

        self.x_dim = state_dim + embedding_dim * n_agents
        self.y_dim = state_dim

        self.prior_net_global = Encoder(
            state_dim=state_dim,
            embedding_dim=embedding_dim,
            n_agents=n_agents,
            n_actions=n_actions,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
        )
        self.generation_net = Decoder(
            state_dim=state_dim, z_dim=z_dim, hidden_dim=hidden_dim
        )
        self.recognition_net_glboal = Encoder(
            state_dim=state_dim,
            embedding_dim=embedding_dim,
            n_agents=n_agents,
            n_actions=n_actions,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
        )

        self.prior_net_local = Encoder(
            state_dim=state_dim,
            embedding_dim=embedding_dim,
            n_agents=n_agents,
            n_actions=n_actions,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
        )
        self.recognition_net_local = Encoder(
            state_dim=state_dim,
            embedding_dim=embedding_dim,
            n_agents=n_agents,
            n_actions=n_actions,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
        )

    def model(self, s, a, baseline_net, prior_net, latent_name, s_prime=None):
        """_summary_
        Args:
            s (_type_): (bs, state_dim)
            a (_type_): (bs, n_agents)
            baseline_net: function
            prior_net: function
            latent_name: string
            s_prime (_type_, optional): (bs, state_dim). Defaults to None.
        """
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("generation_net", self)

        bs, _ = s.shape
        # normalize ELBO
        with pyro.poutine.scale(scale=1.0 / bs):
            with pyro.plate("data"):
                # Prior network uses the baseline predictions as initial guess.
                # This is the generative process with recurrent connection

                with torch.no_grad():
                    # this ensures the training process does not change the
                    # baseline network
                    y_s_prime_hat = baseline_net(s, a)

                # sample the handwriting style from the prior distribution, which is
                # modulated by the input xs.
                prior_mu, prior_sigma = prior_net(s, a, y_s_prime_hat)
                # for numerical stability
                prior_sigma = prior_sigma.clamp(min=0.01)
                zs = pyro.sample(
                    latent_name, dist.Normal(prior_mu, prior_sigma).to_event(1)
                )

                # the output y is generated from the distribution pθ(y|x, z)
                y_mu, y_sigma = self.generation_net(zs)
                # for numerical stability
                y_sigma = y_sigma.clamp(min=0.01)

                if s_prime is not None:
                    # In training, we will only sample in the masked image
                    pyro.sample(
                        "y", dist.Normal(y_mu, y_sigma).to_event(1), obs=s_prime
                    )
                else:
                    # In testing, no need to sample: the output is already a
                    # probability in [0, 1] range, which better represent states
                    # this actually is not used in our method, just implement for the sake
                    # of completeness
                    pyro.deterministic("y", y_mu.detach())

            # return the y_mu and y_sigma so we can visualize it later
            return prior_mu, prior_sigma, y_mu, y_sigma

    def guide(
        self, s, a, baseline_net, prior_net, recognition_net, latent_name, s_prime=None
    ):
        bs, _ = s.shape
        # normalize ELBO
        with pyro.poutine.scale(scale=1.0 / bs):
            with pyro.plate("data"):
                if s_prime is not None:
                    # at training time, uses the variational distribution
                    # q(z|x,y) = normal(loc(x,y),scale(x,y))
                    mu, sigma = recognition_net(s, a, s_prime)
                else:
                    # at inference time, ys is not provided. In that case,
                    # the model uses the prior network
                    y_s_prime_hat = baseline_net(s, a)
                    mu, sigma = prior_net(s, a, y_s_prime_hat)

                # for numerical stability
                sigma = sigma.clamp(min=0.01)
                pyro.sample(latent_name, dist.Normal(mu, sigma).to_event(1))

    def model_global(self, s, a, s_prime=None):
        return self.model(
            s,
            a,
            baseline_net=self.baseline_net_global,
            prior_net=self.prior_net_global,
            latent_name="z_global",
            s_prime=s_prime,
        )

    def model_local(self, s, a, s_prime=None):
        return self.model(
            s,
            a,
            baseline_net=self.baseline_net_local,
            prior_net=self.prior_net_local,
            latent_name="z_local",
            s_prime=s_prime,
        )

    def guide_global(self, s, a, s_prime=None):
        self.guide(
            s,
            a,
            baseline_net=self.baseline_net_global,
            prior_net=self.prior_net_global,
            recognition_net=self.recognition_net_glboal,
            latent_name="z_global",
            s_prime=s_prime,
        )

    def guide_local(self, s, a, s_prime=None):
        self.guide(
            s,
            a,
            baseline_net=self.baseline_net_local,
            prior_net=self.prior_net_local,
            recognition_net=self.recognition_net_local,
            latent_name="z_local",
            s_prime=s_prime,
        )
