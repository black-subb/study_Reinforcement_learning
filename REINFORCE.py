import torch.nn as nn
import torch.optim
from torch.distributions.categorical import Categorical


class REINFORCE(nn.Module):

    def __init__(self,
                 policy: nn.Module,
                 gamma: float = 1.0,
                 lr: float = 0.0002):
        super(REINFORCE, self).__init__()
        self.policy = policy  # make sure that 'policy' return logits!
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.policy.parameters(),
                                    lr=lr)

        self._eps = 1e-25

    def get_action(self, state):
        with torch.no_grad():
            logits = self.policy(state)
            dist = Categorical(logits=logits)
            a = dist.sample()  # sample action from softmax policy
        return a

    @staticmethod
    def _pre_process_inputs(episode):
        states, actions, rewards = episode

        # assume inputs as follows
        # s : torch.tensor [num.steps X state_dim]
        # a : torch.tensor [num.steps]
        # r : torch.tensor [num.steps]

        # reversing inputs
        states = states.flip(dims=[0])
        actions = actions.flip(dims=[0])
        rewards = rewards.flip(dims=[0])
        return states, actions, rewards

    def update(self, episode):
        states, actions, rewards = self._pre_process_inputs(episode)

        g = 0
        for s, a, r in zip(states, actions, rewards):
            g = r + self.gamma * g
            dist = Categorical(logits=self.policy(s))
            prob = dist.probs[a]

            # Don't forget to put '-' in the front of pg_loss !!!
            # the default behavior of pytorch's optimizer is to minimize the targets
            # add 'self_eps' to prevent numerical problems of logarithms
            pg_loss = - torch.log(prob + self._eps) * g

            self.opt_zero_grad()

            pg_loss.backward()
            self.opt.step()
