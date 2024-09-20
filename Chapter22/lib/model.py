import ptan
import torch
import torch.nn as nn
import numpy as np
from typing import List


class DQNModel(nn.Module):
    def __init__(self, view_shape, n_actions):
        super(DQNModel, self).__init__()

        self.view_conv = nn.Sequential(
            nn.Conv2d(view_shape[0], 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=2, padding=1),        # padding was added for deer model
            nn.ReLU(),
            nn.Flatten(),
        )
        view_out_size = self.view_conv(torch.zeros(1, *view_shape)).size()[-1]
        self.fc = nn.Sequential(
            nn.Linear(view_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        conv_out = self.view_conv(x)
        return self.fc(conv_out)


def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = exp.state  # the result will be masked anyway
        else:
            lstate = exp.last_state
        last_states.append(lstate)
    return states, np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=bool), \
           last_states


def calc_loss_dqn(batch, net, tgt_net, preprocessor, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = \
        unpack_batch(batch)

    states = preprocessor(states).to(device)
    next_states = preprocessor(next_states).to(device)

    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_state_vals = tgt_net(next_states).max(1)[0]
        next_state_vals[done_mask] = 0.0

    bellman_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, bellman_vals)
