"""นโยบายของแต่ละทีม: ผู้เล่นสองคนในทีมใช้ actor ร่วมกัน (รู้ว่าตัวเองเป็นคนไหนจาก obs)
critic เป็นแบบ centralized เห็น state เต็ม ใช้เฉพาะตอนเทรน"""
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .env import N_MOVE


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class TeamPolicy(nn.Module):
    def __init__(self, obs_dim, state_dim, vocab, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.move_head = nn.Linear(hidden, N_MOVE)
        self.msg_head = nn.Linear(hidden, vocab)
        self.critic = mlp(state_dim, 1, hidden)

    def dists(self, obs):
        z = self.body(obs)
        return Categorical(logits=self.move_head(z)), Categorical(logits=self.msg_head(z))

    def value(self, state):
        return self.critic(state).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, greedy=False):
        dm, dt = self.dists(obs)
        if greedy:
            m, t = dm.probs.argmax(-1), dt.probs.argmax(-1)
        else:
            m, t = dm.sample(), dt.sample()
        return m, t, dm.log_prob(m) + dt.log_prob(t)
