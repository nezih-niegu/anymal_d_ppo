"""PPO update steps for both trainer variants."""

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import MultivariateNormal


def train_basic(
    policy, optimizer, memory, hparams: dict, action_std: float
) -> tuple[float, float, float, float]:
    """PPO update for the BasicAgent (MultivariateNormal, external std).

    Returns (policy_loss, value_loss, entropy, mean_ratio).
    """
    gamma      = hparams["gamma"]
    ppo_epoch  = hparams["ppo_epoch"]
    batch_size = hparams["batch_size"]
    clip_param = hparams["clip_param"]
    c1         = hparams["c1"]
    c2         = hparams["c2"]

    s        = torch.tensor(memory.buffer["s"],     dtype=torch.float)
    a        = torch.tensor(memory.buffer["a"],     dtype=torch.float)
    r        = torch.tensor(memory.buffer["r"],     dtype=torch.float).view(-1, 1)
    s_       = torch.tensor(memory.buffer["s_"],    dtype=torch.float)
    old_logp = torch.tensor(memory.buffer["a_logp"], dtype=torch.float).view(-1, 1)

    act_len = a.shape[1]
    action_var = torch.full((act_len,), action_std * action_std)
    cov_mat = torch.diag(action_var).unsqueeze(0)

    with torch.no_grad():
        target_v = r + gamma * policy(s_)[1]
        adv = target_v - policy(s)[1]

    last_pl = last_vl = last_ent = last_ratio = 0.0
    for _ in range(ppo_epoch):
        for idx in BatchSampler(
            SubsetRandomSampler(range(memory.buffer_capacity)), batch_size, False
        ):
            probs, values = policy(s[idx])
            dist = MultivariateNormal(probs, cov_mat)
            a_logp = dist.log_prob(a[idx]).unsqueeze(1)
            entropy = dist.entropy().mean()
            ratio = torch.exp(a_logp - old_logp[idx])
            surr1 = ratio * adv[idx]
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * adv[idx]
            policy_loss = torch.min(surr1, surr2).mean()
            value_loss  = F.smooth_l1_loss(values, target_v[idx])
            loss = -policy_loss + c1 * value_loss - c2 * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_pl, last_vl = policy_loss.item(), value_loss.item()
            last_ent, last_ratio = entropy.item(), ratio.mean().item()

    return -last_pl, last_vl, last_ent, last_ratio


def train_advanced(
    policy, optimizer, memory, hparams: dict
) -> tuple[float, float, float, float]:
    """PPO update for the AdvancedAgent (learnable log_std, advantage normalisation).

    Returns (policy_loss, value_loss, entropy, mean_ratio).
    """
    gamma      = hparams["gamma"]
    ppo_epoch  = hparams["ppo_epoch"]
    batch_size = hparams["batch_size"]
    clip_param = hparams["clip_param"]
    c1         = hparams["c1"]
    c2         = hparams["c2"]
    max_grad   = hparams.get("max_grad_norm", 0.5)

    s        = torch.tensor(memory.buffer["s"],     dtype=torch.float)
    a        = torch.tensor(memory.buffer["a"],     dtype=torch.float)
    r        = torch.tensor(memory.buffer["r"],     dtype=torch.float).view(-1, 1)
    s_       = torch.tensor(memory.buffer["s_"],    dtype=torch.float)
    d        = torch.tensor(memory.buffer["d"],     dtype=torch.float).view(-1, 1)
    old_logp = torch.tensor(memory.buffer["a_logp"], dtype=torch.float).view(-1, 1)

    with torch.no_grad():
        v_next   = policy(s_)[1]
        v_curr   = policy(s)[1]
        target_v = r + gamma * v_next * (1.0 - d)   # zero bootstrap at terminals
        adv      = target_v - v_curr
        adv      = (adv - adv.mean()) / (adv.std() + 1e-8)

    last_pl = last_vl = last_ent = last_ratio = 0.0
    for _ in range(ppo_epoch):
        for idx in BatchSampler(
            SubsetRandomSampler(range(memory.buffer_capacity)), batch_size, False
        ):
            dist, value = policy.dist(s[idx])
            a_logp  = dist.log_prob(a[idx]).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1).mean()
            ratio   = torch.exp(a_logp - old_logp[idx])
            surr1   = ratio * adv[idx]
            surr2   = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * adv[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = F.smooth_l1_loss(value, target_v[idx])
            loss = policy_loss + c1 * value_loss - c2 * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad)
            optimizer.step()
            last_pl, last_vl = policy_loss.item(), value_loss.item()
            last_ent, last_ratio = entropy.item(), ratio.mean().item()

    return last_pl, last_vl, last_ent, last_ratio
