#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import random
import math


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, rng=None):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.Experience = namedtuple(
            "Experience",
            ["state", "action", "reward", "next_state", "done"]
        )

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(
                self.Experience(state, action, reward, next_state, done)
            )
        else:
            self.buffer[self.pos] = \
                self.Experience(state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None, None, None, None, None, None

        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = self.rng.choice(
            len(self.buffer), batch_size, p=probabilities
        )

        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in indices:
            exp = self.buffer[i]
            states.append(np.array(exp.state, dtype=np.float32))
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(np.array(exp.next_state, dtype=np.float32))
            dones.append(exp.done)

        states      = np.array(states,      dtype=np.float32)
        actions     = np.array(actions)
        rewards     = np.array(rewards,     dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        # ✅ 修复：np.bool 已在 NumPy 1.24 废弃，改用内置 bool
        dones       = np.array(dones,       dtype=bool)

        self.beta = min(1.0, self.beta + self.beta_increment)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            if priority > self.max_priority:
                self.max_priority = priority

    def __len__(self):
        return len(self.buffer)


class GELU(nn.Module):
    """兼容旧版本PyTorch的GELU"""
    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class DQN(nn.Module):
    """Dueling DQN"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3):
        super().__init__()

        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.layer_norm_input = nn.LayerNorm(hidden_dim)

        self.hidden_layers = nn.ModuleList()
        for _ in range(max(0, num_layers - 2)):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                GELU(),
                nn.Dropout(0.1)
            ))

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            GELU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        out = GELU()(self.layer_norm_input(self.input_layer(x)))

        for layer in self.hidden_layers:
            residual = out
            out = layer(out) + residual

        value     = self.value_stream(out)
        advantage = self.advantage_stream(out)
        q_values  = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class VirtualEdgeRLAgent:
    """
    强化学习智能体，自适应调整虚拟边参数。
    action_dim=63，对应9×7的(Δthreshold, ΔtopK)组合。
    state_dim默认为8: [val_auc, val_f1, val_precision, val_recall,
                         hh_edge_density, tt_edge_density, threshold_norm, topk_norm]
    """
    def __init__(self,
                 state_dim=9,
                 action_dim=63,       # 9×7: 细步长threshold × topK[1,10]
                 action_heads=1,
                 action_head_labels=None,
                 lr=1e-3,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.999,
                 epsilon_min=0.05,
                 buffer_size=20000,
                 batch_size=64,
                 verbose=False,
                 seed=42):

        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.action_heads = max(1, int(action_heads))
        if action_head_labels is None:
            self.action_head_labels = tuple(
                f'head_{idx}' for idx in range(self.action_heads)
            )
        else:
            self.action_head_labels = tuple(action_head_labels)
            if len(self.action_head_labels) != self.action_heads:
                raise ValueError(
                    "action_head_labels length must match action_heads: "
                    f"{len(self.action_head_labels)} != {self.action_heads}"
                )
        self.verbose     = verbose
        self.seed        = int(seed)
        self.np_rng      = np.random.default_rng(self.seed)

        # Q网络
        output_dim = action_dim * self.action_heads
        self.q_net        = DQN(state_dim, output_dim, hidden_dim=128, num_layers=3).to(self.device)
        self.target_q_net = DQN(state_dim, output_dim, hidden_dim=128, num_layers=3).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.AdamW(
            self.q_net.parameters(), lr=lr, weight_decay=1e-5
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2, eta_min=lr * 0.01
        )

        self.memory = PrioritizedReplayBuffer(buffer_size, rng=self.np_rng)

        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_decay   = epsilon_decay
        self.epsilon_min     = epsilon_min
        self.batch_size      = batch_size
        self.train_steps     = 0
        self.target_update_tau = 0.005
        self.best_reward     = -float('inf')

        # 9×7=63 个动作（细步长threshold × topK）
        # Δthreshold ∈ {-0.10,-0.05,-0.02,-0.01,0,+0.01,+0.02,+0.05,+0.10}
        # ΔtopK      ∈ {-3,-2,-1,0,+1,+2,+3}
        # clip范围：threshold∈[0.10,0.99]，topK∈[1,10]
        self.actions = {}
        delta_thresholds = [-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10]
        delta_topks      = [-3, -2, -1, 0, 1, 2, 3]
        action_id = 0
        for dt in delta_thresholds:
            for dk in delta_topks:
                self.actions[action_id] = (dt, dk)
                action_id += 1
        # 此时 action_id == 63，与 action_dim 一致

        # 阶段配置
        self.optimization_stage = 0
        self.stage_switch_steps = [5000, 10000]

    def _predict_eval(self, net, tensor):
        was_training = net.training
        net.eval()
        try:
            with torch.no_grad():
                return net(tensor)
        finally:
            if was_training:
                net.train()

    def _reshape_q_values(self, q_values):
        return q_values.view(-1, self.action_heads, self.action_dim)

    # ─────────────────────────────────────────────────────
    # 动作选择
    # ─────────────────────────────────────────────────────
    def select_action(self, state):
        decay_rate = 1500
        current_epsilon = (
            self.epsilon_min
            + (self.epsilon - self.epsilon_min)
            * math.exp(-1.0 * self.train_steps / decay_rate)
        )

        if self.train_steps < 2000:
            if self.np_rng.random() < current_epsilon:
                return self.np_rng.integers(
                    0, self.action_dim, size=self.action_heads, dtype=np.int64
                )
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self._reshape_q_values(
                    self._predict_eval(self.q_net, state_tensor)
                )
                return q_values.argmax(dim=2).squeeze(0).cpu().numpy().astype(np.int64)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self._reshape_q_values(
                self._predict_eval(self.q_net, state_tensor)
            ).squeeze(0)
            temperature = max(0.1, 2.0 * math.exp(-1.0 * self.train_steps / 500))
            actions = []
            for head_idx in range(self.action_heads):
                exp_q = torch.exp(q_values[head_idx] / temperature)
                probs = (exp_q / torch.sum(exp_q)).cpu().numpy()
                # 数值安全
                probs = np.abs(probs)
                probs = probs / probs.sum()
                actions.append(int(self.np_rng.choice(self.action_dim, p=probs)))
            return np.array(actions, dtype=np.int64)

    # ─────────────────────────────────────────────────────
    # 网络更新
    # ─────────────────────────────────────────────────────
    def update(self):
        if len(self.memory) < self.batch_size:
            return

        result = self.memory.sample(self.batch_size)
        if result[0] is None:
            return

        (state_batch, action_batch, reward_batch,
         next_state_batch, done_batch, weights, indices) = result

        state_batch      = torch.FloatTensor(state_batch).to(self.device)
        action_batch     = torch.LongTensor(action_batch).to(self.device)
        reward_batch     = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch       = torch.BoolTensor(done_batch).to(self.device)
        weights          = torch.FloatTensor(weights).to(self.device)

        if action_batch.dim() == 1:
            action_batch = action_batch.unsqueeze(1)

        current_q_values = self._reshape_q_values(self.q_net(state_batch))
        current_q = current_q_values.gather(
            2, action_batch.unsqueeze(-1)
        ).squeeze(-1).mean(dim=1)

        with torch.no_grad():
            next_q_online = self._reshape_q_values(
                self._predict_eval(self.q_net, next_state_batch)
            )
            next_actions = next_q_online.argmax(dim=2)
            target_q_values = self._reshape_q_values(
                self._predict_eval(self.target_q_net, next_state_batch)
            )
            next_q = target_q_values.gather(
                2, next_actions.unsqueeze(-1)
            ).squeeze(-1).mean(dim=1)
            target_q = reward_batch + self.gamma * next_q * (~done_batch).float()

        td_errors = current_q - target_q
        loss      = ((td_errors ** 2) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.lr_scheduler.step()

        new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 0.01
        self.memory.update_priorities(indices, new_priorities)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_steps += 1
        self._soft_update_target_net()

    def _soft_update_target_net(self):
        for tp, lp in zip(self.target_q_net.parameters(),
                          self.q_net.parameters()):
            tp.data.copy_(
                self.target_update_tau * lp.data
                + (1.0 - self.target_update_tau) * tp.data
            )

    def update_target_net(self):
        self._soft_update_target_net()

    def get_optimal_parameters(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self._reshape_q_values(self.q_net(state_tensor)).squeeze(0)
        best_actions = q_values.argmax(dim=1).cpu().tolist()
        params = [self.actions.get(int(action), (0.0, 0)) for action in best_actions]
        if self.action_heads == 1:
            return params[0]
        return params

    def save_model(self, path):
        torch.save({
            'q_net_state_dict':        self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict':    self.optimizer.state_dict(),
            'epsilon':                 self.epsilon,
            'train_steps':             self.train_steps,
            'state_dim':               self.state_dim,
            'action_dim':              self.action_dim,
            'action_heads':            self.action_heads,
            'action_head_labels':      list(self.action_head_labels),
        }, path)
        if self.verbose:
            print(f"✅ RL模型已保存: {path}")

    def load_model(self, path):
        try:
            state = torch.load(path, map_location=self.device)
            if 'q_net_state_dict' in state:
                ckpt_state_dim = int(state.get('state_dim', self.state_dim))
                ckpt_action_dim = int(state.get('action_dim', self.action_dim))
                ckpt_action_heads = int(state.get('action_heads', 1))
                ckpt_action_head_labels = tuple(
                    state.get('action_head_labels', self.action_head_labels)
                )

                if (
                    ckpt_state_dim != self.state_dim
                    or ckpt_action_dim != self.action_dim
                    or ckpt_action_heads != self.action_heads
                    or ckpt_action_head_labels != self.action_head_labels
                ):
                    print(
                        f"⚠️ RL checkpoint维度不匹配，跳过加载并冷启动: "
                        f"ckpt(state={ckpt_state_dim}, action={ckpt_action_dim}, heads={ckpt_action_heads}, labels={ckpt_action_head_labels}) "
                        f"!= current(state={self.state_dim}, action={self.action_dim}, heads={self.action_heads}, labels={self.action_head_labels})"
                    )
                    return

                self.q_net.load_state_dict(state['q_net_state_dict'])
                self.target_q_net.load_state_dict(
                    state['target_q_net_state_dict']
                )
                self.epsilon     = state.get('epsilon',     self.epsilon)
                self.train_steps = state.get('train_steps', self.train_steps)
                if self.verbose:
                    print(f"✅ RL模型已加载: {path}")
            else:
                # 老版本仅保存了state_dict，无法保证维度兼容，保守地仅在完全匹配时尝试加载
                q_state = state
                first_key = next(iter(q_state.keys())) if len(q_state) > 0 else None
                if first_key is not None and q_state[first_key].shape != self.q_net.state_dict()[first_key].shape:
                    print("⚠️ RL旧checkpoint与当前网络维度不匹配，跳过加载并冷启动。")
                    return
                self.q_net.load_state_dict(q_state)
                self.target_q_net.load_state_dict(self.q_net.state_dict())
        except Exception as e:
            print(f"❌ 加载RL模型失败: {e}，使用新初始化模型继续。")
