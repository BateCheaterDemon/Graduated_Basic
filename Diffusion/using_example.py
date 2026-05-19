import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import math

# ============ 1. Diffusion策略网络 ============
class DiffusionActor(nn.Module):
    """用Diffusion建模策略分布"""
    def __init__(self, state_dim, action_dim, T=100, time_embed_dim=32):
        super().__init__()
        self.T = T
        self.action_dim = action_dim
        self.time_embed_dim = time_embed_dim
        
        # 时间嵌入层
        self.time_embed = nn.Sequential(
            nn.Linear(T, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 简化的Diffusion网络
        # 输入维度：action_dim + state_dim + time_embed_dim
        input_dim = action_dim + state_dim + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Beta调度
        betas = torch.linspace(0.0001, 0.02, T)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, 0))
        
    def forward(self, x, t, state):
        """预测噪声"""
        # 时间步编码
        t_onehot = F.one_hot(t, self.T).float()
        t_emb = self.time_embed(t_onehot)
        
        # 拼接输入
        combined = torch.cat([x, state, t_emb], dim=-1)
        return self.net(combined)
    
    def sample(self, state, return_trajectory=False):
        """从Diffusion采样动作"""
        batch_size = state.shape[0]
        device = state.device
        x = torch.randn(batch_size, self.action_dim, device=device)
        
        trajectory = [x] if return_trajectory else None
        
        for t in reversed(range(self.T)):
            t_batch = torch.full((batch_size,), t, device=device)
            pred_noise = self.forward(x, t_batch, state)
            
            # DDIM采样（简化版）
            alpha_bar = self.alpha_bars[t]
            alpha_bar_prev = self.alpha_bars[t-1] if t > 0 else torch.tensor(1.0, device=device)
            
            x0_pred = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            if t > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * noise
            else:
                x = x0_pred
            
            if return_trajectory:
                trajectory.append(x)
        
        if return_trajectory:
            return x, trajectory
        return x
    
    def get_log_prob(self, state, action):
        """估计动作的对数概率（用于PPO）"""
        # 简化：用ELBO近似
        device = state.device
        t = torch.randint(0, self.T, (state.shape[0],), device=device)
        noise = torch.randn_like(action)
        
        alpha_bar = self.alpha_bars[t].view(-1, 1)
        noisy_action = torch.sqrt(alpha_bar) * action + torch.sqrt(1 - alpha_bar) * noise
        
        pred_noise = self.forward(noisy_action, t, state)
        
        # 负MSE作为对数概率的近似
        log_prob = -F.mse_loss(pred_noise, noise, reduction='none').mean(dim=-1)
        return log_prob


# ============ 2. PPO价值网络 ============
class ValueNetwork(nn.Module):
    """估计状态价值V(s)"""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        return self.net(state)


# ============ 3. 经验回放缓冲区 ============
class PPOBuffer:
    def __init__(self, capacity=2048):
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)
        self.log_probs = deque(maxlen=capacity)
    
    def add(self, state, action, reward, done, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def compute_gae(self, gamma=0.99, lam=0.95):
        """计算GAE优势函数"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t+1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return (torch.FloatTensor(np.array(self.states)),
                torch.FloatTensor(np.array(self.actions)),
                torch.FloatTensor(advantages),
                torch.FloatTensor(returns),
                torch.FloatTensor(self.log_probs))
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()


# ============ 4. 联合训练器 ============
class DiffusionPPOTrainer:
    def __init__(self, env_name="Pendulum-v1", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.env = gym.make(env_name)
        self.device = device
        
        state_dim = self.env.observation_space.shape[0]  # Pendulum: 3维 (cos(theta), sin(theta), theta_dot)
        action_dim = self.env.action_space.shape[0]      # Pendulum: 1维 (torque)
        
        print(f"Environment: {env_name}")
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
        print(f"Action space: [{self.env.action_space.low[0]:.2f}, {self.env.action_space.high[0]:.2f}]")
        print(f"Device: {device}")
        
        self.actor = DiffusionActor(state_dim, action_dim).to(device)
        self.critic = ValueNetwork(state_dim).to(device)
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.buffer = PPOBuffer()
        
        # 动作缩放（将[-1,1]映射到实际动作空间）
        self.action_scale = torch.FloatTensor(self.env.action_space.high).to(device)
        self.action_bias = torch.FloatTensor((self.env.action_space.high + self.env.action_space.low) / 2).to(device)
        
        # 训练统计
        self.best_eval_reward = -float('inf')
    
    def train(self, num_iterations=1000, steps_per_iteration=2048):
        for iteration in range(num_iterations):
            # ===== 1. 收集数据 =====
            state, _ = self.env.reset()
            episode_reward = 0
            episode_count = 0
            
            for step in range(steps_per_iteration):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # Diffusion采样动作（输出范围[-1,1]）
                    action_normalized = self.actor.sample(state_tensor)
                    action_normalized = action_normalized.cpu().numpy()[0]
                    
                    # 获取价值和概率
                    value = self.critic(state_tensor).item()
                    action_tensor = torch.FloatTensor(action_normalized).unsqueeze(0).to(self.device)
                    log_prob = self.actor.get_log_prob(state_tensor, action_tensor).item()
                
                # 缩放到实际动作空间，并确保shape匹配环境(action_dim=1 -> shape=(1,))
                action = action_normalized * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()
                action = np.asarray(action, dtype=np.float32).reshape(self.env.action_space.shape)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.buffer.add(state, action_normalized, reward, done, value, log_prob)
                episode_reward += reward
                
                state = next_state
                if done:
                    episode_count += 1
                    print(f"Iter {iteration}, Ep {episode_count}, Reward: {episode_reward:.2f}")
                    state, _ = self.env.reset()
                    episode_reward = 0
            
            # ===== 2. 计算优势函数 =====
            states, actions, advantages, returns, old_log_probs = self.buffer.compute_gae()
            states, actions = states.to(self.device), actions.to(self.device)
            advantages, returns = advantages.to(self.device), returns.to(self.device)
            old_log_probs = old_log_probs.to(self.device)
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # ===== 3. 更新Critic =====
            critic_losses = []
            for _ in range(10):
                pred_values = self.critic(states).squeeze()
                critic_loss = F.mse_loss(pred_values, returns)
                
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                critic_losses.append(critic_loss.item())
            
            # ===== 4. 更新Actor（Diffusion + PPO） =====
            actor_losses = []
            for _ in range(10):
                # 采样时间步
                t = torch.randint(0, self.actor.T, (states.shape[0],), device=self.device)
                
                # 前向加噪
                noise = torch.randn_like(actions)
                alpha_bar = self.actor.alpha_bars[t].view(-1, 1)
                noisy_actions = torch.sqrt(alpha_bar) * actions + torch.sqrt(1 - alpha_bar) * noise
                
                # 预测噪声
                pred_noise = self.actor(noisy_actions, t, states)
                
                # Diffusion损失（预测噪声）
                diffusion_loss = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=-1)
                
                # PPO风格的策略损失（用advantage加权）
                policy_loss = (diffusion_loss * advantages).mean()
                
                # 熵奖励
                entropy_bonus = 0.01 * torch.log(self.actor.betas[t].view(-1, 1) + 1e-8).mean()
                
                total_actor_loss = policy_loss - entropy_bonus
                
                self.actor_optim.zero_grad()
                total_actor_loss.backward()
                self.actor_optim.step()
                actor_losses.append(total_actor_loss.item())
            
            self.buffer.clear()
            
            # 打印训练信息
            if iteration % 10 == 0:
                print(f"\nIteration {iteration}")
                print(f"  Avg Critic Loss: {np.mean(critic_losses):.4f}")
                print(f"  Avg Actor Loss: {np.mean(actor_losses):.4f}")
                self.evaluate()
    
    def evaluate(self, num_episodes=5):
        """评估当前策略"""
        total_rewards = []
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            step = 0
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_normalized = self.actor.sample(state_tensor).cpu().numpy()[0]

                action = action_normalized * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()
                action = np.asarray(action, dtype=np.float32).reshape(self.env.action_space.shape)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
                step += 1
                
                # Pendulum最大步数为200
                if step >= 200:
                    break
            
            total_rewards.append(episode_reward)
        
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        print(f"Evaluation - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # 保存最佳模型
        if mean_reward > self.best_eval_reward:
            self.best_eval_reward = mean_reward
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'best_eval_reward': float(self.best_eval_reward),
            }, 'best_model.pth')
            print(f"  New best model saved! (Reward: {mean_reward:.2f})")


# ============ 5. 运行训练 ============
if __name__ == "__main__":
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 创建训练器并开始训练
    trainer = DiffusionPPOTrainer(env_name="Pendulum-v1", device=device)
    
    # 可以先测试一下环境
    print("\nTesting environment...")
    test_env = gym.make("Pendulum-v1", render_mode=None)
    state, _ = test_env.reset()
    print(f"Sample state: {state}")
    print(f"State shape: {state.shape}")
    print(f"Action sample: {test_env.action_space.sample()}")
    test_env.close()
    
    print("\nStarting training...\n")
    trainer.train(num_iterations=200, steps_per_iteration=2048)