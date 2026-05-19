import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# ============ 复制模型定义（确保和训练时一致） ============
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
        t_onehot = F.one_hot(t, self.T).float()
        t_emb = self.time_embed(t_onehot)
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
        device = state.device
        t = torch.randint(0, self.T, (state.shape[0],), device=device)
        noise = torch.randn_like(action)
        
        alpha_bar = self.alpha_bars[t].view(-1, 1)
        noisy_action = torch.sqrt(alpha_bar) * action + torch.sqrt(1 - alpha_bar) * noise
        
        pred_noise = self.forward(noisy_action, t, state)
        log_prob = -F.mse_loss(pred_noise, noise, reduction='none').mean(dim=-1)
        return log_prob


def _format_env_action(action, action_space):
    """把动作整理成环境需要的shape和dtype（Pendulum需要shape=(1,)）"""
    action = np.asarray(action, dtype=np.float32).reshape(action_space.shape)
    return np.clip(action, action_space.low, action_space.high)


def visualize_trained_model(model_path="best_model.pth", num_episodes=5):
    """加载训练好的模型并使用Gymnasium渲染可视化"""
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 创建环境（关键：render_mode="human"）
    env = gym.make("Pendulum-v1", render_mode="human")
    
    # 加载模型
    state_dim = env.observation_space.shape[0]  # 3
    action_dim = env.action_space.shape[0]      # 1
    
    actor = DiffusionActor(state_dim, action_dim).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        actor.load_state_dict(checkpoint['actor'])
        print(f"✓ Model loaded successfully from {model_path}")
        if 'best_eval_reward' in checkpoint:
            print(f"  Best evaluation reward during training: {checkpoint.get('best_eval_reward', 'N/A')}")
    except FileNotFoundError:
        print(f"✗ Model file {model_path} not found. Using untrained model.")
    
    actor.eval()
    
    # 动作缩放（Pendulum动作范围是[-2, 2]）
    action_scale = torch.FloatTensor(env.action_space.high).to(device)
    action_bias = torch.FloatTensor((env.action_space.high + env.action_space.low) / 2).to(device)
    
    print(f"Action scale: {action_scale.cpu().numpy()}, Action bias: {action_bias.cpu().numpy()}")
    
    # 开始可视化
    print("\n" + "="*50)
    print("Starting visualization...")
    print("="*50 + "\n")
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        while not done and step < 200:  # Pendulum最大步数200
            # 转换状态为tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # 采样动作
            with torch.no_grad():
                action_normalized = actor.sample(state_tensor).cpu().numpy()[0]  # (action_dim,)

            # 缩放到实际动作空间，并整理成环境需要的数组shape
            action = action_normalized * action_scale.cpu().numpy() + action_bias.cpu().numpy()
            action = _format_env_action(action, env.action_space)

            # 执行动作（Pendulum期望 shape=(1,) 的数组）
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            step += 1
            
            # 可选：添加小延迟让渲染更清晰
            # import time
            # time.sleep(0.02)
        
        total_rewards.append(episode_reward)
        print(f"  Steps: {step}, Reward: {episode_reward:.2f}\n")
    
    env.close()
    
    # 打印统计信息
    print("="*50)
    print("Visualization Complete!")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Best Episode: {np.max(total_rewards):.2f}")
    print(f"Worst Episode: {np.min(total_rewards):.2f}")
    print("="*50)


def visualize_untrained_model(num_episodes=3):
    """可视化未训练的模型（随机策略）用于对比"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make("Pendulum-v1", render_mode="human")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    actor = DiffusionActor(state_dim, action_dim).to(device)
    actor.eval()
    
    action_scale = torch.FloatTensor(env.action_space.high).to(device)
    action_bias = torch.FloatTensor((env.action_space.high + env.action_space.low) / 2).to(device)
    
    print("\n" + "="*50)
    print("Visualizing UNTRAINED model (random policy)")
    print("="*50 + "\n")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        while not done and step < 200:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_normalized = actor.sample(state_tensor).cpu().numpy()[0]  # (action_dim,)

            action = action_normalized * action_scale.cpu().numpy() + action_bias.cpu().numpy()
            action = _format_env_action(action, env.action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            step += 1
        
        print(f"  Steps: {step}, Reward: {episode_reward:.2f}\n")
    
    env.close()


def visualize_random_actions(num_episodes=3):
    """可视化完全随机的动作（baseline）"""
    
    env = gym.make("Pendulum-v1", render_mode="human")
    
    print("\n" + "="*50)
    print("Visualizing RANDOM actions (baseline)")
    print("="*50 + "\n")
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        while not done and step < 200:
            # 完全随机的动作（保持环境原生shape）
            action = env.action_space.sample()
            action = _format_env_action(action, env.action_space)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            step += 1
        
        total_rewards.append(episode_reward)
        print(f"  Steps: {step}, Reward: {episode_reward:.2f}\n")
    
    env.close()
    print(f"Average random reward: {np.mean(total_rewards):.2f}")


# ============ 使用示例 ============
if __name__ == "__main__":
    # 选项1：可视化训练好的模型
    visualize_trained_model(model_path="best_model.pth", num_episodes=3)
    
    # 选项2：可视化未训练的模型（对比）
    # visualize_untrained_model(num_episodes=3)
    
    # 选项3：可视化随机动作作为baseline
    # visualize_random_actions(num_episodes=3)