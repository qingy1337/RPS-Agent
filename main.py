import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass
from enum import Enum
import math
import random

# Configuration and Data Classes
@dataclass
class Config:
    """Configuration class for the Rock-Paper-Scissors team battle environment"""
    # Environment parameters
    grid_width: float = 50.0
    grid_height: float = 50.0
    
    # Agent parameters
    agents_per_team: int = 12
    agent_radius: float = 0.5
    agent_max_speed: float = 2.0
    detection_radius: float = 8.0
    collision_radius: float = 1.0
    
    # Visualization
    window_width: int = 800
    window_height: int = 600
    fps: int = 60
    
    # Training parameters
    max_steps: int = 5000
    reward_conversion: float = 10.0
    reward_being_converted: float = -10.0
    reward_team_growth: float = 1.0
    reward_survival_bonus: float = 0.1

class Team(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2

class Agent:
    """Individual agent in the environment"""
    def __init__(self, agent_id: int, team: Team, position: np.ndarray, config: Config):
        self.id = agent_id
        self.team = team
        self.position = position.copy()
        self.velocity = np.zeros(2)
        self.config = config
        self.converted_this_step = False
        self.conversions_made = 0
        self.times_converted = 0
        
    def can_beat(self, other_team: Team) -> bool:
        """Check if this agent's team beats another team"""
        if self.team == Team.ROCK and other_team == Team.SCISSORS:
            return True
        elif self.team == Team.PAPER and other_team == Team.ROCK:
            return True
        elif self.team == Team.SCISSORS and other_team == Team.PAPER:
            return True
        return False
    
    def get_color(self) -> Tuple[int, int, int]:
        """Get RGB color for visualization"""
        colors = {
            Team.ROCK: (200, 50, 50),      # Red
            Team.PAPER: (50, 50, 200),     # Blue
            Team.SCISSORS: (50, 200, 50)   # Green
        }
        return colors[self.team]

class TeamBattleEnv(gym.Env):
    """
    Rock-Paper-Scissors Team Battle Environment
    
    A continuous 2D environment where agents from three teams (Rock, Paper, Scissors)
    compete by converting opponents through collisions based on RPS rules.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, config: Optional[Config] = None, render_mode: Optional[str] = None):
        super().__init__()
        
        self.config = config if config else Config()
        self.render_mode = render_mode
        
        # Initialize agents
        self.agents: List[Agent] = []
        self.total_agents = self.config.agents_per_team * 3
        
        # Pygame initialization for rendering
        self.screen = None
        self.clock = None
        self.font = None
        
        # Metrics tracking
        self.step_count = 0
        self.team_history = []
        self.conversion_history = []
        
        # Define observation and action spaces
        # Observation: [own_x, own_y, own_vx, own_vy, own_team, nearby_agents_info, team_counts]
        # Nearby agents: [relative_x, relative_y, team] for each agent in detection radius
        max_nearby_agents = 20  # Maximum number of nearby agents to observe
        obs_size = 5 + (max_nearby_agents * 3) + 3  # own_state + nearby_agents + team_counts
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Action space: [velocity_x, velocity_y]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Clear existing agents
        self.agents.clear()
        self.step_count = 0
        self.team_history.clear()
        self.conversion_history.clear()
        
        # Create agents for each team
        agent_id = 0
        for team in Team:
            for _ in range(self.config.agents_per_team):
                # Random position within bounds
                position = np.array([
                    np.random.uniform(self.config.agent_radius, 
                                    self.config.grid_width - self.config.agent_radius),
                    np.random.uniform(self.config.agent_radius, 
                                    self.config.grid_height - self.config.agent_radius)
                ])
                
                agent = Agent(agent_id, team, position, self.config)
                self.agents.append(agent)
                agent_id += 1
        
        # Record initial team counts
        self._record_team_counts()
        
        # Return observation for the first agent (in multi-agent, you'd return dict)
        obs = self._get_observation(0) if self.agents else np.zeros(self.observation_space.shape)
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        self.step_count += 1
        
        # For simplicity, we'll control the first agent and have others act randomly
        # In a full multi-agent setup, you'd receive actions for all agents
        
        # Apply action to the first agent
        if len(self.agents) > 0:
            self._apply_action(0, action)
        
        # Random actions for other agents (in training, these would be from their policies)
        for i in range(1, len(self.agents)):
            random_action = np.random.uniform(-1, 1, 2)
            self._apply_action(i, random_action)
        
        # Update agent positions
        self._update_positions()
        
        # Handle collisions and conversions
        conversions = self._handle_collisions()
        
        # Calculate rewards
        reward = self._calculate_reward(0, conversions)
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.step_count >= self.config.max_steps
        
        # Record metrics
        self._record_team_counts()
        if conversions:
            self.conversion_history.append((self.step_count, conversions))
        
        # Get observation and info
        obs = self._get_observation(0) if self.agents else np.zeros(self.observation_space.shape)
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, agent_idx: int, action: np.ndarray):
        """Apply action to an agent"""
        if agent_idx >= len(self.agents):
            return
            
        agent = self.agents[agent_idx]
        
        # Normalize action and apply speed limit
        action = np.clip(action, -1, 1)
        agent.velocity = action * self.config.agent_max_speed
    
    def _update_positions(self):
        """Update all agent positions based on their velocities"""
        for agent in self.agents:
            # Update position
            agent.position += agent.velocity * (1.0 / self.config.fps)
            
            # Boundary handling (bounce off walls)
            if agent.position[0] <= self.config.agent_radius or agent.position[0] >= self.config.grid_width - self.config.agent_radius:
                agent.velocity[0] *= -0.8  # Some energy loss on bounce
                agent.position[0] = np.clip(agent.position[0], 
                                          self.config.agent_radius, 
                                          self.config.grid_width - self.config.agent_radius)
            
            if agent.position[1] <= self.config.agent_radius or agent.position[1] >= self.config.grid_height - self.config.agent_radius:
                agent.velocity[1] *= -0.8
                agent.position[1] = np.clip(agent.position[1], 
                                          self.config.agent_radius, 
                                          self.config.grid_height - self.config.agent_radius)
    
    def _handle_collisions(self) -> List[Tuple[int, int, Team, Team]]:
        """Handle collisions between agents and return conversion information"""
        conversions = []
        
        for i, agent1 in enumerate(self.agents):
            agent1.converted_this_step = False
            
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                # Calculate distance between agents
                distance = np.linalg.norm(agent1.position - agent2.position)
                
                if distance <= self.config.collision_radius:
                    # Collision detected
                    if agent1.team != agent2.team:
                        # Determine winner
                        if agent1.can_beat(agent2.team):
                            # Agent1 wins, convert agent2
                            old_team = agent2.team
                            agent2.team = agent1.team
                            agent2.converted_this_step = True
                            agent1.conversions_made += 1
                            agent2.times_converted += 1
                            conversions.append((j, i, old_team, agent1.team))
                            
                        elif agent2.can_beat(agent1.team):
                            # Agent2 wins, convert agent1
                            old_team = agent1.team
                            agent1.team = agent2.team
                            agent1.converted_this_step = True
                            agent2.conversions_made += 1
                            agent1.times_converted += 1
                            conversions.append((i, j, old_team, agent2.team))
                    
                    # Simple collision physics (bounce apart)
                    if distance > 0:
                        direction = (agent1.position - agent2.position) / distance
                        overlap = self.config.collision_radius - distance
                        agent1.position += direction * overlap * 0.5
                        agent2.position -= direction * overlap * 0.5
        
        return conversions
    
    def _calculate_reward(self, agent_idx: int, conversions: List) -> float:
        """Calculate reward for the specified agent"""
        if agent_idx >= len(self.agents):
            return 0.0
        
        agent = self.agents[agent_idx]
        reward = 0.0
        
        # Individual rewards
        for conversion in conversions:
            converted_idx, converter_idx, old_team, new_team = conversion
            
            if converter_idx == agent_idx:
                # Agent made a conversion
                reward += self.config.reward_conversion
            elif converted_idx == agent_idx:
                # Agent was converted
                reward += self.config.reward_being_converted
        
        # Team-based rewards
        team_counts = self._get_team_counts()
        agent_team_count = team_counts[agent.team.value]
        total_agents = sum(team_counts.values())
        
        if total_agents > 0:
            team_ratio = agent_team_count / total_agents
            reward += self.config.reward_team_growth * team_ratio
        
        # Survival bonus
        reward += self.config.reward_survival_bonus
        
        return reward
    
    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """Get observation for the specified agent"""
        if agent_idx >= len(self.agents):
            return np.zeros(self.observation_space.shape)
        
        agent = self.agents[agent_idx]
        obs_parts = []
        
        # Own state: [x, y, vx, vy, team]
        obs_parts.extend([
            agent.position[0] / self.config.grid_width,  # Normalize position
            agent.position[1] / self.config.grid_height,
            agent.velocity[0] / self.config.agent_max_speed,  # Normalize velocity
            agent.velocity[1] / self.config.agent_max_speed,
            agent.team.value / 2.0  # Normalize team (0, 0.5, 1.0)
        ])
        
        # Nearby agents information
        nearby_agents = []
        for other_agent in self.agents:
            if other_agent.id != agent.id:
                distance = np.linalg.norm(agent.position - other_agent.position)
                if distance <= self.config.detection_radius:
                    # Relative position and team
                    rel_pos = other_agent.position - agent.position
                    nearby_agents.append([
                        rel_pos[0] / self.config.detection_radius,
                        rel_pos[1] / self.config.detection_radius,
                        other_agent.team.value / 2.0
                    ])
        
        # Sort by distance and take closest agents
        nearby_agents.sort(key=lambda x: x[0]**2 + x[1]**2)
        
        # Pad or truncate to fixed size
        max_nearby = 20
        for i in range(max_nearby):
            if i < len(nearby_agents):
                obs_parts.extend(nearby_agents[i])
            else:
                obs_parts.extend([0.0, 0.0, 0.0])  # Padding
        
        # Team counts
        team_counts = self._get_team_counts()
        total = sum(team_counts.values())
        if total > 0:
            obs_parts.extend([
                team_counts[Team.ROCK.value] / total,
                team_counts[Team.PAPER.value] / total,
                team_counts[Team.SCISSORS.value] / total
            ])
        else:
            obs_parts.extend([0.0, 0.0, 0.0])
        
        return np.array(obs_parts, dtype=np.float32)
    
    def _get_team_counts(self) -> Dict[int, int]:
        """Get current count of agents per team"""
        counts = {Team.ROCK.value: 0, Team.PAPER.value: 0, Team.SCISSORS.value: 0}
        for agent in self.agents:
            counts[agent.team.value] += 1
        return counts
    
    def _record_team_counts(self):
        """Record current team counts for metrics"""
        counts = self._get_team_counts()
        self.team_history.append((self.step_count, counts.copy()))
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate (one team dominates)"""
        team_counts = self._get_team_counts()
        active_teams = sum(1 for count in team_counts.values() if count > 0)
        return active_teams <= 1
    
    def _get_info(self) -> Dict:
        """Get additional information about the environment state"""
        team_counts = self._get_team_counts()
        
        return {
            "step": self.step_count,
            "team_counts": team_counts,
            "total_agents": len(self.agents),
            "dominant_team": max(team_counts, key=team_counts.get) if team_counts else None,
            "conversions_this_episode": len(self.conversion_history)
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render for human viewing using Pygame"""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.config.window_width, self.config.window_height))
            pygame.display.set_caption("Rock-Paper-Scissors Team Battle")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        if self.font is None:
            self.font = pygame.font.Font(None, 36)
        
        # Clear screen
        self.screen.fill((30, 30, 30))  # Dark gray background
        
        # Calculate scaling factors
        scale_x = self.config.window_width / self.config.grid_width
        scale_y = self.config.window_height / self.config.grid_height
        
        # Draw agents
        for agent in self.agents:
            screen_x = int(agent.position[0] * scale_x)
            screen_y = int(agent.position[1] * scale_y)
            radius = max(3, int(self.config.agent_radius * min(scale_x, scale_y)))
            
            color = agent.get_color()
            
            # Draw agent
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), radius)
            
            # Draw velocity vector (optional)
            if np.linalg.norm(agent.velocity) > 0.1:
                end_x = screen_x + agent.velocity[0] * 10
                end_y = screen_y + agent.velocity[1] * 10
                pygame.draw.line(self.screen, color, (screen_x, screen_y), (int(end_x), int(end_y)), 2)
        
        # Draw UI information
        team_counts = self._get_team_counts()
        y_offset = 10
        
        texts = [
            f"Step: {self.step_count}",
            f"Rock: {team_counts[Team.ROCK.value]}",
            f"Paper: {team_counts[Team.PAPER.value]}",
            f"Scissors: {team_counts[Team.SCISSORS.value]}"
        ]
        
        colors = [(255, 255, 255), (200, 50, 50), (50, 50, 200), (50, 200, 50)]
        
        for text, color in zip(texts, colors):
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (10, y_offset))
            y_offset += 30
        
        pygame.display.flip()
        self.clock.tick(self.config.fps)
    
    def _render_rgb_array(self):
        """Render and return RGB array"""
        # This would require implementing rendering to a surface and converting to array
        # For simplicity, returning None here
        return None
    
    def close(self):
        """Clean up resources"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

# Training Integration Classes

class MultiAgentWrapper:
    """Wrapper to handle multiple agents in the environment"""
    
    def __init__(self, config: Config = None):
        self.config = config if config else Config()
        self.env = TeamBattleEnv(config, render_mode="human")
        self.agents_count = self.config.agents_per_team * 3
    
    def train_with_random_policy(self, episodes: int = 100):
        """Train agents using random policy (baseline)"""
        results = {
            'episode_lengths': [],
            'final_team_counts': [],
            'conversion_counts': []
        }
        
        for episode in range(episodes):
            obs, info = self.env.reset()
            done = False
            step_count = 0
            
            while not done:
                # Random action
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                step_count += 1
                
                # Render every few steps
                if step_count % 10 == 0:
                    self.env.render()
            
            results['episode_lengths'].append(step_count)
            results['final_team_counts'].append(info['team_counts'])
            results['conversion_counts'].append(info['conversions_this_episode'])
            
            print(f"Episode {episode + 1}: Length={step_count}, "
                  f"Final counts={info['team_counts']}, "
                  f"Conversions={info['conversions_this_episode']}")
        
        return results

# Analysis and Visualization Tools

class ExperimentAnalyzer:
    """Analyze and visualize experiment results"""
    
    @staticmethod
    def plot_team_dynamics(team_history: List[Tuple[int, Dict]], title: str = "Team Dynamics"):
        """Plot team population over time"""
        if not team_history:
            return
        
        steps = [entry[0] for entry in team_history]
        rock_counts = [entry[1][Team.ROCK.value] for entry in team_history]
        paper_counts = [entry[1][Team.PAPER.value] for entry in team_history]
        scissors_counts = [entry[1][Team.SCISSORS.value] for entry in team_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, rock_counts, 'r-', label='Rock', linewidth=2)
        plt.plot(steps, paper_counts, 'b-', label='Paper', linewidth=2)
        plt.plot(steps, scissors_counts, 'g-', label='Scissors', linewidth=2)
        
        plt.xlabel('Time Steps')
        plt.ylabel('Agent Count')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def analyze_dominance_patterns(results: Dict):
        """Analyze patterns in team dominance"""
        final_counts = results['final_team_counts']
        
        # Count wins per team
        wins = {Team.ROCK.value: 0, Team.PAPER.value: 0, Team.SCISSORS.value: 0}
        
        for counts in final_counts:
            winner = max(counts, key=counts.get)
            if counts[winner] > 0:
                wins[winner] += 1
        
        print("Win Distribution:")
        print(f"Rock: {wins[Team.ROCK.value]} ({wins[Team.ROCK.value]/len(final_counts)*100:.1f}%)")
        print(f"Paper: {wins[Team.PAPER.value]} ({wins[Team.PAPER.value]/len(final_counts)*100:.1f}%)")
        print(f"Scissors: {wins[Team.SCISSORS.value]} ({wins[Team.SCISSORS.value]/len(final_counts)*100:.1f}%)")
        
        avg_length = np.mean(results['episode_lengths'])
        avg_conversions = np.mean(results['conversion_counts'])
        
        print(f"\nAverage Episode Length: {avg_length:.1f} steps")
        print(f"Average Conversions per Episode: {avg_conversions:.1f}")

# Example Usage and Configuration

def create_custom_config(
    grid_size: Tuple[float, float] = (50.0, 50.0),
    agents_per_team: int = 12,
    agent_speed: float = 2.0,
    detection_radius: float = 8.0
) -> Config:
    """Create a custom configuration"""
    config = Config()
    config.grid_width, config.grid_height = grid_size
    config.agents_per_team = agents_per_team
    config.agent_max_speed = agent_speed
    config.detection_radius = detection_radius
    return config

def run_experiment(config: Config = None, episodes: int = 10, render: bool = True):
    """Run a complete experiment with the given configuration"""
    print("Starting Rock-Paper-Scissors Team Battle Experiment")
    print("=" * 50)
    
    if config is None:
        config = Config()
    
    print(f"Configuration:")
    print(f"  Grid Size: {config.grid_width} x {config.grid_height}")
    print(f"  Agents per Team: {config.agents_per_team}")
    print(f"  Max Speed: {config.agent_max_speed}")
    print(f"  Detection Radius: {config.detection_radius}")
    print()
    
    # Create environment wrapper
    wrapper = MultiAgentWrapper(config)
    
    # Run training/simulation
    results = wrapper.train_with_random_policy(episodes)
    
    # Analyze results
    ExperimentAnalyzer.analyze_dominance_patterns(results)
    
    # Plot team dynamics for the last episode
    if wrapper.env.team_history:
        ExperimentAnalyzer.plot_team_dynamics(
            wrapper.env.team_history, 
            title="Team Dynamics - Final Episode"
        )
    
    wrapper.env.close()
    return results

# Integration with Popular RL Libraries

def setup_stable_baselines3_training():
    """Example setup for Stable Baselines3 integration"""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        
        # Create vectorized environment
        def make_env():
            return TeamBattleEnv(Config(), render_mode=None)
        
        env = make_vec_env(make_env, n_envs=4)
        
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            tensorboard_log="./rps_tensorboard/"
        )
        
        print("Stable Baselines3 PPO model created successfully!")
        print("To train: model.learn(total_timesteps=100000)")
        
        return model, env
        
    except ImportError:
        print("Stable Baselines3 not installed. Install with: pip install stable-baselines3")
        return None, None
        
# For ACTUAL RL training, use this:
def train_rl_agents(config: Config = None, total_timesteps: int = 100000):
    """Actually train RL agents using PPO"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    import os
    
    if config is None:
        config = Config()
    
    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Create environment
    env = TeamBattleEnv(config, render_mode=None)
    env = Monitor(env, "./logs")
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log="./logs/tensorboard/"
    )
    
    # Setup callbacks for checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="rps_model"
    )
    
    # Train the model
    print("Starting RL training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("./models/rps_final_model")
    print("Training completed and model saved!")
    
    return model, env

import pygame
import numpy as np
from typing import Optional

class InteractiveVisualizer:
    """Interactive pygame visualizer for the RPS environment"""
    
    def __init__(self, config: Config):
        self.config = config
        self.env = TeamBattleEnv(config, render_mode=None)
        
        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.window_width, self.config.window_height))
        pygame.display.set_caption("RPS Team Battle - Interactive")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)
        
        # Control states
        self.paused = False
        self.speed_multiplier = 1.0
        self.show_velocities = True
        self.show_detection_radius = False
        self.selected_agent = 0
        
        # Colors
        self.colors = {
            'background': (20, 20, 30),
            'ui_text': (255, 255, 255),
            'ui_panel': (40, 40, 50),
            'rock': (220, 60, 60),
            'paper': (60, 60, 220),
            'scissors': (60, 220, 60),
            'velocity': (255, 255, 100),
            'detection': (100, 100, 100, 50)  # Transparent
        }
    
    def run_visualization(self, episodes: int = 5):
        """Run interactive visualization"""
        print("Interactive Visualization Controls:")
        print("SPACE - Pause/Resume")
        print("↑/↓ - Speed up/down")
        print("V - Toggle velocity vectors")
        print("D - Toggle detection radius")
        print("R - Reset episode")
        print("Q/ESC - Quit")
        print("CLICK - Select agent")
        
        for episode in range(episodes):
            print(f"\nStarting Episode {episode + 1}")
            obs, info = self.env.reset()
            done = False
            step_count = 0
            
            while not done:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                        elif event.key == pygame.K_UP:
                            self.speed_multiplier = min(5.0, self.speed_multiplier * 1.5)
                        elif event.key == pygame.K_DOWN:
                            self.speed_multiplier = max(0.1, self.speed_multiplier / 1.5)
                        elif event.key == pygame.K_v:
                            self.show_velocities = not self.show_velocities
                        elif event.key == pygame.K_d:
                            self.show_detection_radius = not self.show_detection_radius
                        elif event.key == pygame.K_r:
                            obs, info = self.env.reset()
                            step_count = 0
                        elif event.key in [pygame.K_q, pygame.K_ESCAPE]:
                            return
                    
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        # Select nearest agent
                        mouse_pos = pygame.mouse.get_pos()
                        self._select_agent_near_mouse(mouse_pos)
                
                # Update simulation
                if not self.paused:
                    for _ in range(int(self.speed_multiplier)):
                        if not done:
                            action = self.env.action_space.sample()  # Random for demo
                            obs, reward, terminated, truncated, info = self.env.step(action)
                            done = terminated or truncated
                            step_count += 1
                
                # Render everything
                self._render_frame(step_count, info)
                self.clock.tick(60)
            
            print(f"Episode {episode + 1} completed in {step_count} steps")
            print(f"Final team counts: {info['team_counts']}")
            
            # Wait for space or click to continue
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            waiting = False
                        elif event.key in [pygame.K_q, pygame.K_ESCAPE]:
                            return
                
                # Show episode complete screen
                self._render_episode_complete(episode + 1, step_count, info)
                self.clock.tick(30)
        
        pygame.quit()
    
    def _select_agent_near_mouse(self, mouse_pos):
        """Select the agent nearest to mouse click"""
        scale_x = self.config.window_width / self.config.grid_width
        scale_y = self.config.window_height / self.config.grid_height
        
        # Convert mouse position to world coordinates
        world_x = mouse_pos[0] / scale_x
        world_y = mouse_pos[1] / scale_y
        world_pos = np.array([world_x, world_y])
        
        min_distance = float('inf')
        nearest_agent = 0
        
        for i, agent in enumerate(self.env.agents):
            distance = np.linalg.norm(agent.position - world_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_agent = i
        
        self.selected_agent = nearest_agent
    
    def _render_frame(self, step_count, info):
        """Render a single frame"""
        self.screen.fill(self.colors['background'])
        
        scale_x = self.config.window_width / self.config.grid_width
        scale_y = self.config.window_height / self.config.grid_height
        
        # Draw detection radius for selected agent
        if (self.show_detection_radius and 
            self.selected_agent < len(self.env.agents)):
            agent = self.env.agents[self.selected_agent]
            screen_x = int(agent.position[0] * scale_x)
            screen_y = int(agent.position[1] * scale_y)
            radius = int(self.config.detection_radius * min(scale_x, scale_y))
            
            # Create transparent surface for detection radius
            detection_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(detection_surf, self.colors['detection'], 
                             (radius, radius), radius)
            self.screen.blit(detection_surf, (screen_x - radius, screen_y - radius))
        
        # Draw agents
        for i, agent in enumerate(self.env.agents):
            screen_x = int(agent.position[0] * scale_x)
            screen_y = int(agent.position[1] * scale_y)
            radius = max(4, int(self.config.agent_radius * min(scale_x, scale_y)))
            
            color = agent.get_color()
            
            # Highlight selected agent
            if i == self.selected_agent:
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                 (screen_x, screen_y), radius + 3, 2)
            
            # Draw agent
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), radius)
            
            # Draw velocity vector
            if self.show_velocities and np.linalg.norm(agent.velocity) > 0.1:
                end_x = screen_x + agent.velocity[0] * 15
                end_y = screen_y + agent.velocity[1] * 15
                pygame.draw.line(self.screen, self.colors['velocity'],
                               (screen_x, screen_y), (int(end_x), int(end_y)), 2)
                # Arrowhead
                pygame.draw.circle(self.screen, self.colors['velocity'],
                                 (int(end_x), int(end_y)), 3)
        
        # Draw UI
        self._draw_ui(step_count, info)
        
        pygame.display.flip()
    
    def _draw_ui(self, step_count, info):
        """Draw user interface elements"""
        # Team counts
        team_counts = info['team_counts']
        y_pos = 10
        
        texts = [
            f"Step: {step_count}",
            f"Rock: {team_counts[0]} agents",
            f"Paper: {team_counts[1]} agents", 
            f"Scissors: {team_counts[2]} agents"
        ]
        colors = [self.colors['ui_text'], self.colors['rock'], 
                 self.colors['paper'], self.colors['scissors']]
        
        for text, color in zip(texts, colors):
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (10, y_pos))
            y_pos += 25
        
        # Controls info
        y_pos += 10
        control_texts = [
            f"Speed: {self.speed_multiplier:.1f}x {'(PAUSED)' if self.paused else ''}",
            f"Velocities: {'ON' if self.show_velocities else 'OFF'}",
            f"Detection: {'ON' if self.show_detection_radius else 'OFF'}"
        ]
        
        for text in control_texts:
            surface = self.font.render(text, True, self.colors['ui_text'])
            self.screen.blit(surface, (10, y_pos))
            y_pos += 20
        
        # Selected agent info
        if self.selected_agent < len(self.env.agents):
            agent = self.env.agents[self.selected_agent]
            y_pos += 10
            agent_texts = [
                f"Selected Agent #{self.selected_agent}",
                f"Team: {agent.team.name}",
                f"Position: ({agent.position[0]:.1f}, {agent.position[1]:.1f})",
                f"Velocity: ({agent.velocity[0]:.1f}, {agent.velocity[1]:.1f})",
                f"Conversions made: {agent.conversions_made}",
                f"Times converted: {agent.times_converted}"
            ]
            
            for text in agent_texts:
                surface = self.font.render(text, True, self.colors['ui_text'])
                self.screen.blit(surface, (10, y_pos))
                y_pos += 18
    
    def _render_episode_complete(self, episode_num, steps, info):
        """Render episode completion screen"""
        self.screen.fill(self.colors['background'])
        
        # Main message
        title = f"Episode {episode_num} Complete!"
        title_surface = self.big_font.render(title, True, self.colors['ui_text'])
        title_rect = title_surface.get_rect(center=(self.config.window_width//2, 100))
        self.screen.blit(title_surface, title_rect)
        
        # Stats
        team_counts = info['team_counts']
        winner = max(team_counts, key=team_counts.get)
        winner_names = {0: "Rock", 1: "Paper", 2: "Scissors"}
        
        stats_texts = [
            f"Duration: {steps} steps",
            f"Winner: {winner_names[winner]} ({team_counts[winner]} agents)",
            f"Final counts - Rock: {team_counts[0]}, Paper: {team_counts[1]}, Scissors: {team_counts[2]}",
            "",
            "Press SPACE to continue, Q to quit"
        ]
        
        y_pos = 200
        for text in stats_texts:
            if text:  # Skip empty lines
                surface = self.font.render(text, True, self.colors['ui_text'])
                text_rect = surface.get_rect(center=(self.config.window_width//2, y_pos))
                self.screen.blit(surface, text_rect)
            y_pos += 30
        
        pygame.display.flip()

# Demo function
def run_interactive_demo():
    """Run the interactive visualization demo"""
    config = create_custom_config(
        grid_size=(40.0, 30.0),
        agents_per_team=10,
        agent_speed=2.0,
        detection_radius=6.0
    )
    
    visualizer = InteractiveVisualizer(config)
    visualizer.run_visualization(episodes=3)

import os
import json
import pickle
from datetime import datetime

class ModelManager:
    """Handle model saving, loading, and checkpoints"""
    
    def __init__(self, base_dir: str = "./models"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(f"{base_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{base_dir}/configs", exist_ok=True)
    
    def save_model_with_config(self, model, config: Config, name: str = None):
        """Save model with its configuration"""
        if name is None:
            name = f"rps_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = f"{self.base_dir}/{name}"
        config_path = f"{self.base_dir}/configs/{name}_config.json"
        
        # Save model
        model.save(model_path)
        
        # Save config
        config_dict = {
            'grid_width': config.grid_width,
            'grid_height': config.grid_height,
            'agents_per_team': config.agents_per_team,
            'agent_radius': config.agent_radius,
            'agent_max_speed': config.agent_max_speed,
            'detection_radius': config.detection_radius,
            'collision_radius': config.collision_radius,
            'max_steps': config.max_steps,
            'reward_conversion': config.reward_conversion,
            'reward_being_converted': config.reward_being_converted,
            'reward_team_growth': config.reward_team_growth,
            'reward_survival_bonus': config.reward_survival_bonus
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Model and config saved as: {name}")
        return model_path, config_path
    
    def load_model_with_config(self, name: str):
        """Load model with its configuration"""
        from stable_baselines3 import PPO
        
        model_path = f"{self.base_dir}/{name}"
        config_path = f"{self.base_dir}/configs/{name}_config.json"
        
        # Load config
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        # Load model
        model = PPO.load(model_path)
        
        return model, config

def complete_training_example():
    """Complete example of training with checkpoints and visualization"""
    
    # 1. Create configuration
    config = create_custom_config(
        grid_size=(40.0, 40.0),
        agents_per_team=15,
        agent_speed=9.0
    )
    
    # 2. Setup model manager
    manager = ModelManager()
    
    # 3. Train the model
    print("Starting training...")
    model, env = train_rl_agents(config, total_timesteps=1000000)
    
    # 4. Save model and config
    model_name = "rps_trained_model"
    manager.save_model_with_config(model, config, model_name)
    
    # 5. Test the trained model with visualization
    print("Testing trained model...")
    test_env = TeamBattleEnv(config, render_mode="human")
    
    obs, info = test_env.reset()
    for step in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        
        if terminated or truncated:
            break
    
    test_env.close()
    print("Training and testing complete!")

# Usage examples:
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Interactive Visualization Demo")
    print("2. Complete Training Example")
    print("3. Quick Random Simulation")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        run_interactive_demo()
    elif choice == "2":
        complete_training_example()
    elif choice == "3":
        config = create_custom_config(agents_per_team=8)
        run_experiment(config, episodes=3)
    