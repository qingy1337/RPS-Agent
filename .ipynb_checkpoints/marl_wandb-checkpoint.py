"""
Multi-Agent Rock-Paper-Scissors Battle RL Framework
Features: Train/Inference modes, Pygame visualization, Checkpoint system
"""

import logging
import argparse
import json
import os
import time
import cv2
from typing import Optional
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
import torch
import gymnasium as gym
from gymnasium import spaces
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn

### ADDED ###
# Import the W&B callback. Make sure you have wandb installed: pip install wandb
try:
    from ray.air.integrations.wandb import WandbLoggerCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
### END ADDED ###


@dataclass
class AgentState:
    """State of a single agent"""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    team: int = 0  # 0: Rock, 1: Paper, 2: Scissors
    agent_id: str = ""
    alive: bool = True
    conversion_timer: float = 0.0


class RPSBattleEnv(MultiAgentEnv):
    """Multi-agent Rock-Paper-Scissors battle environment"""
    
    ROCK, PAPER, SCISSORS = 0, 1, 2
    TEAM_COLORS = {
        0: (255, 100, 100),  # Red for Rock
        1: (100, 100, 255),  # Blue for Paper
        2: (100, 255, 100),  # Green for Scissors
    }
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Environment parameters
        self.grid_size = config.get("grid_size", 100.0)
        self.agents_per_team = config.get("agents_per_team", 10)
        self.max_speed = config.get("max_speed", 2.0)
        self.collision_radius = config.get("collision_radius", 2.0)
        self.dt = config.get("dt", 0.1)
        self.max_steps = config.get("max_steps", 1000)
        
        # Agent tracking
        self.agents: Dict[str, AgentState] = {}
        self.step_count = 0
        
        # Define action and observation spaces
        self._agent_ids = [f"agent_{i}" for i in range(self.agents_per_team * 3)]
        
        # Action space: [velocity_x, velocity_y] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation space: Fixed size regardless of agent count
        # [own_x, own_y, own_team] + [other_x, other_y, other_team] * max_other_agents
        max_other_agents = len(self._agent_ids) - 1
        obs_dim = 3 + max_other_agents * 3
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
        
        self.step_count = 0
        self.agents.clear()
        
        # Store agent teams for policy mapping
        agent_teams = {}
        
        # Initialize agents for each team
        for team in range(3):
            for i in range(self.agents_per_team):
                agent_id = f"agent_{team * self.agents_per_team + i}"
                
                # Random position with team clustering
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(self.grid_size * 0.2, self.grid_size * 0.4)
                center_offset = team * 2 * np.pi / 3
                
                x = self.grid_size / 2 + radius * np.cos(angle + center_offset)
                y = self.grid_size / 2 + radius * np.sin(angle + center_offset)
                
                self.agents[agent_id] = AgentState(
                    x=x, y=y, team=team, agent_id=agent_id
                )
                agent_teams[agent_id] = team
        
        return self._get_observations(), {"agent_teams": agent_teams}
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents"""
        obs = {}
        
        # Get list of all agents (for consistent ordering)
        all_agents = [(aid, agent) for aid, agent in self.agents.items()]
        
        for agent_id, agent in all_agents:
            # Own state (normalized to [-1, 1] range)
            agent_obs = [
                np.clip((agent.x / self.grid_size) * 2 - 1, -1, 1),
                np.clip((agent.y / self.grid_size) * 2 - 1, -1, 1),
                np.clip((agent.team / 2.0) * 2 - 1, -1, 1)
            ]
            
            # Other agents' states (fixed order)
            for other_id, other in all_agents:
                if other_id != agent_id:
                    agent_obs.extend([
                        np.clip((other.x / self.grid_size) * 2 - 1, -1, 1),
                        np.clip((other.y / self.grid_size) * 2 - 1, -1, 1),
                        np.clip((other.team / 2.0) * 2 - 1, -1, 1)
                    ])
            
            # Pad to fixed size
            while len(agent_obs) < self.observation_space.shape[0]:
                agent_obs.extend([-2.0, -2.0, -2.0])
            
            # Truncate to exact size and ensure correct dtype
            obs_array = np.array(agent_obs[:self.observation_space.shape[0]], dtype=np.float32)
            obs_array = np.clip(obs_array, -2.0, 2.0)
            obs[agent_id] = obs_array
        
        return obs
    
    def step(self, actions: Dict[str, np.ndarray]):
        """Execute one environment step"""
        self.step_count += 1
        
        # Update agent velocities based on actions
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                # Ensure action is the right shape and type
                if isinstance(action, (list, tuple)):
                    action = np.array(action, dtype=np.float32)
                elif not isinstance(action, np.ndarray):
                    action = np.array([action], dtype=np.float32)
                
                if len(action) >= 2:
                    agent.vx = np.clip(float(action[0]), -1.0, 1.0) * self.max_speed
                    agent.vy = np.clip(float(action[1]), -1.0, 1.0) * self.max_speed
        
        # Update positions
        for agent in self.agents.values():
            agent.x += agent.vx * self.dt
            agent.y += agent.vy * self.dt
            
            # Boundary conditions
            agent.x = np.clip(agent.x, 0, self.grid_size)
            agent.y = np.clip(agent.y, 0, self.grid_size)
            
            # Update conversion timer
            if agent.conversion_timer > 0:
                agent.conversion_timer -= self.dt
        
        # Initialize rewards for all agents
        rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
        
        # Check collisions and conversions
        self._handle_collisions(rewards)
        
        # Calculate team-based rewards
        team_counts = self._count_teams()
        total_agents = sum(team_counts.values())
        
        if total_agents > 0:
            for agent_id, agent in self.agents.items():
                # Reward for team dominance
                rewards[agent_id] += team_counts[agent.team] / total_agents
                
                # Small penalty for being far from teammates (you already reduced this)
                teammate_distance = self._avg_teammate_distance(agent)
                if teammate_distance > 0:
                    rewards[agent_id] -= 0.002 * teammate_distance / self.grid_size  # Reduced from 0.01
                
                # NEW: Interaction incentive - reward for being near enemies
                enemy_proximity = self._calculate_enemy_proximity(agent)
                rewards[agent_id] += 0.02 / (enemy_proximity / self.grid_size + 0.1)  # Normalized
                
                # NEW: Contested area bonus - reward for being in mixed team areas
                contested_area_bonus = self._calculate_contested_area_bonus(agent)
                rewards[agent_id] += contested_area_bonus * 0.1
        
        # Check termination conditions
        terminated = {}
        truncated = {}
        
        # Check if episode should end
        episode_done = self.step_count >= self.max_steps
        winning_team = self._check_victory()
        
        if winning_team is not None:
            episode_done = True
            # Bonus for winning team
            for agent_id, agent in self.agents.items():
                if agent.team == winning_team:
                    rewards[agent_id] += 10.0
        
        # Set termination flags for all agents
        for agent_id in self.agents.keys():
            terminated[agent_id] = episode_done
            truncated[agent_id] = False
        
        terminated["__all__"] = episode_done
        truncated["__all__"] = False
        
        return self._get_observations(), rewards, terminated, truncated, {}
    
    def _handle_collisions(self, rewards: Dict[str, float]):
        """Handle agent collisions and team conversions"""
        agents_list = list(self.agents.values())
        
        for i, agent1 in enumerate(agents_list):
            for agent2 in agents_list[i+1:]:
                # Check collision
                try:
                    dist = np.sqrt((agent1.x - agent2.x)**2 + (agent1.y - agent2.y)**2)
                    if dist < self.collision_radius:
                        # Rock beats Scissors, Scissors beats Paper, Paper beats Rock
                        if (agent1.team == self.ROCK and agent2.team == self.SCISSORS) or \
                           (agent1.team == self.SCISSORS and agent2.team == self.PAPER) or \
                           (agent1.team == self.PAPER and agent2.team == self.ROCK):
                            # Agent1 wins
                            agent2.team = agent1.team
                            agent2.conversion_timer = 1.0
                            rewards[agent1.agent_id] += 1.0
                            rewards[agent2.agent_id] -= 0.5
                        elif agent1.team != agent2.team:
                            # Agent2 wins
                            agent1.team = agent2.team
                            agent1.conversion_timer = 1.0
                            rewards[agent2.agent_id] += 1.0
                            rewards[agent1.agent_id] -= 0.5
                except (ValueError, FloatingPointError):
                    # Handle any numerical errors gracefully
                    continue
    
    def _count_teams(self) -> Dict[int, int]:
        """Count agents per team"""
        counts = {0: 0, 1: 0, 2: 0}
        for agent in self.agents.values():
            counts[agent.team] += 1
        return counts
    
    def _check_victory(self) -> Optional[int]:
        """Check if one team has won"""
        counts = self._count_teams()
        alive_teams = [team for team, count in counts.items() if count > 0]
        return alive_teams[0] if len(alive_teams) == 1 else None
    
    def _avg_teammate_distance(self, agent: AgentState) -> float:
        """Calculate average distance to teammates"""
        distances = []
        for other in self.agents.values():
            if other.agent_id != agent.agent_id and other.team == agent.team:
                try:
                    dist = np.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
                    distances.append(dist)
                except (ValueError, FloatingPointError):
                    continue
        return np.mean(distances) if distances else 0.0

    def _calculate_enemy_proximity(self, agent: AgentState) -> float:
        """Calculate average distance to nearest enemies (lower = closer)"""
        enemy_distances = []
        
        for other in self.agents.values():
            if other.agent_id != agent.agent_id and other.team != agent.team:
                try:
                    dist = np.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
                    enemy_distances.append(dist)
                except (ValueError, FloatingPointError):
                    continue
        
        if not enemy_distances:
            return self.grid_size  # Max distance if no enemies
        
        # Return average distance to closest 3 enemies (or all if fewer than 3)
        closest_enemies = sorted(enemy_distances)[:3]
        return np.mean(closest_enemies)

    def _calculate_contested_area_bonus(self, agent: AgentState, radius: float = 15.0) -> float:
        """Calculate bonus for being in areas with multiple teams nearby"""
        teams_nearby = set()
        agents_in_radius = 0
        
        for other in self.agents.values():
            if other.agent_id != agent.agent_id:
                try:
                    dist = np.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
                    if dist <= radius:
                        teams_nearby.add(other.team)
                        agents_in_radius += 1
                except (ValueError, FloatingPointError):
                    continue
        
        # Bonus increases with number of different teams nearby
        team_diversity = len(teams_nearby)
        density_factor = min(agents_in_radius / 5.0, 1.0)  # Cap at 5 agents
        
        return team_diversity * density_factor
        
    def render(self):
        """Render is handled separately in inference mode"""
        pass


class CustomPPOModel(TorchModelV2, nn.Module):
    """Custom neural network for PPO agents"""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        hidden_size = model_config.get("fcnet_hiddens", [256, 256])
        
        layers = []
        prev_size = obs_space.shape[0]
        
        for size in hidden_size:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        
        self.base_model = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev_size, num_outputs)
        self.value_head = nn.Linear(prev_size, 1)
        
        self._value_out = None
    
    def forward(self, input_dict, state, seq_lens):
        features = self.base_model(input_dict["obs"])
        self._value_out = self.value_head(features)
        return self.policy_head(features), state
    
    def value_function(self):
        return self._value_out.squeeze(1)



def train_agents(args):
    """Training mode: Initialize and train multi-agent RL models"""
    print("Starting training mode...")
    
    # Register custom model
    ModelCatalog.register_custom_model("custom_ppo", CustomPPOModel)
    
    # Environment configuration
    env_config = {
        "grid_size": args.grid_size,
        "agents_per_team": args.agents_per_team,
        "max_speed": args.max_speed,
        "collision_radius": args.collision_radius,
        "max_steps": args.max_steps
    }
    
    # Create a sample environment to get agent IDs
    sample_env = RPSBattleEnv(env_config)
    agent_ids = list(sample_env.agents.keys())

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # Get team from episode's custom info
        if hasattr(episode, 'custom_info') and 'agent_teams' in episode.custom_info:
            team = episode.custom_info['agent_teams'].get(agent_id, 0)
        else:
            # Default to determining team from agent_id if we can't get it from the environment
            team = int(agent_id.split('_')[1]) // sample_env.agents_per_team
        return f"{'rock' if team == 0 else 'paper' if team == 1 else 'scissors'}_policy"
    
    # PPO configuration for multi-agent training
    config = (
        PPOConfig()
        .environment(RPSBattleEnv, env_config=env_config)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .env_runners(
            num_env_runners=max(1, args.num_workers),  # Ensure at least 1
            num_envs_per_env_runner=1,
            rollout_fragment_length='auto',
        )
        .training(
            train_batch_size=2000,
            lr=args.learning_rate,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            model={
                "custom_model": "custom_ppo",
                "fcnet_hiddens": [128, 128],
            }
        )
        .multi_agent(
            policies={
                "rock_policy": (None, sample_env.observation_space, sample_env.action_space, {}),
                "paper_policy": (None, sample_env.observation_space, sample_env.action_space, {}),
                "scissors_policy": (None, sample_env.observation_space, sample_env.action_space, {})
            },
            policy_mapping_fn=policy_mapping_fn
        )
        .resources(
            num_gpus=1,
        )
        .debugging(log_level="ERROR")
    )
    
    # Set up checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training with periodic checkpointing
    def trial_name_creator(trial):
        return f"RPSBattle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    ### ADDED: W&B Integration ###
    # Set up callbacks for logging. This will be an empty list if W&B is not used.
    callbacks = []
    if args.use_wandb:
        if WANDB_AVAILABLE:
            # A group for this entire training run
            run_group_name = f"RPSBattle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            callbacks.append(
                WandbLoggerCallback(
                    project=args.wandb_project,
                    group=run_group_name,
                    api_key_file=args.wandb_api_key_file,
                    log_config=True,  # Logs the Tune config dict
                )
            )
            print(f"Weights & Biases logging enabled. Project: '{args.wandb_project}', Group: '{run_group_name}'")
        else:
            print("Warning: --use-wandb flag was passed, but 'wandb' could not be imported. Please run 'pip install wandb' to use this feature.")
    ### END ADDED ###

    try:
        ### CHANGED: Added `callbacks` argument to tune.run ###
        
        analysis = tune.run(
            "PPO",
            config=config.to_dict(),
            stop={"training_iteration": args.training_iterations},
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_at_end=True,
            storage_path=os.path.join(os.getcwd(), str(checkpoint_dir)), # Use the checkpoint_dir directly
            trial_name_creator=trial_name_creator,
            verbose=1,
            callbacks=callbacks,  # Pass the list of callbacks here
        )
        
        # Replace the best_trial selection with:
        trials = analysis.trials
        last_trial = trials[-1]  # Get the last trial
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "env_config": env_config,
            "training_iterations": args.training_iterations,
            "best_checkpoint": last_trial.checkpoint.path if last_trial and last_trial.checkpoint else None,
            "final_reward": last_trial.last_result.get("env_runners/episode_return_mean", None) if last_trial and last_trial.last_result else None
        }
        
        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Training completed! Checkpoints saved to {checkpoint_dir}")
        if last_trial and last_trial.checkpoint:
            print(f"Final checkpoint: {last_trial.checkpoint.path}")
            
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("This might be due to Ray/RLlib version compatibility.")
        print("Please ensure you have compatible versions installed:")
        print("pip install ray[rllib] torch gymnasium pygame wandb")

class PygameVisualizer:
    # ... (This class remains unchanged)
    def __init__(self, env: RPSBattleEnv, width: int = 1200, height: int = 800, 
                 video_path: Optional[str] = None, fps: int = 30, headless: bool = False):
        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Rock-Paper-Scissors Battle")
        else:
            pygame.init()
            self.screen = pygame.Surface((width, height))
        
        self.env = env
        self.width = width
        self.height = height
        self.headless = headless
        
        # Video recording setup
        self.video_path = video_path
        self.video_writer = None
        self.fps = fps
        
        if video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            print(f"Recording video to: {video_path}")
        
        # Fonts
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Camera
        self.camera_scale = min(width * 0.7 / env.grid_size, 
                               height * 0.7 / env.grid_size)
        self.camera_x = width // 2
        self.camera_y = height // 2
        
        # HUD elements
        self.minimap_size = 150
        self.timeline_data = deque(maxlen=200)
        
        # Visual settings
        self.show_trails = True
        self.trail_positions = {agent_id: deque(maxlen=20) 
                               for agent_id in env.agents.keys()}
        
        # Controls
        self.paused = False
        self.speed_multiplier = 1.0
        self.clock = pygame.time.Clock()
        
        # Frame counter for video
        self.frame_count = 0
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        screen_x = int(self.camera_x + (x - self.env.grid_size / 2) * self.camera_scale)
        screen_y = int(self.camera_y + (y - self.env.grid_size / 2) * self.camera_scale)
        return screen_x, screen_y
    
    def render(self):
        """Render the current environment state"""
        self.screen.fill((20, 20, 20))
        
        # Draw grid boundary
        corners = [
            self.world_to_screen(0, 0),
            self.world_to_screen(self.env.grid_size, 0),
            self.world_to_screen(self.env.grid_size, self.env.grid_size),
            self.world_to_screen(0, self.env.grid_size)
        ]
        pygame.draw.polygon(self.screen, (50, 50, 50), corners, 2)
        
        # Draw trails
        if self.show_trails:
            for agent_id, trail in self.trail_positions.items():
                if len(trail) > 1 and agent_id in self.env.agents:
                    agent = self.env.agents[agent_id]
                    color = RPSBattleEnv.TEAM_COLORS[agent.team]
                    faded_color = tuple(c // 3 for c in color)
                    
                    points = [self.world_to_screen(x, y) for x, y in trail]
                    if len(points) > 1:
                        pygame.draw.lines(self.screen, faded_color, False, points, 1)
        
        # Draw agents
        for agent in self.env.agents.values():
            if not agent.alive:
                continue
            
            screen_x, screen_y = self.world_to_screen(agent.x, agent.y)
            color = RPSBattleEnv.TEAM_COLORS[agent.team]
            
            # Conversion effect
            if agent.conversion_timer > 0:
                flash = int(255 * (0.5 + 0.5 * np.sin(agent.conversion_timer * 20)))
                color = tuple(min(255, c + flash // 2) for c in color)
                pygame.draw.circle(self.screen, (255, 255, 255), 
                                 (screen_x, screen_y), 
                                 int(15 * self.camera_scale), 2)
            
            # Draw agent
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), 
                             int(2 * self.camera_scale))
            
            # Update trail
            if agent.agent_id in self.trail_positions:
                self.trail_positions[agent.agent_id].append((agent.x, agent.y))
        
        # Draw HUD
        self._draw_hud()
        
        # Add frame counter for video
        if self.video_writer:
            frame_text = self.font_small.render(f"Frame: {self.frame_count}", True, (200, 200, 200))
            self.screen.blit(frame_text, (self.width - 150, 10))
        
        # Save frame to video if recording
        if self.video_writer:
            self.save_frame()
            self.frame_count += 1
        
        # Only update display if not headless
        if not self.headless:
            pygame.display.flip()
    
    def save_frame(self):
        """Save current frame to video"""
        if self.video_writer:
            # Convert pygame surface to numpy array
            frame_array = pygame.surfarray.array3d(self.screen)
            # Rotate and flip to correct orientation for OpenCV
            frame_array = np.rot90(frame_array)
            frame_array = np.flipud(frame_array)
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            # Write frame
            self.video_writer.write(frame_bgr)
    
    def _draw_hud(self):
        """Draw HUD elements"""
        # Team counters
        team_counts = self.env._count_teams()
        y_offset = 10
        
        for team, count in team_counts.items():
            team_name = ["Rock", "Paper", "Scissors"][team]
            color = RPSBattleEnv.TEAM_COLORS[team]
            text = self.font_large.render(f"{team_name}: {count}", True, color)
            self.screen.blit(text, (10, y_offset))
            y_offset += 40
        
        # Controls info (skip when recording video)
        if not self.video_writer:
            control_texts = [
                f"Speed: {self.speed_multiplier:.1f}x",
                "SPACE: Pause/Resume",
                "+/-: Adjust Speed",
                "T: Toggle Trails"
            ]
            
            y_offset = self.height - 120
            for text in control_texts:
                surf = self.font_small.render(text, True, (200, 200, 200))
                self.screen.blit(surf, (10, y_offset))
                y_offset += 25
        
        # Minimap
        self._draw_minimap()
        
        # Timeline graph
        self._draw_timeline()
        
        # Statistics
        self._draw_statistics()
    
    def _draw_minimap(self):
        """Draw minimap showing agent positions"""
        minimap_x = self.width - self.minimap_size - 10
        minimap_y = 10
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), 
                        (minimap_x, minimap_y, self.minimap_size, self.minimap_size))
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (minimap_x, minimap_y, self.minimap_size, self.minimap_size), 2)
        
        # Agents on minimap
        scale = self.minimap_size / self.env.grid_size
        for agent in self.env.agents.values():
            if agent.alive:
                x = int(minimap_x + agent.x * scale)
                y = int(minimap_y + agent.y * scale)
                color = RPSBattleEnv.TEAM_COLORS[agent.team]
                pygame.draw.circle(self.screen, color, (x, y), 2)
    
    def _draw_timeline(self):
        """Draw dominance timeline graph"""
        if len(self.timeline_data) < 2:
            return
        
        graph_x = self.width - 310
        graph_y = self.minimap_size + 30
        graph_width = 300
        graph_height = 150
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 40), 
                        (graph_x, graph_y, graph_width, graph_height))
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (graph_x, graph_y, graph_width, graph_height), 2)
        
        # Title
        title = self.font_small.render("Team Dominance", True, (200, 200, 200))
        self.screen.blit(title, (graph_x + 10, graph_y - 25))
        
        # Plot lines for each team
        max_agents = sum(self.timeline_data[0].values()) if self.timeline_data else 1
        
        for team in range(3):
            color = RPSBattleEnv.TEAM_COLORS[team]
            points = []
            
            for i, counts in enumerate(self.timeline_data):
                x = graph_x + int(i * graph_width / len(self.timeline_data))
                y = graph_y + graph_height - int(counts[team] / max_agents * graph_height)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
    
    def _draw_statistics(self):
        """Draw additional statistics"""
        stats_y = self.height - 250
        
        # Calculate statistics
        if self.env.agents:
            # Average speed
            avg_speed = np.mean([np.sqrt(a.vx**2 + a.vy**2) 
                               for a in self.env.agents.values() if a.alive])
            
            # Largest cluster size
            clusters = self._find_clusters()
            largest_cluster = max(len(c) for c in clusters) if clusters else 0
            
            # Strategy entropy (diversity measure)
            team_counts = self.env._count_teams()
            total = sum(team_counts.values())
            if total > 0:
                probs = [count/total for count in team_counts.values() if count > 0]
                entropy = -sum(p * np.log(p) for p in probs if p > 0)
            else:
                entropy = 0
            
            stats = [
                f"Avg Speed: {avg_speed:.2f}",
                f"Largest Cluster: {largest_cluster}",
                f"Diversity: {entropy:.2f}"
            ]
            
            for i, stat in enumerate(stats):
                text = self.font_small.render(stat, True, (200, 200, 200))
                self.screen.blit(text, (10, stats_y + i * 25))
    
    def _find_clusters(self, threshold: float = 10.0) -> List[List[str]]:
        """Find agent clusters based on proximity"""
        clusters = []
        visited = set()
        
        for agent_id, agent in self.env.agents.items():
            if agent_id in visited or not agent.alive:
                continue
            
            # BFS to find connected agents
            cluster = [agent_id]
            queue = [agent_id]
            visited.add(agent_id)
            
            while queue:
                current_id = queue.pop(0)
                current = self.env.agents[current_id]
                
                for other_id, other in self.env.agents.items():
                    if other_id not in visited and other.alive:
                        dist = np.sqrt((current.x - other.x)**2 + (current.y - other.y)**2)
                        if dist < threshold:
                            visited.add(other_id)
                            queue.append(other_id)
                            cluster.append(other_id)
            
            clusters.append(cluster)
        
        return clusters
    
    def handle_events(self) -> bool:
        """Handle pygame events, return False to quit"""
        if self.headless:
            return True  # No events to handle in headless mode
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.speed_multiplier = min(4.0, self.speed_multiplier * 1.5)
                elif event.key == pygame.K_MINUS:
                    self.speed_multiplier = max(0.25, self.speed_multiplier / 1.5)
                elif event.key == pygame.K_t:
                    self.show_trails = not self.show_trails
                elif event.key == pygame.K_r:
                    # Reset camera
                    self.camera_scale = min(self.width * 0.7 / self.env.grid_size,
                                          self.height * 0.7 / self.env.grid_size)
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom with mouse wheel
                self.camera_scale *= 1.1 if event.y > 0 else 0.9
                self.camera_scale = max(0.1, min(5.0, self.camera_scale))
        
        # Camera panning with mouse
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0]:  # Left click
            mouse_dx, mouse_dy = pygame.mouse.get_rel()
            self.camera_x += mouse_dx
            self.camera_y += mouse_dy
        
        return True
    
    def update(self):
        """Update visualization state"""
        # Update timeline data
        team_counts = self.env._count_teams()
        self.timeline_data.append(team_counts.copy())
        
        # Control frame rate based on speed multiplier
        if not self.video_writer:  # Only use pygame clock when not recording video
            self.clock.tick(60 * self.speed_multiplier)
    
    def close(self):
        """Clean up resources"""
        if self.video_writer:
            self.video_writer.release()
            print(f"Video saved successfully to: {self.video_path}")
        if not self.headless:
            pygame.quit()
    

def inference_mode(args):
    """Inference mode: Load checkpoint and run with Pygame visualization or video recording"""
    print("Starting inference mode...")
    
    # Load checkpoint
    if args.load:
        checkpoint_path = Path(args.load)
    else:
        # Find latest checkpoint
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoints = list(checkpoint_dir.rglob("checkpoint_*")) # Use rglob to search subdirectories
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load metadata if available
    metadata_path = checkpoint_path.parent.parent / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            env_config = metadata.get("env_config", {})
    else:
        env_config = {
            "grid_size": args.grid_size,
            "agents_per_team": args.agents_per_team,
            "max_speed": args.max_speed,
            "collision_radius": args.collision_radius,
            "max_steps": args.max_steps
        }
    
    # Register custom model
    ModelCatalog.register_custom_model("custom_ppo", CustomPPOModel)
    
    try:
        # Initialize algorithm from checkpoint
        algo = PPO.from_checkpoint(checkpoint_path)
        
        # Create environment and visualizer
        env = RPSBattleEnv(env_config)
        
        # Determine if recording video
        video_path = None
        headless = False
        if args.save_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"rps_battle_{timestamp}.mp4"
            headless = True  # Run without display when saving video
            print(f"Recording video to: {video_path}")
        
        visualizer = PygameVisualizer(
            env, 
            width=args.video_width, 
            height=args.video_height,
            video_path=video_path,
            fps=args.video_fps,
            headless=headless
        )
        
        # Run simulation
        obs, _ = env.reset()
        running = True
        step_count = 0
        max_episodes = args.video_episodes if args.save_video else float('inf')
        episode_count = 0
        
        if not args.save_video:
            print("\nControls:")
            print("SPACE: Pause/Resume")
            print("+/-: Adjust speed")
            print("T: Toggle trails")
            print("Mouse drag: Pan camera")
            print("Mouse wheel: Zoom")
        else:
            print(f"Recording {max_episodes} episode(s) to video...")
        
        while running and episode_count < max_episodes:
            # Handle events
            running = visualizer.handle_events()
            
            if not visualizer.paused or args.save_video:  # Always run when recording video
                # In inference_mode function:
                actions = {}
                for agent_id, agent_obs in obs.items():
                    current_team = env.agents[agent_id].team  # Use current team, not initial team
                    policy_id = f"{'rock' if current_team == 0 else 'paper' if current_team == 1 else 'scissors'}_policy"
                    action = algo.compute_single_action(
                        agent_obs, 
                        policy_id=policy_id
                    )
                    actions[agent_id] = action
                
                # Step environment
                obs, rewards, terminated, truncated, info = env.step(actions)
                step_count += 1
                
                # Check for episode end
                if terminated.get("__all__", False):
                    episode_count += 1
                    print(f"Episode {episode_count} finished after {step_count} steps!")
                    winning_team = env._check_victory()
                    if winning_team is not None:
                        team_name = ["Rock", "Paper", "Scissors"][winning_team]
                        print(f"Winner: {team_name}!")
                    
                    if episode_count < max_episodes:
                        # Reset for new episode
                        obs, _ = env.reset()
                        step_count = 0
                        visualizer.trail_positions = {agent_id: deque(maxlen=20) 
                                                    for agent_id in env.agents.keys()}
            
            # Update and render visualization
            visualizer.update()
            visualizer.render()
            
            # Control frame rate for video recording
            if args.save_video:
                time.sleep(1.0 / args.video_fps)
        
        # Clean up
        visualizer.close()
        print("Inference mode ended.")
        
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("Please make sure the checkpoint path is correct and compatible.")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Rock-Paper-Scissors Battle RL Framework"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "eval"], 
        required=True,
        help="Execution mode: train or eval (inference)"
    )
    
    # Checkpoint management
    parser.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default="./checkpoints",
        help="Directory for saving/loading checkpoints"
    )
    parser.add_argument(
        "--load", 
        type=str, 
        default=None,
        help="Specific checkpoint path to load (eval mode)"
    )
    
    # Training parameters
    parser.add_argument(
        "--training-iterations", 
        type=int, 
        default=100,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--checkpoint-freq", 
        type=int, 
        default=10,
        help="Save checkpoint every N iterations"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=2,
        help="Number of parallel workers for training"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.0003,
        help="Learning rate for PPO"
    )
    
    # Environment parameters
    parser.add_argument(
        "--grid-size", 
        type=float, 
        default=100.0,
        help="Size of the grid world"
    )
    parser.add_argument(
        "--agents-per-team", 
        type=int, 
        default=10,
        help="Number of agents per team"
    )
    parser.add_argument(
        "--max-speed", 
        type=float, 
        default=2.0,
        help="Maximum agent movement speed"
    )
    parser.add_argument(
        "--collision-radius", 
        type=float, 
        default=2.0,
        help="Collision detection radius"
    )
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=1000,
        help="Maximum steps per episode"
    )
    # Video recording parameters (for eval mode)
    parser.add_argument(
        "--save-video", 
        action="store_true",
        help="Save visualization as video file instead of displaying (eval mode only)"
    )
    parser.add_argument(
        "--video-episodes", 
        type=int, 
        default=3,
        help="Number of episodes to record in video"
    )
    parser.add_argument(
        "--video-fps", 
        type=int, 
        default=30,
        help="Frames per second for video recording"
    )
    parser.add_argument(
        "--video-width", 
        type=int, 
        default=1200,
        help="Video width in pixels"
    )
    parser.add_argument(
        "--video-height", 
        type=int, 
        default=800,
        help="Video height in pixels"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Whether to log"
    )
    
    ### ADDED: W&B Command-line arguments ###
    wandb_group = parser.add_argument_group('Weights & Biases Options')
    wandb_group.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable logging to Weights & Biases (requires 'wandb' to be installed)."
    )
    wandb_group.add_argument(
        "--wandb-project",
        type=str,
        default="RPSBattleRL",
        help="Name of the W&B project to log to."
    )
    wandb_group.add_argument(
        "--wandb-api-key-file",
        type=str,
        default=None,
        help="Path to a file containing your W&B API key. If not set, uses environment variables or a prior `wandb login`."
    )
    ### END ADDED ###
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Initialize Ray
    # Note: I removed the hardcoded num_cpus to make it more portable
    if args.log:
        ray.init(ignore_reinit_error=True,logging_level=logging.DEBUG, log_to_driver=True)
    else:
        ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    try:
        if args.mode == "train":
            train_agents(args)
        elif args.mode == "eval":
            inference_mode(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise # Re-raise the exception to see the full traceback
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()