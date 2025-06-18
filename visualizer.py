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