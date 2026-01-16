# FILE: src/core/human_agent.py
"""
Human-controlled agent for playable mode
"""

import cv2
import numpy as np
from typing import Dict, Any

from src.core.constants import Actions


class HumanAgent:
    """Human-controlled agent using keyboard input"""
    
    def __init__(self):
        self.action_map = {
            ord('a'): Actions.LEFT,      # 'a' for left
            ord('d'): Actions.RIGHT,     # 'd' for right
            ord('w'): Actions.UP,        # 'w' for up
            ord('s'): Actions.DOWN,      # 's' for down
            ord(' '): Actions.STAY,      # space for stay
            ord('b'): Actions.BUTTON,    # 'b' for button
            81: Actions.LEFT,            # Left arrow key
            82: Actions.UP,              # Up arrow key
            83: Actions.RIGHT,           # Right arrow key
            84: Actions.DOWN,            # Down arrow key
            13: Actions.BUTTON,          # Enter key for button
        }
        
        self.key_descriptions = {
            Actions.LEFT: "A / Left Arrow",
            Actions.RIGHT: "D / Right Arrow",
            Actions.UP: "W / Up Arrow",
            Actions.DOWN: "S / Down Arrow",
            Actions.STAY: "Space",
            Actions.BUTTON: "B / Enter"
        }
    
    def act(self, observation: np.ndarray = None) -> int:
        """Get action from keyboard input"""
        print("\n" + "="*50)
        print("HUMAN CONTROL MODE - Waiting for input...")
        print("="*50)
        print("Controls:")
        print(f"  Move Left:    {self.key_descriptions[Actions.LEFT]}")
        print(f"  Move Right:   {self.key_descriptions[Actions.RIGHT]}")
        print(f"  Move Up:      {self.key_descriptions[Actions.UP]}")
        print(f"  Move Down:    {self.key_descriptions[Actions.DOWN]}")
        print(f"  Stay:         {self.key_descriptions[Actions.STAY]}")
        print(f"  Press Button: {self.key_descriptions[Actions.BUTTON]}")
        print("  Quit:         Q or Esc")
        print("="*50)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or Esc to quit
                print("Quitting human play mode...")
                return -1
            
            if key in self.action_map:
                action = self.action_map[key]
                action_name = Actions(action).name
                print(f"Action selected: {action_name}")
                return action
            
            print(f"Invalid key: {chr(key) if key < 128 else key}. Try again.")
    
    def test(self, env, episodes: int = 1, visualize: bool = True) -> Dict[str, Any]:
        """Test human agent performance"""
        rewards = []
        success_flags = []
        steps_list = []
        
        print("\n" + "="*60)
        print("HUMAN PLAY MODE STARTED")
        print("="*60)
        print(f"Task Class: {env.task_class}")
        print(f"Complexity Level: {env.complexity_level:.2f}")
        print(f"Max Steps: {env.max_steps}")
        print("="*60)
        
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            
            obs, info = env.reset()
            episode_reward = 0
            steps = 0
            terminated = truncated = False
            
            while not (terminated or truncated) and steps < env.max_steps:
                # Render the environment
                if visualize:
                    frame = env.render()
                    if frame is not None:
                        cv2.imshow('Human Play Mode', frame)
                
                # Get human action
                action = self.act()
                
                if action == -1:  # User quit
                    print("Episode ended early by user.")
                    break
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                
                print(f"Step {steps}: Action={Actions(action).name}, "
                      f"Reward={reward:.2f}, Energy={info['energy']:.1f}")
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            success_flags.append(steps == env.max_steps)
            steps_list.append(steps)
            
            print(f"\nEpisode {episode + 1} finished:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Steps: {steps}/{env.max_steps}")
            print(f"  Final Energy: {info['energy']:.1f}")
            
            if visualize:
                cv2.waitKey(1000)  # Pause between episodes
        
        if visualize:
            cv2.destroyAllWindows()
        
        return {
            'rewards': rewards,
            'success_flags': success_flags,
            'steps': steps_list,
            'avg_reward': np.mean(rewards) if rewards else 0,
            'success_rate': np.mean(success_flags) * 100 if success_flags else 0,
            'avg_steps': np.mean(steps_list) if steps_list else 0,
            'std_reward': np.std(rewards) if rewards else 0
        }