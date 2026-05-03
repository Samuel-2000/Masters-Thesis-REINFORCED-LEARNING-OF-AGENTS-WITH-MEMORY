# FILE: src/core/agent_human.py
"""
Human-controlled agent for playable mode
"""

import cv2
import numpy as np
from typing import Dict, Any

from src.core.constants import Actions
from src.visualization.visualizer import Visualizer
from pathlib import Path

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
    

    def test(self, env, args) -> Dict[str, Any]:
        """
        Human play mode with options from args.
        """
        rewards = []
        success_flags = []
        steps_list = []

        print("\n" + "="*60)
        print("HUMAN PLAY MODE STARTED")
        print("="*60)
        print(f"Task Class: {env.task_class}")
        print(f"Complexity Level: {env.complexity_level:.2f}")
        print(f"Max Steps: {env.max_steps}")
        print(f"Test epochs: {args.epochs}")
        print(f"Episodes per epoch: {args.consecutive_episodes}")
        print("="*60)

        print("\n" + "="*50)
        print("CONTROLS:")
        print(f"  Move Left:    {self.key_descriptions[Actions.LEFT]}")
        print(f"  Move Right:   {self.key_descriptions[Actions.RIGHT]}")
        print(f"  Move Up:      {self.key_descriptions[Actions.UP]}")
        print(f"  Move Down:    {self.key_descriptions[Actions.DOWN]}")
        print(f"  Stay:         {self.key_descriptions[Actions.STAY]}")
        print(f"  Press Button: {self.key_descriptions[Actions.BUTTON]}")
        print("  Quit:         Q or Esc")
        print("="*50)

        total_episodes = 0

        for epoch in range(args.epochs):
            obs, info = env.reset()
            print(f"\n--- EPOCH {epoch+1}/{args.epochs}: New grid (Type: {env.task_class}, Complexity: {env.complexity_level:.2f}) ---")

            for ep_in_epoch in range(args.consecutive_episodes):
                if ep_in_epoch > 0:
                    obs, info = env.soft_reset()
                    print("  Soft reset: same grid, new chance!")

                vid_name = f"human_{env.task_class}_comp_{env.complexity_level:.2f}_ep_{epoch}_{ep_in_epoch}"
                vid_path = Path("results/videos") / f"{vid_name}.{'gif' if args.as_gif else 'mp4'}" if args.save_video else None
                viz = Visualizer(env, args.save_video, vid_path, args.agent_view, args.fog_of_war, args.show_trail, args.as_gif)

                print(f"\nEpisode {total_episodes + 1} (epoch {epoch+1}, episode {ep_in_epoch+1}/{args.consecutive_episodes})")
                episode_reward = 0
                steps = 0
                terminated = truncated = False

                while not (terminated or truncated) and steps < env.max_steps:
                    frame = viz.render(steps)
                    if frame is not None:
                        cv2.imshow('Human Play Mode', frame)

                    action = self.act()
                    if action == -1:
                        print("Episode ended early by user.")
                        break

                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    print(f"Step {steps}: {Actions(action).name}, Reward={reward:.2f}, Energy={info['energy']:.1f}")

                    if terminated or truncated:
                        break

                viz.finalize()

                rewards.append(episode_reward)
                success_flags.append(steps == env.max_steps)
                steps_list.append(steps)

                print(f"\nEpisode finished: Reward={episode_reward:.2f}, Steps={steps}/{env.max_steps}, Final Energy={info['energy']:.1f}")
                cv2.waitKey(1000)
                total_episodes += 1

        cv2.destroyAllWindows()

        return {
            'rewards': rewards,
            'success_flags': success_flags,
            'steps': steps_list,
            'avg_reward': np.mean(rewards) if rewards else 0,
            'success_rate': np.mean(success_flags) * 100 if success_flags else 0,
            'avg_steps': np.mean(steps_list) if steps_list else 0,
            'std_reward': np.std(rewards) if rewards else 0,
            'total_episodes': total_episodes
        }