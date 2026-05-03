import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Set, List, Optional
import imageio


class Visualizer:
    """Handles all visualisation effects and video saving for an episode."""

    def __init__(self, env, save_video: bool, video_path: Optional[Path],
                 agent_view: bool, fog_of_war: bool, show_trail: bool, as_gif: bool):
        self.env = env
        self.save_video = save_video
        self.agent_view = agent_view
        self.fog_of_war = fog_of_war
        self.show_trail = show_trail
        self.as_gif = as_gif

        self.visited: Set[Tuple[int, int]] = set()
        self.trail: List[Tuple[int, int, int]] = []  # (y, x, step)
        self.frames = []

        self.video_writer = None
        if save_video and video_path:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            if as_gif:
                self.video_path = video_path
            else:
                h, w = env.render_size, env.render_size
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (w, h))

    def reset(self):
        """Reset visualisation state for a new episode."""
        self.visited.clear()
        self.trail.clear()
        self.frames.clear()
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def render(self, step: int) -> np.ndarray:
        """
        Get the environment frame, apply visual effects, and optionally store/display.
        Returns the processed frame (ready for display).
        """
        raw_frame = self.env.render()
        if raw_frame is None:
            return None

        frame = raw_frame.copy()
        y, x = self.env.agent_pos
        cell_size = self.env._cell_size
        grid_size = self.env.grid_size

        # Update state: mark current cell and its 3x3 neighbourhood for fog of war
        self.visited.add((int(y), int(x)))
        if self.fog_of_war:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = int(y) + dy, int(x) + dx
                    if 0 <= ny < grid_size and 0 <= nx < grid_size:
                        self.visited.add((ny, nx))

        self.trail.append((int(y), int(x), step))

        # Apply fog of war (preserves UI text)
        if self.fog_of_war:
            frame = self._apply_fog_of_war(frame, raw_frame, grid_size, cell_size)

        # Apply agent view (crop)
        crop_offset = (0, 0)
        if self.agent_view:
            frame, crop_offset = self._apply_agent_view(frame, y, x, cell_size)

        # Draw trail with alpha blending
        if self.show_trail:
            frame = self._draw_trail_alpha(frame, cell_size, step, crop_offset)

        # Store for video
        if self.save_video:
            if self.as_gif:
                self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            elif self.video_writer:
                self.video_writer.write(frame)

        return frame

    def finalize(self):
        """Close video writer and save GIF if needed."""
        if self.save_video and self.as_gif and self.frames:
            imageio.mimsave(self.video_path, self.frames, duration=50, loop=0)
            print(f"✓ Saved GIF to {self.video_path}")
        elif self.video_writer:
            self.video_writer.release()

    # ---- Private helper methods ----
    def _apply_fog_of_war(self, frame, original_frame, grid_size, cell_size):
        """
        Black out cells not visited, but keep the UI text from original_frame.
        """
        # First, black out grid cells
        for yy in range(grid_size):
            for xx in range(grid_size):
                if (yy, xx) not in self.visited:
                    y0, y1 = yy * cell_size, (yy+1) * cell_size
                    x0, x1 = xx * cell_size, (xx+1) * cell_size
                    frame[y0:y1, x0:x1] = (0, 0, 0)

        # Now copy back the text area (assuming text is drawn at top-left)
        # The environment draws two lines of text at y=15 and y=35 (as per env.render())
        # We'll copy these small rectangular regions from original_frame.
        text_region_height = 46
        if original_frame.shape[0] > text_region_height and original_frame.shape[1] > 0:
            frame[0:text_region_height, 0:original_frame.shape[1]] = original_frame[0:text_region_height, 0:original_frame.shape[1]]
        return frame

    def _apply_agent_view(self, frame, y, x, cell_size, view_size=3):
        win_size = cell_size * view_size
        half = win_size // 2
        cx = int((x + 0.5) * cell_size)
        cy = int((y + 0.5) * cell_size)
        top = cy - half
        bottom = cy + half
        left = cx - half
        right = cx + half

        cropped = np.zeros((win_size, win_size, 3), dtype=np.uint8)
        src_top = max(0, top)
        src_bottom = min(frame.shape[0], bottom)
        src_left = max(0, left)
        src_right = min(frame.shape[1], right)
        dst_top = src_top - top
        dst_bottom = dst_top + (src_bottom - src_top)
        dst_left = src_left - left
        dst_right = dst_left + (src_right - src_left)
        if dst_top < win_size and dst_left < win_size:
            cropped[dst_top:dst_bottom, dst_left:dst_right] = frame[src_top:src_bottom, src_left:src_right]
        return cropped, (left, top)

    def _draw_trail_alpha(self, frame, cell_size, current_step, crop_offset, trail_length=30):
        """
        Draw a trail that fades linearly to zero over `trail_length` steps.
        Only the last `trail_length` segments of the trail are rendered.
        """
        if len(self.trail) < 2:
            return frame

        left, top = crop_offset

        # Create an alpha map (single channel, 0-255)
        alpha_map = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Draw segments in chronological order (oldest first becomes background,
        # newer segments overwrite – that’s fine because newer = brighter)
        for i in range(len(self.trail) - 1):
            y1, x1, s1 = self.trail[i]
            y2, x2, s2 = self.trail[i + 1]

            steps_ago = current_step - s1
            if steps_ago > trail_length:
                continue                        # segment completely faded out, skip drawing

            # Linear fade: 1.0 for steps_ago=0 → 0.0 for steps_ago=trail_length
            alpha = int(255 * np.exp(-steps_ago / (trail_length / 3)))
            if alpha == 0:
                continue

            # Compute pixel positions, adjusted for the crop offset
            p1 = (int((x1 + 0.5) * cell_size) - left, int((y1 + 0.5) * cell_size) - top)
            p2 = (int((x2 + 0.5) * cell_size) - left, int((y2 + 0.5) * cell_size) - top)

            # Draw the line segment with the calculated alpha value
            cv2.line(alpha_map, p1, p2, alpha, thickness=max(1, cell_size // 20))

        # Blend the frame with a white overlay using the alpha map
        alpha_norm = alpha_map.astype(np.float32) / 255.0
        alpha_norm = np.expand_dims(alpha_norm, axis=2)          # shape (H, W, 1)
        blended = (frame.astype(np.float32) * (1 - alpha_norm) + 255 * alpha_norm).astype(np.uint8)
        return blended