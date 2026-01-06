"""
===============================================================================
ASSIGNMENT: AUTONOMOUS CAR NAVIGATION - VERSION 2 (WITH SAFETY OVERRIDE)
===============================================================================

This version includes a SAFETY SYSTEM that uses sensor data directly to prevent
crashes. Instead of waiting for the neural network to learn "obstacle = bad",
this version actively overrides dangerous actions.

Key Differences from Version 1:
1. Safety Override - Prevents actions that would lead to crashes
2. Wider FOV (150°) - Better junction/obstacle detection
3. Direct sensor utilization - Uses sensor data for immediate decision making

Good luck! 🚗💨
===============================================================================
"""

import sys
import os
import math
import numpy as np
import random
from collections import deque

# --- PYTORCH ---
import torch
import torch.nn as nn
import torch.optim as optim

# --- PYQT ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGraphicsScene, 
                             QGraphicsView, QGraphicsItem, QFrame, QFileDialog,
                             QTextEdit, QGridLayout)
from PyQt6.QtGui import (QImage, QPixmap, QColor, QPen, QBrush, QPainter, 
                         QPolygonF, QFont, QPainterPath)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

# ==========================================
# 1. CONFIGURATION & THEME
# ==========================================
# Nordic Theme
C_BG_DARK   = QColor("#2E3440") 
C_PANEL     = QColor("#3B4252")
C_INFO_BG   = QColor("#4C566A") 
C_ACCENT    = QColor("#88C0D0") 
C_TEXT      = QColor("#ECEFF4") 
C_SUCCESS   = QColor("#A3BE8C") 
C_FAILURE   = QColor("#BF616A") 
C_SENSOR_ON = QColor("#A3BE8C") # Green
C_SENSOR_OFF= QColor("#BF616A") # Red

# ==========================================
# PHYSICS PARAMETERS - WITH WIDER FOV
# ==========================================
CAR_WIDTH = 14     
CAR_HEIGHT = 8   
SENSOR_DIST = 50   # Sensor range for obstacle detection
SENSOR_ANGLE = 15   # WIDER: 25° between sensors = 150° total FOV (-75° to +75°)
SPEED = 2           # Slower speed for better control
TURN_SPEED = 8      # Effective turning for gradual curves (degrees/step)
SHARP_TURN = 25     # Sharp turn angle for tight corners (degrees)

# ==========================================
# REINFORCEMENT LEARNING HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 64     # Stable batch size for learning

GAMMA = 0.95        # Discount factor - agent plans ~20 steps ahead

LR = 0.001          # Stable learning rate for Adam optimizer

TAU = 0.005         # Soft target network updates (slow and stable)

MAX_CONSECUTIVE_CRASHES = 5    # Reset position after 5 consecutive crashes

# Target Colors (for multiple targets)
TARGET_COLORS = [
    QColor(0, 255, 255),      # Cyan
    QColor(255, 100, 255),    # Magenta
    QColor(0, 255, 100),      # Green
    QColor(255, 150, 0),      # Orange
    QColor(100, 150, 255),    # Blue
    QColor(255, 50, 150),     # Pink
    QColor(150, 255, 50),     # Lime
    QColor(255, 255, 0),      # Yellow
]

# ==========================================
# 2. NEURAL NETWORK - ENHANCED WITH EXTRA FC LAYER
# ==========================================
class DrivingDQN(nn.Module):
    """
    Deep Q-Network for autonomous car navigation.
    Architecture: Input(9) → FC(128) → FC(256) → FC(512) → FC(256) → FC(128) → Output(5)
    """
    def __init__(self, input_dim, output_dim):
        super(DrivingDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),      # Additional layer for increased capacity
            nn.ReLU(),
            nn.Linear(512, 256),      # Bottleneck back down
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

# ==========================================
# 3. PHYSICS & LOGIC - WITH SAFETY OVERRIDE
# ==========================================
class CarBrain:
    def __init__(self, map_image: QImage):
        self.map = map_image
        self.w, self.h = map_image.width(), map_image.height()
        # Calculate map diagonal for scale-invariant distance normalization
        self.map_diagonal = math.sqrt(self.w**2 + self.h**2)
        
        # RL Init
        self.input_dim = 9  # 7 sensors + angle_to_target + distance_to_target
        self.n_actions = 5  # 0: left, 1: straight, 2: right, 3: sharp left, 4: sharp right 
        self.policy_net = DrivingDQN(self.input_dim, self.n_actions)
        self.target_net = DrivingDQN(self.input_dim, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=10000)
        
        # Prioritized Replay: separate buffer for high-reward episodes
        self.priority_memory = deque(maxlen=3000)  # Store successful episodes
        self.current_episode_buffer = []  # Temporary buffer for current episode
        self.episode_scores = deque(maxlen=100)  # Track recent episode scores
        
        self.steps = 0
        self.epsilon = 1.0   # Start with 100% exploration
        self.base_epsilon = 1.0  # Track the naturally decayed epsilon
        self.consecutive_crashes = 0
        
        # Locations
        self.start_pos = QPointF(100, 100) 
        self.car_pos = QPointF(100, 100)   
        self.car_angle = 0
        self.target_pos = QPointF(200, 200)
        
        # Multiple Targets Support
        self.targets = []
        self.current_target_idx = 0
        self.targets_reached = 0
        self.best_target_reached = 0  # Track furthest target ever reached
        
        self.alive = True
        self.score = 0
        self.sensor_coords = [] 
        self.prev_dist = None

    def set_start_pos(self, point):
        self.start_pos = point
        self.car_pos = point

    def reset(self):
        self.alive = True
        self.score = 0
        self.car_pos = QPointF(self.start_pos.x(), self.start_pos.y())
        self.car_angle = random.randint(0, 360)
        self.current_target_idx = 0
        self.targets_reached = 0
        # Reset epsilon to base value (remove any frontier boost)
        self.epsilon = self.base_epsilon
        if len(self.targets) > 0:
            self.target_pos = self.targets[0]
        state, dist = self.get_state()
        self.prev_dist = dist
        return state
    
    def add_target(self, point):
        self.targets.append(QPointF(point.x(), point.y()))
        if len(self.targets) == 1:
            self.target_pos = self.targets[0]
            self.current_target_idx = 0
    
    def switch_to_next_target(self):
        if self.current_target_idx < len(self.targets) - 1:
            self.current_target_idx += 1
            self.target_pos = self.targets[self.current_target_idx]
            self.targets_reached += 1
            return True
        return False

    def get_state(self):
        sensor_vals = []
        self.sensor_coords = []
        # 7 sensors with SENSOR_ANGLE=25: angles = [-75, -50, -25, 0, 25, 50, 75] (150° FOV)
        angles = [(i - 3) * SENSOR_ANGLE for i in range(7)]
        
        for a in angles:
            rad = math.radians(self.car_angle + a)
            
            # RAY-CASTING: Check along the entire ray, not just the endpoint
            # Returns normalized distance to first obstacle (0 = obstacle at car, 1 = clear to SENSOR_DIST)
            obstacle_dist = SENSOR_DIST  # Assume clear initially
            step_size = 5  # Check every 5 pixels along the ray
            
            for dist in range(step_size, SENSOR_DIST + 1, step_size):
                check_x = self.car_pos.x() + math.cos(rad) * dist
                check_y = self.car_pos.y() + math.sin(rad) * dist
                
                if not (0 <= check_x < self.w and 0 <= check_y < self.h):
                    obstacle_dist = dist  # Hit map boundary
                    break
                    
                c = QColor(self.map.pixel(int(check_x), int(check_y)))
                brightness = (c.red() + c.green() + c.blue()) / 3.0 / 255.0
                
                if brightness < 0.4:  # Found obstacle (dark pixel)
                    obstacle_dist = dist
                    break
            
            # Store endpoint for visualization
            sx = self.car_pos.x() + math.cos(rad) * obstacle_dist
            sy = self.car_pos.y() + math.sin(rad) * obstacle_dist
            self.sensor_coords.append(QPointF(sx, sy))
            
            # Normalized sensor value: 0 = obstacle very close, 1 = clear path
            val = obstacle_dist / SENSOR_DIST
            sensor_vals.append(val)
            
        dx = self.target_pos.x() - self.car_pos.x()
        dy = self.target_pos.y() - self.car_pos.y()
        dist = math.sqrt(dx*dx + dy*dy)
        
        rad_to_target = math.atan2(dy, dx)
        angle_to_target = math.degrees(rad_to_target)
        
        angle_diff = (angle_to_target - self.car_angle) % 360
        if angle_diff > 180: angle_diff -= 360
        
        norm_dist = min(dist / self.map_diagonal, 1.0)
        norm_angle = angle_diff / 180.0
        
        state = sensor_vals + [norm_angle, norm_dist]
        return np.array(state, dtype=np.float32), dist

    def safety_override(self, action, sensors):
        """
        SAFETY SYSTEM: Override dangerous actions using sensor data directly.
        Sensors now return DISTANCE to obstacle (0=touching, 1=clear to max range).
        
        With SENSOR_ANGLE=25:
        Sensor indices: 0=-75°, 1=-50°, 2=-25°, 3=0°(center), 4=25°, 5=50°, 6=75°
        """
        # Thresholds based on normalized distance (sensor value = dist/SENSOR_DIST)
        DANGER_CLOSE = 0.15   # Obstacle within 15% of sensor range (~15px) = DANGER
        DANGER_MEDIUM = 0.3   # Obstacle within 30% (~30px) = caution
        SAFE_THRESHOLD = 0.5  # More than 50% clear = safe
        
        # Check which directions are clear (at different safety levels)
        far_left_safe = sensors[0] > SAFE_THRESHOLD
        left_safe = sensors[1] > SAFE_THRESHOLD or sensors[2] > SAFE_THRESHOLD
        center_safe = sensors[3] > SAFE_THRESHOLD
        right_safe = sensors[4] > SAFE_THRESHOLD or sensors[5] > SAFE_THRESHOLD
        far_right_safe = sensors[6] > SAFE_THRESHOLD
        
        # Check for immediate danger (very close obstacles)
        center_danger = sensors[3] < DANGER_MEDIUM or sensors[2] < DANGER_CLOSE or sensors[4] < DANGER_CLOSE
        left_danger = sensors[1] < DANGER_CLOSE or sensors[2] < DANGER_CLOSE
        right_danger = sensors[4] < DANGER_CLOSE or sensors[5] < DANGER_CLOSE
        
        # EMERGENCY: Center is in danger - must turn!
        if center_danger:
            # Find the safest direction
            left_score = sensors[0] + sensors[1] + sensors[2]
            right_score = sensors[4] + sensors[5] + sensors[6]
            
            if left_score > right_score and not left_danger:
                return 3  # Sharp left
            elif right_score > left_score and not right_danger:
                return 4  # Sharp right
            elif sensors[0] > sensors[6]:
                return 3  # Sharp left (far left is clearer)
            else:
                return 4  # Sharp right (far right is clearer)
        
        # If going straight but center not fully safe
        if action == 1 and not center_safe:
            if left_safe and not right_safe:
                return 0  # Gentle left
            elif right_safe and not left_safe:
                return 2  # Gentle right
            # Keep going straight but check for better options
            left_score = sensors[1] + sensors[2]
            right_score = sensors[4] + sensors[5]
            if left_score > right_score + 0.3:
                return 0
            elif right_score > left_score + 0.3:
                return 2
        
        # If turning left but left has danger
        if action in [0, 3] and left_danger:
            if center_safe:
                return 1  # Go straight instead
            elif right_safe:
                return 2  # Turn right instead
            elif far_right_safe:
                return 4  # Sharp right
            
        # If turning right but right has danger
        if action in [2, 4] and right_danger:
            if center_safe:
                return 1  # Go straight instead
            elif left_safe:
                return 0  # Turn left instead
            elif far_left_safe:
                return 3  # Sharp left
        
        return action  # Original action is acceptable

    def step(self, action):
        # SAFETY SYSTEM: Check sensors BEFORE executing action
        current_state, _ = self.get_state()
        sensors = current_state[:7]
        
        # Override dangerous actions with safe ones
        action = self.safety_override(action, sensors)
        
        turn = 0
        if action == 0:   # Left turn
            turn = -TURN_SPEED
        elif action == 1: # Straight
            turn = 0
        elif action == 2: # Right turn
            turn = TURN_SPEED
        elif action == 3: # Sharp left turn
            turn = -SHARP_TURN
        elif action == 4: # Sharp right turn
            turn = SHARP_TURN
        
        self.car_angle += turn
        rad = math.radians(self.car_angle)
        
        new_x = self.car_pos.x() + math.cos(rad) * SPEED
        new_y = self.car_pos.y() + math.sin(rad) * SPEED
        self.car_pos = QPointF(new_x, new_y)
        
        next_state, dist = self.get_state()
        sensors = next_state[:7]
        
        reward = -0.1
        done = False
        
        car_center_val = self.check_pixel(self.car_pos.x(), self.car_pos.y())
        
        if car_center_val < 0.4:
            reward = -100
            done = True
            self.alive = False
        elif dist < 25:  # Target detection radius
            reward = 100
            has_next = self.switch_to_next_target()
            if has_next:
                done = False
                _, new_dist = self.get_state()
                self.prev_dist = new_dist
                # Boost epsilon at frontier
                if self.targets_reached >= self.best_target_reached:
                    if self.targets_reached > self.best_target_reached:
                        self.best_target_reached = self.targets_reached
                    if self.epsilon < 0.3:
                        self.epsilon = 0.3
            else:
                self.best_target_reached = len(self.targets)
                done = True
        else:
            # Reward for clear center path
            center_sensor = next_state[3]
            reward += center_sensor * 10
            
            # Penalty for moving away from target
            if self.prev_dist is not None and dist > self.prev_dist:
                reward -= 5
            self.prev_dist = dist
            
        self.score += reward
        return next_state, reward, done

    def check_pixel(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            c = QColor(self.map.pixel(int(x), int(y)))
            return ((c.red() + c.green() + c.blue()) / 3.0) / 255.0
        return 0.0

    def optimize(self):
        total_memory_size = len(self.memory) + len(self.priority_memory)
        if total_memory_size < BATCH_SIZE: return 0
        
        success_rate = len(self.priority_memory) / max(total_memory_size, 1)
        priority_ratio = 0.3 + (success_rate * 0.4)
        
        priority_samples = int(BATCH_SIZE * priority_ratio)
        regular_samples = BATCH_SIZE - priority_samples
        
        batch = []
        
        if len(self.priority_memory) >= priority_samples:
            batch.extend(random.sample(self.priority_memory, priority_samples))
        else:
            batch.extend(list(self.priority_memory))
            regular_samples += priority_samples - len(self.priority_memory)
        
        if len(self.memory) >= regular_samples:
            batch.extend(random.sample(self.memory, regular_samples))
        else:
            batch.extend(list(self.memory))
        
        if len(batch) < BATCH_SIZE // 2:
            return 0
        
        s, a, r, ns, d = zip(*batch)
        s = torch.FloatTensor(np.array(s))
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        ns = torch.FloatTensor(np.array(ns))
        d = torch.FloatTensor(d).unsqueeze(1)
        
        q = self.policy_net(s).gather(1, a)
        next_q = self.target_net(ns).max(1)[0].detach().unsqueeze(1)
        target = r + GAMMA * next_q * (1 - d)
        
        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Natural epsilon decay
        if self.base_epsilon > 0.001: 
            self.base_epsilon *= 0.9999
        if self.epsilon > 0.001 and self.epsilon <= self.base_epsilon:
            self.epsilon = self.base_epsilon
        return loss.item()
    
    def store_experience(self, experience):
        self.current_episode_buffer.append(experience)
    
    def finalize_episode(self, episode_reward):
        if len(self.current_episode_buffer) == 0:
            return
        
        self.episode_scores.append(episode_reward)
        
        if not self.alive:
            self.consecutive_crashes += 1
        else:
            self.consecutive_crashes = 0
        
        if episode_reward > 0:
            for exp in self.current_episode_buffer:
                self.priority_memory.append(exp)
        else:
            for exp in self.current_episode_buffer:
                self.memory.append(exp)
        
        self.current_episode_buffer = []

# ==========================================
# 4. CUSTOM WIDGETS (VISUALS)
# ==========================================
class RewardChart(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        self.scores = []
        self.max_points = 50

    def update_chart(self, new_score):
        self.scores.append(new_score)
        if len(self.scores) > self.max_points:
            self.scores.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        painter.fillRect(0, 0, w, h, C_PANEL)
        
        if len(self.scores) < 2:
            return

        min_val = min(self.scores)
        max_val = max(self.scores)
        if max_val == min_val: max_val += 1
        
        points = []
        step_x = w / (self.max_points - 1)
        
        for i, score in enumerate(self.scores):
            x = i * step_x
            ratio = (score - min_val) / (max_val - min_val)
            y = h - (ratio * (h * 0.8) + (h * 0.1))
            points.append(QPointF(x, y))

        path = QPainterPath()
        path.moveTo(points[0])
        for p in points[1:]:
            path.lineTo(p)
            
        pen = QPen(C_ACCENT, 2)
        painter.setPen(pen)
        painter.drawPath(path)
        
        if len(self.scores) >= 2:
            avg_points = []
            window_size = 10
            
            for i in range(len(self.scores)):
                start_idx = max(0, i - window_size + 1)
                avg_score = sum(self.scores[start_idx:i+1]) / (i - start_idx + 1)
                
                x = i * step_x
                ratio = (avg_score - min_val) / (max_val - min_val)
                y = h - (ratio * (h * 0.8) + (h * 0.1))
                avg_points.append(QPointF(x, y))
            
            if len(avg_points) > 1:
                avg_path = QPainterPath()
                avg_path.moveTo(avg_points[0])
                for p in avg_points[1:]:
                    avg_path.lineTo(p)
                
                avg_pen = QPen(QColor(255, 215, 0), 3)
                painter.setPen(avg_pen)
                painter.drawPath(avg_path)
        
        if min_val < 0 and max_val > 0:
            zero_ratio = (0 - min_val) / (max_val - min_val)
            y_zero = h - (zero_ratio * (h * 0.8) + (h * 0.1))
            painter.setPen(QPen(QColor(255, 255, 255, 50), 1, Qt.PenStyle.DashLine))
            painter.drawLine(0, int(y_zero), w, int(y_zero))
        
        legend_x = 10
        legend_y = 15
        
        painter.setPen(QPen(C_ACCENT, 2))
        painter.drawLine(legend_x, legend_y, legend_x + 20, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(legend_x + 25, legend_y + 4, "Raw")
        
        painter.setPen(QPen(QColor(255, 215, 0), 3))
        painter.drawLine(legend_x + 60, legend_y, legend_x + 80, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(legend_x + 85, legend_y + 4, "Avg (10)")

class SensorItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(90)
        self.pulse = 0
        self.pulse_speed = 0.3
        self.is_detecting = True
        
    def set_detecting(self, detecting):
        self.is_detecting = detecting
        self.update()
    
    def boundingRect(self):
        return QRectF(-4, -4, 8, 8)
    
    def paint(self, painter, option, widget):
        self.pulse += self.pulse_speed
        if self.pulse > 1.0:
            self.pulse = 0
        
        if self.is_detecting:
            color = C_SENSOR_ON
            outer_alpha = int(150 * (1 - self.pulse))
        else:
            color = C_SENSOR_OFF
            outer_alpha = int(200 * (1 - self.pulse))
        
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        outer_size = 3 + (2 * self.pulse)
        outer_color = QColor(color)
        outer_color.setAlpha(outer_alpha)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outer_color))
        painter.drawEllipse(QPointF(0, 0), outer_size, outer_size)
        
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(0, 0), 2, 2)

class CarItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(100)
        self.brush = QBrush(C_ACCENT)
        self.pen = QPen(Qt.GlobalColor.white, 1)

    def boundingRect(self):
        return QRectF(-CAR_WIDTH/2, -CAR_HEIGHT/2, CAR_WIDTH, CAR_HEIGHT)

    def paint(self, painter, option, widget):
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawRoundedRect(self.boundingRect(), 2, 2)
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(int(CAR_WIDTH/2)-2, -3, 2, 6)

class TargetItem(QGraphicsItem):
    def __init__(self, color=None, is_active=True, number=1):
        super().__init__()
        self.setZValue(50)
        self.pulse = 0
        self.growing = True
        self.color = color if color else QColor(0, 255, 255)
        self.is_active = is_active
        self.number = number

    def set_active(self, active):
        self.is_active = active
        self.update()
    
    def set_color(self, color):
        self.color = color
        self.update()

    def boundingRect(self):
        return QRectF(-20, -20, 40, 40)

    def paint(self, painter, option, widget):
        if self.is_active:
            if self.growing:
                self.pulse += 0.5
                if self.pulse > 10: self.growing = False
            else:
                self.pulse -= 0.5
                if self.pulse < 0: self.growing = True
            
            r = 10 + self.pulse
            painter.setPen(Qt.PenStyle.NoPen)
            outer_color = QColor(self.color)
            outer_color.setAlpha(100)
            painter.setBrush(QBrush(outer_color)) 
            painter.drawEllipse(QPointF(0,0), r, r)
            painter.setBrush(QBrush(self.color)) 
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawEllipse(QPointF(0,0), 8, 8)
        else:
            dimmed_color = QColor(self.color)
            dimmed_color.setAlpha(120)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.setBrush(QBrush(dimmed_color))
            painter.drawEllipse(QPointF(0,0), 6, 6)
        
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(-10, -10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(self.number))

# ==========================================
# 5. APP
# ==========================================
class NeuralNavApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuralNav V2: DQN + Safety Override")
        self.resize(1300, 850)
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {C_BG_DARK.name()}; }}
            QLabel {{ color: {C_TEXT.name()}; font-family: Segoe UI; font-size: 13px; }}
            QPushButton {{ background-color: {C_PANEL.name()}; color: white; border: 1px solid {C_INFO_BG.name()}; padding: 8px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: {C_INFO_BG.name()}; }}
            QPushButton:checked {{ background-color: {C_ACCENT.name()}; color: black; }}
            QTextEdit {{ background-color: {C_PANEL.name()}; color: #D8DEE9; border: none; font-family: Consolas; font-size: 11px; }}
            QFrame {{ border: none; }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # LEFT PANEL
        panel = QFrame()
        panel.setFixedWidth(280)
        panel.setStyleSheet(f"background-color: {C_BG_DARK.name()};")
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(10)
        
        lbl_title = QLabel("CONTROLS (V2 - Safety)")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        vbox.addWidget(lbl_title)
        
        self.lbl_status = QLabel("1. Click Map -> CAR\n2. Click Map -> TARGET(S)\n   (Multiple clicks for sequence)")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; padding: 10px; border-radius: 5px; color: #E5E9F0;")
        vbox.addWidget(self.lbl_status)

        self.btn_run = QPushButton("▶ START (Space)")
        self.btn_run.setCheckable(True)
        self.btn_run.setEnabled(False) 
        self.btn_run.clicked.connect(self.toggle_training)
        vbox.addWidget(self.btn_run)
        
        self.btn_reset = QPushButton("↺ RESET ALL")
        self.btn_reset.clicked.connect(self.full_reset)
        vbox.addWidget(self.btn_reset)
        
        self.btn_load = QPushButton("📂 LOAD MAP")
        self.btn_load.clicked.connect(self.load_map_dialog)
        vbox.addWidget(self.btn_load)

        vbox.addSpacing(15)
        vbox.addWidget(QLabel("REWARD HISTORY"))
        self.chart = RewardChart()
        vbox.addWidget(self.chart)

        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(10, 10, 10, 10)
        
        self.val_eps = QLabel("1.00")
        self.val_eps.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Epsilon:"), 0,0)
        sf_layout.addWidget(self.val_eps, 0,1)
        
        self.val_rew = QLabel("0")
        self.val_rew.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Last Reward:"), 1,0)
        sf_layout.addWidget(self.val_rew, 1,1)
        
        vbox.addWidget(stats_frame)

        vbox.addWidget(QLabel("LOGS"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)

        main_layout.addWidget(panel)

        # RIGHT PANEL
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet(f"border: 2px solid {C_PANEL.name()}; background-color: {C_BG_DARK.name()}")
        self.view.mousePressEvent = self.on_scene_click
        main_layout.addWidget(self.view)

        # Logic - Load default map
        self.setup_map("maps/map3.jpg") 
        self.setup_state = 0 
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.game_loop)
        
        self.car_item = CarItem()
        self.target_items = []
        self.sensor_items = []
        for _ in range(7):
            si = SensorItem()
            self.scene.addItem(si)
            self.sensor_items.append(si)

    def log(self, msg):
        self.log_console.append(msg)
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def setup_map(self, path):
        if not os.path.exists(path):
            self.create_dummy_map(path)
        self.map_img = QImage(path).convertToFormat(QImage.Format.Format_RGB32)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(self.map_img))
        self.brain = CarBrain(self.map_img)
        self.log(f"Map Loaded (V2 with Safety Override).")

    def create_dummy_map(self, path):
        img = QImage(1000, 800, QImage.Format.Format_RGB32)
        img.fill(C_BG_DARK)
        p = QPainter(img)
        p.setBrush(Qt.GlobalColor.white)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(100, 100, 800, 600)
        p.setBrush(C_BG_DARK)
        p.drawEllipse(250, 250, 500, 300)
        p.end()
        img.save(path)

    def load_map_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Map", "", "Images (*.png *.jpg)")
        if f: 
            self.full_reset()
            self.setup_map(f)

    def on_scene_click(self, event):
        pt = self.view.mapToScene(event.pos())
        if self.setup_state == 0:
            self.brain.set_start_pos(pt) 
            self.scene.addItem(self.car_item)
            self.car_item.setPos(pt)
            self.setup_state = 1
            self.lbl_status.setText("Click Map -> TARGET(S)\nRight-click when done")
        elif self.setup_state == 1:
            if event.button() == Qt.MouseButton.LeftButton:
                self.brain.add_target(pt)
                target_idx = len(self.brain.targets) - 1
                color = TARGET_COLORS[target_idx % len(TARGET_COLORS)]
                is_active = (target_idx == 0)
                num_targets = len(self.brain.targets)
                
                target_item = TargetItem(color, is_active, num_targets)
                target_item.setPos(pt)
                self.scene.addItem(target_item)
                self.target_items.append(target_item)
                
                self.lbl_status.setText(f"Targets: {num_targets}\nRight-click to finish setup")
                self.log(f"Target #{num_targets} added at ({pt.x():.0f}, {pt.y():.0f})")
            
            elif event.button() == Qt.MouseButton.RightButton:
                if len(self.brain.targets) > 0:
                    self.setup_state = 2
                    self.lbl_status.setText(f"READY. {len(self.brain.targets)} target(s). Press SPACE.")
                    self.lbl_status.setStyleSheet(f"background-color: {C_SUCCESS.name()}; color: #2E3440; font-weight: bold; padding: 10px; border-radius: 5px;")
                    self.btn_run.setEnabled(True)
                    self.update_visuals()

    def full_reset(self):
        self.sim_timer.stop()
        self.btn_run.setChecked(False)
        self.btn_run.setEnabled(False)
        self.setup_state = 0
        self.scene.removeItem(self.car_item)
        for target_item in self.target_items:
            self.scene.removeItem(target_item)
        self.target_items = []
        self.brain.targets = []
        self.brain.current_target_idx = 0
        self.brain.targets_reached = 0
        
        for s in self.sensor_items: 
            if s.scene() == self.scene: self.scene.removeItem(s)
        self.lbl_status.setText("1. Click Map -> CAR\n2. Click Map -> TARGET(S)")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; color: white; padding: 10px; border-radius: 5px;")
        self.log("--- RESET ---")
        self.chart.scores = []
        self.chart.update()

    def toggle_training(self):
        if self.btn_run.isChecked():
            self.sim_timer.start(16)
            self.btn_run.setText("⏸ PAUSE")
        else:
            self.sim_timer.stop()
            self.btn_run.setText("▶ RESUME")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.setup_state == 2:
            self.btn_run.click()

    def game_loop(self):
        if self.setup_state != 2: return

        state, _ = self.brain.get_state()
        action = 0
        
        prev_target_idx = self.brain.current_target_idx
        
        if random.random() < self.brain.epsilon:
            action = random.randint(0, 4)
        else:
            with torch.no_grad():
                q = self.brain.policy_net(torch.FloatTensor(state).unsqueeze(0))
                action = q.argmax().item()

        next_s, rew, done = self.brain.step(action)
        
        self.brain.store_experience((state, action, rew, next_s, done))
        self.brain.optimize()
        
        if self.brain.current_target_idx != prev_target_idx:
            target_num = self.brain.current_target_idx + 1
            total = len(self.brain.targets)
            self.log(f"<font color='#88C0D0'>🎯 Target {prev_target_idx + 1} reached! Moving to target {target_num}/{total}</font>")
        
        for target_param, policy_param in zip(self.brain.target_net.parameters(), 
                                               self.brain.policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
        
        self.brain.steps += 1
        
        if done:
            self.brain.finalize_episode(self.brain.score)
            
            should_reset_position = False
            
            if self.brain.consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                self.log(f"<font color='#BF616A'><b>⚠️ {MAX_CONSECUTIVE_CRASHES} consecutive crashes! Resetting to origin...</b></font>")
                self.log(f"<font color='#88C0D0'>💡 Tip: Safety override active - crashes should be rare!</font>")
                self.brain.consecutive_crashes = 0
                should_reset_position = True
            
            if not self.brain.alive:
                txt = f"CRASH ({self.brain.consecutive_crashes}/{MAX_CONSECUTIVE_CRASHES})"
                col = "#BF616A"
            else:
                if self.brain.targets_reached == len(self.brain.targets) - 1:
                    txt = f"ALL {len(self.brain.targets)} TARGETS COMPLETED!"
                    col = "#A3BE8C"
                else:
                    txt = "GOAL"
                    col = "#A3BE8C"
                should_reset_position = True
            
            priority_size = len(self.brain.priority_memory)
            regular_size = len(self.brain.memory)
            total_mem = priority_size + regular_size
            priority_pct = (priority_size / total_mem * 100) if total_mem > 0 else 0
            
            success_rate = priority_size / max(total_mem, 1)
            sampling_ratio = 0.3 + (success_rate * 0.4)
            
            self.log(f"<font color='{col}'>{txt} (Scr: {self.brain.score:.0f}) | "
                    f"Mem: {priority_size}P/{regular_size}R ({priority_pct:.1f}%) | "
                    f"Sample: {sampling_ratio*100:.0f}%P</font>")
            
            self.chart.update_chart(self.brain.score)
            
            if should_reset_position:
                self.brain.reset()
            else:
                self.brain.score = 0
                self.brain.alive = True
                self.brain.current_target_idx = 0
                self.brain.targets_reached = 0
                if len(self.brain.targets) > 0:
                    self.brain.target_pos = self.brain.targets[0]
                _, dist = self.brain.get_state()
                self.brain.prev_dist = dist

        self.update_visuals()
        self.val_eps.setText(f"{self.brain.epsilon:.3f}")
        self.val_rew.setText(f"{self.brain.score:.0f}")

    def update_visuals(self):
        self.car_item.setPos(self.brain.car_pos)
        self.car_item.setRotation(self.brain.car_angle)
        
        for i, target_item in enumerate(self.target_items):
            is_active = (i == self.brain.current_target_idx)
            target_item.set_active(is_active)
        
        self.scene.update() 
        
        for i, coord in enumerate(self.brain.sensor_coords):
            self.sensor_items[i].setPos(coord)
            s_val = self.brain.get_state()[0][i]
            self.sensor_items[i].set_detecting(s_val > 0.5)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NeuralNavApp()
    win.show()
    sys.exit(app.exec())

