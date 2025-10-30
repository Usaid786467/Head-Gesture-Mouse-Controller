"""
Facial Expression OS Navigator - OS Control Module
Smooth, accelerated mouse control and system interaction
"""

import pyautogui
import numpy as np
from typing import Tuple, Optional
from collections import deque
import time
import platform

from config import mouse_config, action_config, safety_config


class OSController:
    """
    High-precision OS control with smooth mouse movement
    Implements acceleration curves and safety features
    """
    
    def __init__(self):
        """Initialize OS controller with safety settings"""
        # Disable PyAutoGUI failsafe (we'll implement our own)
        pyautogui.FAILSAFE = False
        
        # Set PyAutoGUI to be faster
        pyautogui.PAUSE = 0.0
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Mouse position smoothing buffer
        self.target_x: float = self.screen_width / 2
        self.target_y: float = self.screen_height / 2
        self.current_x: float = self.target_x
        self.current_y: float = self.target_y
        
        # Mouse movement history for acceleration
        self.movement_history: deque = deque(maxlen=10)
        
        # Action cooldowns
        self.last_click_time: float = 0
        self.last_right_click_time: float = 0
        self.last_scroll_time: float = 0
        self.last_menu_time: float = 0
        
        # Detect OS for platform-specific commands
        self.os_type = platform.system()  # 'Windows', 'Darwin' (macOS), 'Linux'
        
        # Mouse control state
        self.mouse_enabled = True
        self.scroll_accumulator = 0.0
        
        print(f"[OSController] Initialized on {self.os_type}")
        print(f"[OSController] Screen: {self.screen_width}x{self.screen_height}")
    
    def update_mouse_position(self, dx: float, dy: float):
        """
        Update mouse position with smooth acceleration
        
        Args:
            dx: Normalized delta X (-1 to 1)
            dy: Normalized delta Y (-1 to 1)
        """
        if not self.mouse_enabled:
            return
        
        # Apply deadzone
        if abs(dx) < 0.1:
            dx = 0
        if abs(dy) < 0.1:
            dy = 0
        
        # Apply acceleration curve
        dx = np.sign(dx) * (abs(dx) ** mouse_config.ACCELERATION_CURVE)
        dy = np.sign(dy) * (abs(dy) ** mouse_config.ACCELERATION_CURVE)
        
        # Scale by sensitivity
        dx *= mouse_config.SENSITIVITY * 10
        dy *= mouse_config.SENSITIVITY * 10
        
        # Update target position
        self.target_x += dx
        self.target_y += dy
        
        # Clamp to screen bounds with padding
        padding = mouse_config.SCREEN_PADDING
        self.target_x = np.clip(
            self.target_x, 
            padding, 
            self.screen_width - padding
        )
        self.target_y = np.clip(
            self.target_y, 
            padding, 
            self.screen_height - padding
        )
        
        # Smooth interpolation to target
        self.current_x += (self.target_x - self.current_x) * mouse_config.SMOOTHING_FACTOR
        self.current_y += (self.target_y - self.current_y) * mouse_config.SMOOTHING_FACTOR
        
        # Move mouse
        try:
            pyautogui.moveTo(int(self.current_x), int(self.current_y), duration=0)
        except Exception as e:
            print(f"[OSController] Mouse move error: {e}")
    
    def set_mouse_position(self, x: int, y: int):
        """
        Directly set mouse position (for absolute positioning)
        
        Args:
            x: Screen X coordinate
            y: Screen Y coordinate
        """
        if not self.mouse_enabled:
            return
        
        self.target_x = float(x)
        self.target_y = float(y)
        self.current_x = self.target_x
        self.current_y = self.target_y
        
        try:
            pyautogui.moveTo(x, y, duration=0)
        except Exception as e:
            print(f"[OSController] Set position error: {e}")
    
    def left_click(self) -> bool:
        """
        Perform left mouse click with cooldown
        
        Returns:
            True if click was performed, False if on cooldown
        """
        current_time = time.time() * 1000  # Convert to ms
        
        if current_time - self.last_click_time < action_config.CLICK_COOLDOWN_MS:
            return False
        
        try:
            pyautogui.click()
            self.last_click_time = current_time
            print(f"[OSController] Left click at ({int(self.current_x)}, {int(self.current_y)})")
            return True
        except Exception as e:
            print(f"[OSController] Click error: {e}")
            return False
    
    def right_click(self) -> bool:
        """
        Perform right mouse click with cooldown
        
        Returns:
            True if click was performed, False if on cooldown
        """
        current_time = time.time() * 1000
        
        if current_time - self.last_right_click_time < action_config.CLICK_COOLDOWN_MS:
            return False
        
        try:
            pyautogui.rightClick()
            self.last_right_click_time = current_time
            print(f"[OSController] Right click at ({int(self.current_x)}, {int(self.current_y)})")
            return True
        except Exception as e:
            print(f"[OSController] Right click error: {e}")
            return False
    
    def double_click(self) -> bool:
        """Perform double click"""
        try:
            pyautogui.doubleClick()
            self.last_click_time = time.time() * 1000
            print("[OSController] Double click")
            return True
        except Exception as e:
            print(f"[OSController] Double click error: {e}")
            return False
    
    def scroll(self, amount: float, direction: str = 'vertical') -> bool:
        """
        Perform scroll with acceleration
        
        Args:
            amount: Scroll amount (-1 to 1, negative = down/left)
            direction: 'vertical' or 'horizontal'
            
        Returns:
            True if scroll was performed
        """
        current_time = time.time() * 1000
        
        if current_time - self.last_scroll_time < action_config.SCROLL_COOLDOWN_MS:
            return False
        
        # Accumulate small movements
        self.scroll_accumulator += amount
        
        if abs(self.scroll_accumulator) < 0.3:
            return False
        
        # Calculate scroll clicks
        scroll_clicks = int(self.scroll_accumulator * action_config.SCROLL_AMOUNT)
        
        if scroll_clicks == 0:
            return False
        
        try:
            if direction == 'vertical':
                pyautogui.scroll(scroll_clicks)
            else:
                pyautogui.hscroll(scroll_clicks)
            
            self.scroll_accumulator = 0
            self.last_scroll_time = current_time
            print(f"[OSController] Scroll {direction}: {scroll_clicks}")
            return True
        except Exception as e:
            print(f"[OSController] Scroll error: {e}")
            return False
    
    def press_key(self, key: str) -> bool:
        """
        Press a keyboard key
        
        Args:
            key: Key name (e.g., 'space', 'enter', 'esc')
            
        Returns:
            True if successful
        """
        try:
            pyautogui.press(key)
            print(f"[OSController] Key press: {key}")
            return True
        except Exception as e:
            print(f"[OSController] Key press error: {e}")
            return False
    
    def hotkey(self, *keys) -> bool:
        """
        Press combination of keys
        
        Args:
            *keys: Keys to press together (e.g., 'ctrl', 'c')
            
        Returns:
            True if successful
        """
        try:
            pyautogui.hotkey(*keys)
            print(f"[OSController] Hotkey: {' + '.join(keys)}")
            return True
        except Exception as e:
            print(f"[OSController] Hotkey error: {e}")
            return False
    
    def open_start_menu(self) -> bool:
        """
        Open OS start menu / application launcher
        
        Returns:
            True if successful
        """
        current_time = time.time() * 1000
        
        if current_time - self.last_menu_time < action_config.MENU_COOLDOWN_MS:
            return False
        
        try:
            if self.os_type == 'Windows':
                pyautogui.press('win')
            elif self.os_type == 'Darwin':  # macOS
                pyautogui.hotkey('command', 'space')
            else:  # Linux
                pyautogui.press('super')
            
            self.last_menu_time = current_time
            print("[OSController] Start menu opened")
            return True
        except Exception as e:
            print(f"[OSController] Start menu error: {e}")
            return False
    
    def switch_application(self, forward: bool = True) -> bool:
        """
        Switch between applications (Alt+Tab / Cmd+Tab)
        
        Args:
            forward: True for next app, False for previous
            
        Returns:
            True if successful
        """
        try:
            if self.os_type == 'Darwin':  # macOS
                if forward:
                    pyautogui.hotkey('command', 'tab')
                else:
                    pyautogui.hotkey('command', 'shift', 'tab')
            else:  # Windows/Linux
                if forward:
                    pyautogui.hotkey('alt', 'tab')
                else:
                    pyautogui.hotkey('alt', 'shift', 'tab')
            
            print(f"[OSController] Switch app: {'forward' if forward else 'backward'}")
            return True
        except Exception as e:
            print(f"[OSController] Switch app error: {e}")
            return False
    
    def minimize_window(self) -> bool:
        """Minimize current window"""
        try:
            if self.os_type == 'Windows':
                pyautogui.hotkey('win', 'down')
            elif self.os_type == 'Darwin':
                pyautogui.hotkey('command', 'm')
            else:
                pyautogui.hotkey('super', 'h')
            
            print("[OSController] Window minimized")
            return True
        except Exception as e:
            print(f"[OSController] Minimize error: {e}")
            return False
    
    def maximize_window(self) -> bool:
        """Maximize current window"""
        try:
            if self.os_type == 'Windows':
                pyautogui.hotkey('win', 'up')
            elif self.os_type == 'Darwin':
                # macOS doesn't have a direct maximize hotkey
                pyautogui.hotkey('ctrl', 'command', 'f')
            else:
                pyautogui.hotkey('super', 'up')
            
            print("[OSController] Window maximized")
            return True
        except Exception as e:
            print(f"[OSController] Maximize error: {e}")
            return False
    
    def close_window(self) -> bool:
        """Close current window"""
        try:
            if self.os_type == 'Darwin':
                pyautogui.hotkey('command', 'w')
            else:
                pyautogui.hotkey('alt', 'f4')
            
            print("[OSController] Window closed")
            return True
        except Exception as e:
            print(f"[OSController] Close window error: {e}")
            return False
    
    def toggle_mouse_control(self, enabled: bool):
        """Enable or disable mouse control"""
        self.mouse_enabled = enabled
        print(f"[OSController] Mouse control: {'enabled' if enabled else 'disabled'}")
    
    def get_current_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        return (int(self.current_x), int(self.current_y))
    
    def emergency_stop(self):
        """Emergency stop all actions"""
        self.mouse_enabled = False
        print("[OSController] EMERGENCY STOP ACTIVATED")
    
    def reset(self):
        """Reset controller to initial state"""
        self.mouse_enabled = True
        self.target_x = self.screen_width / 2
        self.target_y = self.screen_height / 2
        self.current_x = self.target_x
        self.current_y = self.target_y
        self.scroll_accumulator = 0
        print("[OSController] Reset to default state")