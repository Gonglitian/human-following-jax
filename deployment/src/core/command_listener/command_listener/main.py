#!/usr/bin/env python3

import sys
import threading
import rclpy
from rclpy.node import Node

from std_msgs.msg import String

import tty
import termios
import select

class CommandListenerNode(Node):
    def __init__(self):
        super().__init__('command_listener')
        
        # Publisher for sending commands. Adjust topic name / QoS as needed.
        self.command_publisher_ = self.create_publisher(String, '/command', 10)
        
        self.running_ = True

        # We'll use a separate thread to handle user input, so the node can still be spun if needed.
        self.input_thread_ = threading.Thread(target=self.user_input_loop, daemon=True)
        self.input_thread_.start()

    def user_input_loop(self):
        """
        This loop continuously handles user prompts for mode selection and processes the selected mode.
        """
        while self.running_:
            # 1. Ask for the mode
            self.current_mode_ = None
            while self.current_mode_ not in [1, 2, 3] and self.running_:
                print("\nWhich mode do you want to choose? (1) Manual (2) Automatic Human Following (3) Combined mode")
                mode_str = self.blocking_input("Enter 1, 2, or 3: ")
                try:
                    mode = int(mode_str)
                    if mode in [1, 2, 3]:
                        self.current_mode_ = mode
                        self.publish_mode_command(mode)
                    else:
                        print("Invalid mode. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter 1, 2, or 3.")
            
            if not self.running_:
                break

            # 2. Handle the selected mode
            if self.current_mode_ == 1:
                self.listen_manual_input()
            elif self.current_mode_ == 2:
                self.handle_automatic_mode()
            elif self.current_mode_ == 3:
                self.handle_combined_mode()

    def publish_mode_command(self, mode: int):
        """Publishes the chosen mode to the /command topic."""
        msg = String()
        if mode == 1:
            msg.data = "mode:manual"
        elif mode == 2:
            msg.data = "mode:automatic"
        elif mode == 3:
            msg.data = "mode:combined"
        self.command_publisher_.publish(msg)
        self.get_logger().info(f"Published mode command: {msg.data}")

    def listen_manual_input(self):
        """Continuously listens to WASD inputs, echoes them, and publishes corresponding commands until exit."""
        print("\nListening to manual input .... (Press 'q' to quit back to mode selection)")
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(sys.stdin.fileno())
            while self.running_:
                if self.kbhit():
                    key = sys.stdin.read(1).lower()
                    
                    # Echo the key press
                    if key != '\x03':  # Ignore Ctrl+C
                        print(key, end='', flush=True)
                    
                    if key == 'q':
                        print("\nExiting manual input to mode selection.")
                        break  # Exit manual input

                    elif key in ['w', 'a', 's', 'd']:
                        msg = String()
                        msg.data = f"manual:{key}"
                        self.command_publisher_.publish(msg)
                        self.get_logger().debug(f"Published manual command: {msg.data}")
                    elif key == '\x03':
                        print("\nKeyboard interrupt received. Exiting.")
                        self.running_ = False
                        break
                    else:
                        print("\nInvalid key. Use W/A/S/D or Q to quit.")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def handle_automatic_mode(self):
        """
        Handle automatic human following mode - no additional configuration needed.
        Just start human following and wait for quit command.
        """
        print("\nAutomatic human following mode is active.")
        print("The robot will automatically follow detected humans.")
        print("Press 'q' to quit to mode selection.")
        
        # Publish human following command
        self.publish_human_following_command()
        
        # Listen for quit input
        self.listen_quit_input()

    def handle_combined_mode(self):
        """
        Handle combined mode - human following with manual override capability.
        """
        print("\nCombined mode is active.")
        print("The robot will follow humans by default.")
        print("Manual WASD commands will override the decider if pressed.")
        print("Press 'q' to quit to mode selection.")
        
        # Publish human following command
        self.publish_human_following_command()
        
        # Listen for manual input (including override)
        self.listen_manual_input_combined()

    def listen_manual_input_combined(self):
        """Listen for manual input in combined mode, allowing both WASD override and quit."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(sys.stdin.fileno())
            while self.running_:
                if self.kbhit():
                    key = sys.stdin.read(1).lower()
                    
                    # Echo the key press
                    if key != '\x03':  # Ignore Ctrl+C
                        print(key, end='', flush=True)
                    
                    if key == 'q':
                        # Publish stop command when exiting combined mode
                        self.publish_stop_command()
                        print("\nExiting combined mode to mode selection.")
                        break  # Exit combined mode

                    elif key in ['w', 'a', 's', 'd']:
                        msg = String()
                        msg.data = f"manual:{key}"
                        self.command_publisher_.publish(msg)
                        self.get_logger().debug(f"Published manual override command: {msg.data}")
                    elif key == '\x03':
                        print("\nKeyboard interrupt received. Exiting.")
                        self.running_ = False
                        break
                    else:
                        print("\nInvalid key. Use W/A/S/D for manual override or Q to quit.")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def publish_human_following_command(self):
        """Publishes command to start human following."""
        msg = String()
        msg.data = "auto:human_following"
        self.command_publisher_.publish(msg)
        self.get_logger().info(f"Published human following command: {msg.data}")

    def publish_stop_command(self):
        """Publishes a stop command to the /command topic."""
        msg = String()
        msg.data = "stop"
        self.command_publisher_.publish(msg)
        self.get_logger().info(f"Published stop command: {msg.data}")

    def listen_quit_input(self):
        """Listen for a key to quit current mode back to mode selection."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(sys.stdin.fileno())
            while self.running_:
                if self.kbhit():
                    key = sys.stdin.read(1).lower()
                    
                    if key == 'q':
                        # Return to mode selection and stop
                        print("\nExiting to mode selection.")
                        self.publish_stop_command()
                        break
                    else:
                        print("\nInvalid key. Press 'q'.")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def kbhit(self):
        """Check if a key has been pressed."""
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        return dr != []

    def blocking_input(self, prompt=""):
        """
        Displays a prompt and reads input without interfering with raw mode.
        This function is used only for initial mode selection and sub-mode selection,
        where line-buffered input is acceptable.
        """
        print(prompt, end='', flush=True)
        return sys.stdin.readline().strip()


def main(args=None):
    rclpy.init(args=args)
    node = CommandListenerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Shutting down.")
    finally:
        node.running_ = False
        if node.input_thread_.is_alive():
            node.input_thread_.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()