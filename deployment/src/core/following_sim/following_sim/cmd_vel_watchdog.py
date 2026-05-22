"""
If /cmd_vel goes silent for too long (decider crash, RL inference hang,
DDS drop), publishes a zero Twist so the robot doesn't run away on the
last command. Only kicks in after the decider has been seen alive once,
so it doesn't fight the launch sequence before arming.

When the watchdog fires it logs at WARN every time it overrides, plus
publishes a heartbeat to /cmd_vel_watchdog/state so external tools can
plot when the safety net engaged.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class CmdVelWatchdog(Node):
    def __init__(self):
        super().__init__('cmd_vel_watchdog')
        self.declare_parameter('timeout', 1.0)
        self.declare_parameter('check_period', 0.2)
        self.timeout = self.get_parameter('timeout').value
        self.period = self.get_parameter('check_period').value

        self.last_cmd_time = None
        self.was_overriding = False
        self.has_seen_cmd = False

        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel',
                                                self._cmd_cb, 10)
        # Same topic, different node — Gazebo planar_move accepts the latest msg
        # regardless of publisher; our zero Twist will just stack on top of any
        # stale decider command.
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(String,
                                               '/cmd_vel_watchdog/state', 10)

        self.create_timer(self.period, self._tick)
        self.get_logger().info(
            f'cmd_vel_watchdog up; will publish zero after '
            f'{self.timeout:.1f}s of silence (after first /cmd_vel seen)')

    def _cmd_cb(self, msg: Twist):
        # Ignore our own zero override so we don't keep ourselves "alive".
        if (msg.linear.x == 0.0 and msg.linear.y == 0.0
                and msg.angular.z == 0.0 and self.was_overriding):
            return
        self.last_cmd_time = self.get_clock().now()
        if not self.has_seen_cmd:
            self.has_seen_cmd = True
            self.get_logger().info('first /cmd_vel observed; watchdog armed')

    def _tick(self):
        if not self.has_seen_cmd or self.last_cmd_time is None:
            return
        elapsed = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        if elapsed > self.timeout:
            zero = Twist()
            self.cmd_pub.publish(zero)
            state = String()
            state.data = f'OVERRIDING (silent {elapsed:.2f}s)'
            self.state_pub.publish(state)
            if not self.was_overriding:
                self.get_logger().warn(
                    f'/cmd_vel silent for {elapsed:.1f}s — publishing zero')
                self.was_overriding = True
        else:
            if self.was_overriding:
                self.get_logger().info('/cmd_vel resumed; watchdog disengaged')
                self.was_overriding = False
            state = String()
            state.data = f'OK (last {elapsed*1000:.0f}ms ago)'
            self.state_pub.publish(state)


def main():
    rclpy.init()
    node = CmdVelWatchdog()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
