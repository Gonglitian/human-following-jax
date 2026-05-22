"""
Periodically asserts that the critical TF chain is intact. Bringup races
where odom -> base_footprint isn't yet published cause RViz to silently
draw nothing; this monitor flips the symptom from "I see no robot" to
"WARN: TF link odom->base_footprint missing for 3.0s".

Negligible overhead (one canTransform call per 1s).
"""
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros


REQUIRED_LINKS = [
    ('map', 'odom'),
    ('odom', 'base_footprint'),
    ('base_footprint', 'laser_frame'),
]


class TFHealthMonitor(Node):
    def __init__(self):
        super().__init__('tf_health_monitor')
        self.declare_parameter('check_period', 1.0)
        self.declare_parameter('warn_threshold', 3.0)
        self.period = self.get_parameter('check_period').value
        self.warn_threshold = self.get_parameter('warn_threshold').value

        self.buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.listener = tf2_ros.TransformListener(self.buffer, self)
        self.first_seen = {pair: None for pair in REQUIRED_LINKS}
        self.missing_since = {pair: None for pair in REQUIRED_LINKS}

        self.create_timer(self.period, self._tick)
        self.get_logger().info(
            f'tf_health_monitor up; checking {len(REQUIRED_LINKS)} links '
            f'every {self.period:.1f}s, warning after {self.warn_threshold:.1f}s')

    def _tick(self):
        now = self.get_clock().now()
        for src, dst in REQUIRED_LINKS:
            try:
                self.buffer.lookup_transform(src, dst, rclpy.time.Time())
                if self.first_seen[(src, dst)] is None:
                    self.first_seen[(src, dst)] = now.nanoseconds / 1e9
                    self.get_logger().info(f'TF {src}->{dst} OK')
                if self.missing_since[(src, dst)] is not None:
                    duration = now.nanoseconds / 1e9 - self.missing_since[(src, dst)]
                    self.get_logger().info(
                        f'TF {src}->{dst} recovered after {duration:.1f}s')
                    self.missing_since[(src, dst)] = None
            except Exception:
                if self.missing_since[(src, dst)] is None:
                    self.missing_since[(src, dst)] = now.nanoseconds / 1e9
                else:
                    elapsed = now.nanoseconds / 1e9 - self.missing_since[(src, dst)]
                    if elapsed > self.warn_threshold and (elapsed % 5.0) < self.period:
                        self.get_logger().warn(
                            f'TF {src}->{dst} missing for {elapsed:.1f}s')


def main():
    rclpy.init()
    node = TFHealthMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
