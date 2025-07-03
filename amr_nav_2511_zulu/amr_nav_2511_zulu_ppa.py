import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


from visualization_msgs.msg import Marker

from geometry_msgs.msg import PoseWithCovarianceStamped

import math

def euler_from_quaternion(quat):
    """Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)."""
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        self.path = []
        self.lookahead_distance = 0.5  # meters
        self.linear_speed = 0.25  # m/s
        self.robot_pose = None
        
        # Define QoS with transient_local so we receive latched map data
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(Path, '/plan', self.path_callback, 10)
        
         # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        
        self.lookahead_pub = self.create_publisher(Marker, '/lookahead_point', 10)

        self.create_timer(0.05, self.control_loop)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        
        self.get_logger().info("Pure Pursuit Node initialized.")

   
    def get_robot_pose(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                now,
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return x, y, yaw
        except (LookupException, ConnectivityException, ExtrapolationException):
            self.get_logger().warn("TF lookup failed.")
            return None
   
   
        
    def path_callback(self, msg):
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.get_logger().info(f"Received path with {len(self.path)} points.")
        

    def find_lookahead_point(self, x, y):
        if not self.path:
            return None

        for px, py in self.path:
            distance = math.hypot(px - x, py - y)
            if distance >= self.lookahead_distance:
                return (px, py)

        # Fallback: use last point (goal)
        return self.path[-1]

    
    def is_goal_reached(self, x, y, tolerance=0.1):
        if not self.path:
            return True
        goal_x, goal_y = self.path[-1]
        return math.hypot(goal_x - x, goal_y - y) < tolerance
    
    def publish_lookahead_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.a = 1.0
        self.lookahead_pub.publish(marker)
    
    def control_loop(self):
        if not self.path:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        x, y, yaw = pose

        if self.is_goal_reached(x, y):
            self.get_logger().info_once("Goal reached.")
            self.cmd_pub.publish(Twist())  # Stop the robot
            self.path = []
            return

        lookahead = self.find_lookahead_point(x, y)
        if lookahead is None:
            self.get_logger().info_once("No valid lookahead point.")
            self.cmd_pub.publish(Twist())  # Stop
            return

        lx, ly = lookahead

        # Transform lookahead point to robot's local frame
        dx = lx - x
        dy = ly - y

        # Convert lookahead to robot frame
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy

        if local_x <= 0:
            curvature = 0.0
        else:
            curvature = 2 * local_y / (self.lookahead_distance ** 2)
            
        if local_x <= 0.01:
            self.get_logger().info("Lookahead behind robot. Crawling forward.")
            twist = Twist()
            twist.linear.x = 0.05  # creep forward
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return
        


        # Compute control commands
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = self.linear_speed * curvature

        
        dist_to_goal = math.hypot(self.path[-1][0] - x, self.path[-1][1] - y)
        if dist_to_goal < 0.3:
            twist.linear.x = min(self.linear_speed, 0.1)
        
        self.cmd_pub.publish(twist)
        
        self.publish_lookahead_marker(lx, ly)
        
        self.get_logger().info(f"Lookahead world: ({lx:.2f}, {ly:.2f})")
        self.get_logger().info(f"Robot pose: ({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f} deg)")
        self.get_logger().info(f"Lookahead in robot frame: x={local_x:.2f}, y={local_y:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()