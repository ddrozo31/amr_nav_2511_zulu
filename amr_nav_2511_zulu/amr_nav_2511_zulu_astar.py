import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped


import numpy as np
import heapq
import cv2

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def a_star(grid, start, goal):
    h, w = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w and grid[neighbor] == 0:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    priority = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, tentative_g_score, neighbor))
                    came_from[neighbor] = current
    return None  # No path

def world_to_grid(x, y, map_info):
    resolution = map_info.resolution
    origin_x = map_info.origin.position.x
    origin_y = map_info.origin.position.y
    grid_x = int((x - origin_x) / resolution)
    grid_y = int((y - origin_y) / resolution)
    return (grid_y, grid_x)  # row, col

def grid_to_world(i, j, map_info):
    resolution = map_info.resolution
    origin_x = map_info.origin.position.x
    origin_y = map_info.origin.position.y
    x = j * resolution + origin_x + resolution / 2
    y = i * resolution + origin_y + resolution / 2
    return (x, y)

def inflate_obstacles(binary_map, inflation_radius_m, map_info):
    resolution = map_info.resolution
    inflation_radius_px = int(np.ceil(inflation_radius_m / resolution))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * inflation_radius_px + 1, 2 * inflation_radius_px + 1))
    
    inflated = cv2.dilate(binary_map.astype(np.uint8), kernel)
    return inflated


class AStarPlanner(Node):

    def __init__(self):
        super().__init__('astar_planner')

        # Define QoS with transient_local so we receive latched map data
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.map_received = False
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos_profile
        )
        
        self.sub_goalpose = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        
        self.sub_goalpose = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amclpose_callback,
            qos_profile
        )
        
        self.path_publisher = self.create_publisher(Path, '/plan', 10)

        self.map = None
        self.binary_map = None
        self.start_world = 0.0
        self.goal_world = 0.0

        self.get_logger().info('A* Planner Node has been started!')
        
    def amclpose_callback(self, msg):
        self.get_logger().info('AMCL pose received!')
        # Here you can implement logic to handle the AMCL pose if needed
        # For example, you might want to update the start position for A* planning
        self.start_world = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        if self.map_received:
            print("Start position:", self.start_world)
        else:
            self.get_logger().warn('Map not received yet, cannot set start position!')

    def goal_callback(self, msg):
        self.get_logger().info('Goal pose received!')
        # Here you can implement the logic to plan a path using A* algorithm
        # based on the received map and goal pose.
        self.goal_world = [msg.pose.position.x, msg.pose.position.y]
        if self.map_received:
            print("Planning path to goal:",self.goal_world)
            
            # 
            self.plan_path()
        else:
            self.get_logger().warn('Map not received yet, cannot plan path!')
        

    def map_callback(self, msg):
        self.get_logger().info('Map received!')
        self.map_received = True
        # You can store the map and trigger planning from here
        self.map = msg
        
        
    def occupancy_grid_to_numpy(self,occupancy_grid_msg):
        width = occupancy_grid_msg.info.width
        height = occupancy_grid_msg.info.height
        data = np.array(occupancy_grid_msg.data).reshape((height, width))
        # Thresholding: -1=unknown, 0=free, 100=occupied
        obstacle_threshold = 50
        binary_map = np.where(data > obstacle_threshold, 1, 0)
        return binary_map
    
    def plan_path(self):
        
        binary_map  = self.occupancy_grid_to_numpy(self.map)
        print("Binary map shape:", self.binary_map)
        
        robot_radius = 0.35  # meters (adjust based on your robot's size + safety margin)
        inflated_map = inflate_obstacles(binary_map, robot_radius, self.map.info)
        
        start = world_to_grid(*self.start_world, self.map.info)
        goal = world_to_grid(*self.goal_world, self.map.info)
        
        path = a_star(inflated_map, start, goal)
        if path is None:
            self.get_logger().warn('No path found')
            return
        
        self.get_logger().info('Path found!')
        
        ros_path = Path()
        ros_path.header.frame_id = 'map'
        ros_path.header.stamp = self.get_clock().now().to_msg()

        for i, j in path:
            x, y = grid_to_world(i, j, self.map.info)
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            ros_path.poses.append(pose)

        self.path_publisher.publish(ros_path)
        self.get_logger().info(f"Path published with {len(ros_path.poses)} poses.")
        

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()

    # Wait until the map is received before spinning
    while not node.map_received and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

    # Do something with the map, like initialize planning...

    # Keep the node alive for future callbacks (if needed)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
# This code is a simple A* planner node that subscribes to the /map topic