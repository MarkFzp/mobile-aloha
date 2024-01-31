import math

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class DifferentialDrive:
    def __init__(self, wheel_distance):
        self.wheel_distance = wheel_distance
    def compute_wheel_speeds(self, v, omega):
        v_left = v - self.wheel_distance / 2 * omega
        v_right = v + self.wheel_distance / 2 * omega
        return v_left, v_right

class RobotController:
    def __init__(self, kp, ki, kd, wheel_distance, v_max, position_threshold, yaw_threshold):
        self.pid = PIDController(kp, ki, kd)
        self.drive = DifferentialDrive(wheel_distance)
        self.v_max = v_max
        self.position_threshold = position_threshold
        self.yaw_threshold = yaw_threshold
    
    def control(self, x_robot, y_robot, theta_current, waypoints, target_yaws):
        # Check if we reached the current waypoint and its orientation
        x_target, y_target = waypoints[0]
        theta_target = target_yaws[0]
        distance_to_target = math.sqrt((x_target - x_robot)**2 + (y_target - y_robot)**2)
        if distance_to_target < self.position_threshold and abs(self.normalize_angle(theta_current - theta_target)) < self.yaw_threshold:
            waypoints.pop(0)  # Remove the reached waypoint
            target_yaws.pop(0)  # Remove the reached yaw target
            if not waypoints:  # All waypoints traversed
                return 0, 0  # Stop the robot
            x_target, y_target = waypoints[0]
            theta_target = target_yaws[0]
        # Calculate desired heading for position
        theta_desired = math.atan2(y_target - y_robot, x_target - x_robot)
        # Compute heading errors
        delta_theta_position = self.normalize_angle(theta_desired - theta_current)
        delta_theta_yaw = self.normalize_angle(theta_target - theta_current)
        # Blending between positional and yaw error based on distance to target
        blend_factor = min(1, distance_to_target / self.position_threshold)
        delta_theta = blend_factor * delta_theta_position + (1 - blend_factor) * delta_theta_yaw
        # Get angular speed from PID
        omega = self.pid.compute(delta_theta, 0.1)  # assuming dt = 0.1 for this example
        # Decide linear speed
        v = min(self.v_max, 0.1 * distance_to_target)  # proportionality constant of 0.1
        # Compute wheel speeds
        v_left, v_right = self.drive.compute_wheel_speeds(v, omega)
        return v_left, v_right
    
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

# Example usage:
waypoints = [(1, 1), (2, 2), (3, 1)]
target_yaws = [math.pi/4, 0, -math.pi/4]
controller = RobotController(kp=1.0, ki=0.1, kd=0.05, wheel_distance=0.5, v_max=1.0, position_threshold=0.1, yaw_threshold=0.1)
# Simulating the robot’s movement towards waypoints:
while waypoints:
    v_left, v_right = controller.control(x_robot=0, y_robot=0, theta_current=0, waypoints=waypoints, target_yaws=target_yaws)
    print(f'v_left: {v_left}, v_right: {v_right}')
    # Here, update the robot’s position (x_robot, y_robot) and orientation (theta_current) based on v_left and v_right.