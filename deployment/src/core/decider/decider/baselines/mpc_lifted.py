"""
MPC policy lifted verbatim from
  github.com/tasl-lab/human-following-robot @ baseline_mpc_orca
  crowd_nav/policy/mpc.py
on 2026-05-01. Only the imports were modified to make this file
self-contained inside the ROS 2 decider package — the algorithm,
cost function, IPOPT options, OGM handling, and predict() entry
point are unchanged from the source branch.

Used by both MPC-DC (paper §V.B guiding-policy baseline; static
preference_distance) and MPC-ADC (paper §V.B meta-policy baseline;
preference_distance updated at runtime via set_preference_distance()
when the controller receives `auto:distance:N`).
"""
import numpy as np
import csv
import math

import casadi as ca
from collections import namedtuple
import time
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, Polygon


# Lifted from crowd_sim/envs/utils/action.py (one-liner namedtuples).
ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
ActionRot = namedtuple('ActionRot', ['v', 'r'])


# Minimal stub of crowd_nav/policy/policy.Policy — original is purely
# abstract scaffolding. We only reproduce the fields the MPC class
# actually touches via super().__init__(config).
class Policy:
    def __init__(self, config):
        self.trainable = False
        self.phase = None
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None
        self.env = None
        self.config = config


class MPC(Policy):
    def __init__(self, config):
        """
        Model Predictive Control with preference distance following and static obstacle avoidance
        """
        super().__init__(config)
        self.name = 'mpc'

        # Get MPC config if available, otherwise use defaults
        mpc_config = getattr(config, 'mpc', None)

        self.horizon = getattr(mpc_config, 'horizon', 5) if mpc_config else 5
        self.time_step = config.env.time_step

        self.max_speed = getattr(mpc_config, 'max_speed', 1.0) if mpc_config else 1.0
        self.robot_radius = config.robot.radius

        # Weights for cost function
        self.w_goal = getattr(mpc_config, 'w_goal', 1.0) if mpc_config else 1.0
        self.w_control = getattr(mpc_config, 'w_control', 0.1) if mpc_config else 0.1
        self.w_human = getattr(mpc_config, 'w_human', 100.0) if mpc_config else 100.0
        self.w_obstacle = getattr(mpc_config, 'w_obstacle', 150.0) if mpc_config else 150.0
        self.w_preference = getattr(mpc_config, 'w_preference', 5.0) if mpc_config else 5.0

        # Safety margins
        self.safety_margin = getattr(mpc_config, 'safety_margin', 0.5) if mpc_config else 0.5
        self.obstacle_safety_margin = getattr(mpc_config, 'obstacle_safety_margin', 0.3) if mpc_config else 0.3

        # Preference distance for following (from config or default)
        self.preference_distance = getattr(config, 'preference_distance', 1.4)
        self.preference_tolerance = getattr(config, 'preference_tolerance', 0.2)

        # Timeout for MPC solver (must be less than time_step for real-time operation)
        self.solver_timeout = getattr(mpc_config, 'solver_timeout', 0.2) if mpc_config else 0.2

        self.last_input = None
        self.last_state = None
        self.mpc_success = None
        self.predicted_path = None
        self.last_robot_state = None

        # 静态障碍物列表 (will be set by environment)
        self.static_obstacles = None
        # OGM for obstacle avoidance (more efficient than obstacle_list)
        self.ogm = None

        # solver 选项
        self.solver_opts = {
            'print_time': False,
            'verbose': False,
            'expand': False,
            'error_on_fail': False
        }

        # IPOPT设置 (with timeout for real-time operation)
        self.ipopt_opts = {
            'ipopt.max_iter': 500,              # Reduced iterations for faster solving
            'ipopt.print_level': 0,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_obj_change_tol': 1e-3,
            'ipopt.max_cpu_time': self.solver_timeout,  # Timeout in seconds
            'ipopt.warm_start_init_point': 'yes'        # Use warm start for faster convergence
        }

        # 创建状态方程
        self._create_functions()
    
    def _create_functions(self):
        # 状态变量: [x, y, vx, vy]
        nx = 4
        # 控制变量: [ax, ay]
        nu = 2
        
        x = ca.SX.sym('x', nx)                # 状态向量 [x, y, vx, vy]
        u = ca.SX.sym('u', nu)                # 控制向量 [ax, ay]
        
        # 状态方程
        x_next = ca.vertcat(
            x[0] + self.time_step * x[2],     # x + vx * dt
            x[1] + self.time_step * x[3],     # y + vy * dt
            x[2] + self.time_step * u[0],     # vx + ax * dt
            x[3] + self.time_step * u[1]      # vy + ay * dt
        )
        
        # 使用函数封装状态转移
        self.f = ca.Function('f', [x, u], [x_next], ['x', 'u'], ['x_next'])


    def reset(self):
        self.last_input = None
        self.last_state = None
        self.mpc_success = None
        self.predicted_path = None
        self.last_robot_state = None

    def set_static_obstacles(self, obstacles):
        """Set static obstacles list from environment (Shapely Polygon objects)"""
        self.static_obstacles = obstacles

    def set_preference_distance(self, distance):
        """Set the preference distance for following"""
        self.preference_distance = distance

    def _get_obstacle_points_from_ogm(self, robot_pos, ogm, grid_size=10.0, resolution=0.2, max_points=20):
        """
        Extract obstacle points from OGM (cells marked as 1).
        Returns list of (x, y) world coordinates for occupied cells.

        Args:
            robot_pos: (px, py) robot position in world frame
            ogm: 2D numpy array (50x50), 1=occupied, 0=free
            grid_size: OGM coverage in meters (default 10m)
            resolution: meters per cell (default 0.2m)
            max_points: maximum number of obstacle points to return
        """
        if ogm is None:
            return []

        # Handle 3D OGM (history stack) - use latest frame
        if len(ogm.shape) == 3:
            ogm = ogm[-1]

        rx, ry = robot_pos[0], robot_pos[1]

        # Find occupied cells
        occupied = np.where(ogm == 1)
        if len(occupied[0]) == 0:
            return []

        # Convert grid indices to world coordinates
        # OGM is robot-centered: grid center (25, 25) = robot position
        grid_center = ogm.shape[0] // 2  # 25

        points = []
        for gy, gx in zip(occupied[0], occupied[1]):
            # Grid to world transformation
            wx = rx + (gx - grid_center) * resolution
            wy = ry + (gy - grid_center) * resolution
            points.append((wx, wy))

        # If too many points, sample uniformly
        if len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points = [points[i] for i in indices]

        return points


    def _get_human_trajectories(self, state):
        human_trajectories = []
        
        future_traj = state.human_future_traj  # 形状为 [6, human_num, 4]
        human_num = future_traj.shape[1]
        
        # 对每个人转换轨迹格式
        for h_idx in range(human_num):
            trajectory = []
            # 遍历时间步
            for t_idx in range(min(future_traj.shape[0], self.horizon + 1)):
                x, y = future_traj[t_idx, h_idx, 0:2]
                # 获取人的半径
                if h_idx < len(state.human_states):
                    radius = state.human_states[h_idx].radius
                else:
                    radius = 0.5  # 默认人类半径
                trajectory.append((x, y, radius))
            human_trajectories.append(trajectory)
            
        return human_trajectories
    
    def _solve_mpc(self, x0, x_ref, human_trajectories):
        start_time = time.time()

        # 检查初始化状态
        if self.last_input is None or self.last_state is None:
            # 初始化last_input和last_state
            self.last_input = np.zeros((self.horizon, 2))
            self.last_state = np.zeros((self.horizon+1, 4))
            for i in range(self.horizon+1):
                self.last_state[i] = x0
            self.mpc_success = False
        
        # 创建一个新的opti实例
        opti = ca.Opti()
        
        # 状态变量: [x, y, vx, vy]
        nx = 4
        # 控制变量: [ax, ay]
        nu = 2
        
        # 定义决策变量
        # 状态变量 (N+1 个时间步)
        opt_states = opti.variable(self.horizon + 1, nx)
        # 控制变量 (N 个时间步)
        opt_controls = opti.variable(self.horizon, nu)

        # Warm start: use previous solution shifted by one time step
        if self.mpc_success and self.last_state is not None and self.last_input is not None:
            # Shift previous state trajectory
            init_states = np.zeros((self.horizon + 1, nx))
            init_states[:-1] = self.last_state[1:]  # Shift left
            init_states[-1] = self.last_state[-1]   # Repeat last state
            init_states[0] = x0  # Current state

            # Shift previous control trajectory
            init_controls = np.zeros((self.horizon, nu))
            init_controls[:-1] = self.last_input[1:]  # Shift left
            init_controls[-1] = self.last_input[-1]   # Repeat last control

            opti.set_initial(opt_states, init_states)
            opti.set_initial(opt_controls, init_controls)
        else:
            # Initialize with simple trajectory toward goal
            opti.set_initial(opt_states, np.tile(x0, (self.horizon + 1, 1)))
            opti.set_initial(opt_controls, np.zeros((self.horizon, nu)))

        # 参考状态
        goal_ref = np.array([x_ref[0], x_ref[1], 0, 0])
        
        # 设置初始状态参数
        opt_x0 = opti.parameter(nx)
        opti.set_value(opt_x0, x0)
        
        # 添加初始状态约束 - 使用逐元素约束
        for j in range(nx):
            opti.subject_to(opt_states[0, j] == opt_x0[j])
        
        # 动力学约束（简化版，减少计算复杂度）
        dt = self.time_step
        for i in range(self.horizon):
            opti.subject_to(opt_states[i+1, 0] == opt_states[i, 0] + dt * opt_states[i, 2])  # x + vx*dt
            opti.subject_to(opt_states[i+1, 1] == opt_states[i, 1] + dt * opt_states[i, 3])  # y + vy*dt
            opti.subject_to(opt_states[i+1, 2] == opt_states[i, 2] + dt * opt_controls[i, 0])  # vx + ax*dt
            opti.subject_to(opt_states[i+1, 3] == opt_states[i, 3] + dt * opt_controls[i, 1])  # vy + ay*dt
        
        # 速度约束（平方形式避免非线性）
        for i in range(self.horizon + 1):
            opti.subject_to(opt_states[i, 2]**2 + opt_states[i, 3]**2 <= self.max_speed**2)
        
        # 加速度约束（平方形式）
        for i in range(self.horizon):
            opti.subject_to(opt_controls[i, 0]**2 + opt_controls[i, 1]**2 <= 2.0**2)
        
        # 构建目标函数
        obj = 0

        # ============ 1. Preference Distance Cost (Soft Constraint) ============
        for i in range(self.horizon + 1):
            dist_to_target = ca.sqrt((opt_states[i, 0] - goal_ref[0])**2 +
                                      (opt_states[i, 1] - goal_ref[1])**2 + 1e-6)
            distance_error = dist_to_target - self.preference_distance
            obj += self.w_preference * distance_error**2

        # ============ 2. Control Input Smoothness Cost ============
        for i in range(self.horizon):
            obj += self.w_control * (opt_controls[i, 0]**2 + opt_controls[i, 1]**2)

        # ============ 3. Human Avoidance Cost (Dynamic Obstacles) ============
        for i in range(self.horizon):
            for h_idx, human_traj in enumerate(human_trajectories):
                max_steps_to_consider = 6
                for t_idx in range(min(max_steps_to_consider, len(human_traj))):
                    h_x, h_y, h_radius = human_traj[t_idx]
                    dist_sq = (opt_states[i, 0] - h_x)**2 + (opt_states[i, 1] - h_y)**2
                    safe_dist = self.robot_radius + h_radius + self.safety_margin
                    obj += self.w_human * ca.fmax(0, safe_dist**2 - dist_sq)**2

        # ============ 4. Static Obstacle Avoidance Cost (using OGM) ============
        if self.ogm is not None:
            # Extract obstacle points from OGM (much faster than iterating obstacle_list)
            obstacle_points = self._get_obstacle_points_from_ogm(x0[:2], self.ogm, max_points=100)
            for (obs_x, obs_y) in obstacle_points:
                for i in range(self.horizon + 1):
                    dist_sq = (opt_states[i, 0] - obs_x)**2 + (opt_states[i, 1] - obs_y)**2
                    safe_dist = self.robot_radius + self.obstacle_safety_margin
                    obj += self.w_obstacle * ca.fmax(0, safe_dist**2 - dist_sq)**2
        
        # # CBF
        # gamma = 0.5
        # alpha_slack = self.w_human / 1000  # slack penalty weight, adjust as needed
        # for i in range(self.horizon):
        #     # Use the predicted robot velocity from the state (since control is acceleration)
        #     robot_vx = opt_states[i, 2]
        #     robot_vy = opt_states[i, 3]
            
        #     for h_idx, human_traj in enumerate(human_trajectories):
        #         max_steps_to_consider = 3
        #         for t_idx in range(min(max_steps_to_consider, len(human_traj))):
        #             h_x, h_y, h_radius = human_traj[t_idx]
                    
        #             # Calculate safe distance using original logic
        #             safe_dist = self.robot_radius + h_radius
        #             if t_idx == 0:
        #                 safe_dist += self.safety_margin
        #             if 0 < t_idx <= 2:
        #                 if not math.isnan(human_uncertainty[h_idx][t_idx]):
        #                     safe_dist += human_uncertainty[h_idx][t_idx]
                    
        #             # Calculate relative position
        #             dx = opt_states[i, 0] - h_x
        #             dy = opt_states[i, 1] - h_y
        #             dist_sq = dx**2 + dy**2
                    
        #             # CBF function: C(x) = dist_sq - safe_dist^2, capped at 2.0 using CasADi's fmin
        #             C = ca.fmin(dist_sq - safe_dist**2, 2.0)
                    
        #             # Estimate human velocity via finite difference
        #             if t_idx < len(human_traj) - 1:
        #                 h_vx = (human_traj[t_idx+1][0] - h_x) / dt
        #                 h_vy = (human_traj[t_idx+1][1] - h_y) / dt
        #             else:
        #                 h_vx, h_vy = 0.0, 0.0
                    
        #             # Compute derivative of C: dC/dt
        #             dot_C = 2 * dx * (robot_vx - h_vx) + 2 * dy * (robot_vy - h_vy)
                    
        #             # Form the CBF condition: dot_C + gamma * C >= 0
        #             # Introduce a slack variable s to soften the constraint
        #             s = opti.variable()
        #             opti.subject_to(s >= 0)
        #             opti.subject_to(dot_C + gamma * C + s >= 0)
                    
        #             # Add penalty term for the slack variable in the objective
        #             obj += alpha_slack * s
        
        
        opti.minimize(obj)
        
        # 使用IPOPT求解
        ipopt_options = {**self.solver_opts, **self.ipopt_opts}
        opti.solver('ipopt', ipopt_options)
        
        try:
            # 求解优化问题
            sol = opti.solve()
            
            # 获取结果
            state_res = sol.value(opt_states)
            u_res = sol.value(opt_controls)
            
            self.last_input = u_res
            self.last_state = state_res  # 这里存储的是数值数组，不是JointState对象
            self.mpc_success = True
            
            # 使用下一个时间步的速度
            vx_optimal = state_res[1, 2]
            vy_optimal = state_res[1, 3]
            
            # 存储预测轨迹用于可视化 - 直接存储所有点，在绘制时连成线
            # 将点按时间先后顺序排列，以确保线段连续
            self.predicted_path = [(state_res[i, 0], state_res[i, 1]) for i in range(self.horizon + 1)]
            
            # print("成功使用MPC求解")
            
            return vx_optimal, vy_optimal
            
        except Exception as e:
            print(f"MPC求解失败: {e}")
            self.predicted_path = None
            self.mpc_success = False
            # 确保last_state保留为数值数组或None，而不是JointState对象
            # self.last_state保持不变，继续使用上一次成功的解
            return self._backup_control(x0, x_ref)

    
    def _backup_control(self, x0, x_ref):
        """Backup control strategy when MPC solver fails or times out.
        Uses proportional control to maintain preference_distance from target."""
        try:
            px, py, vx, vy = x0
            gx, gy = x_ref

            # Calculate direction to target
            dx = gx - px
            dy = gy - py
            dist = np.sqrt(dx**2 + dy**2)

            # Normalize direction vector
            if dist > 1e-6:
                dx = dx / dist
                dy = dy / dist
            else:
                return 0.0, 0.0

            # Calculate distance error from preference distance
            distance_error = dist - self.preference_distance

            # Proportional control: move toward/away from target based on distance error
            # If too far (error > 0): move toward target
            # If too close (error < 0): move away from target
            speed = 0.5 * self.max_speed

            # Scale speed based on distance error
            if abs(distance_error) < 0.5:
                # Close to preference distance - move slowly
                speed = speed * abs(distance_error)
            elif distance_error < 0:
                # Too close - reverse direction
                dx = -dx
                dy = -dy

            # Calculate velocity components
            vx_cmd = dx * speed
            vy_cmd = dy * speed

            print(f"Using backup control (dist={dist:.2f}, pref={self.preference_distance:.2f})")
            self.predicted_path = None
            return vx_cmd, vy_cmd
        except Exception as e:
            print(f"Backup control error: {e}")
            self.predicted_path = None
            return 0.0, 0.0


    def predict(self, state):
        self_state = state.self_state

        if hasattr(self_state, 'sensor_range'):
            sensor_range = self_state.sensor_range
        else:
            sensor_range = 5.0

        # Get OGM from state if available (preferred over static_obstacles)
        if hasattr(state, 'ogm') and state.ogm is not None:
            self.ogm = state.ogm
        elif hasattr(state, 'static_obstacles') and state.static_obstacles is not None:
            self.static_obstacles = state.static_obstacles

        # Get preference distance from state if available
        if hasattr(state, 'preference_distance') and state.preference_distance is not None:
            self.preference_distance = state.preference_distance

        x0 = np.array([self_state.px, self_state.py, self_state.vx, self_state.vy])
        self.last_robot_state = x0

        x_ref = np.array([self_state.gx, self_state.gy])

        human_trajectories = self._get_human_trajectories(state)

        start = time.time()
        vx_optimal, vy_optimal = self._solve_mpc(x0, x_ref, human_trajectories)
        elapsed_time = time.time() - start
        print(f"mpc time per step: {elapsed_time:.4f}s")

        speed = np.sqrt(vx_optimal**2 + vy_optimal**2)
        if speed > self.max_speed:
            vx_optimal = vx_optimal / speed * self.max_speed
            vy_optimal = vy_optimal / speed * self.max_speed

        action = ActionXY(vx=vx_optimal, vy=vy_optimal)

        return action