import matplotlib.pyplot as plt
import numpy as np
import math
from frenet_path import *
import cvxpy
# mpc parameters
NX = 3  # x = x, y, yaw
NU = 2  # u = [v,delta]
T = 20  # horizon length
R = np.diag([0.1, 0.1])  # input cost matrix
Rd = np.diag([0.1, 0.1])  # input difference cost matrix
Q = np.diag([1, 1, 1])  # state cost matrix
Qf = Q  # state final matrix


#车辆
dt = 0.2 # 时间间隔，单位：s
L = 2  # 车辆轴距，单位：m
v = 2  # 初始速度
x_0 = 0  # 初始x
y_0 = -3  # 初始y
psi_0 = 0  # 初始航向角

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(45.0)  # maximum steering speed [rad/s]

MAX_VEL = 2.0  # maximum accel [m/s]

def get_nparray_from_matrix(x):
    return np.array(x).flatten()
class semi_truck:
    def __init__(self, x, y, L1, L2, yaw1, yaw2, v):
        self.x = x
        self.y = y
        self.L_head = L1
        self.L_trailer = L2
        self.yaw_h = yaw1
        self.yaw_t = yaw2
        self.v = v
        self.a = 0
        self.dt = 0.2
        self.L = L1

        self.x_trailer = self.x - self.L_trailer * math.cos(self.yaw_t)
        self.y_trailer = self.y - self.L_trailer * math.sin(self.yaw_t)

    def update_state(self, a, delta_f):
        self.x = self.x + self.v * np.cos(self.yaw_h) * self.dt
        self.y = self.y + self.v * np.sin(self.yaw_h) * self.dt
        # theta_h2t = self.yaw_h - self.yaw_t
        # x_trailer_next = self.x_trailer + self.v * np.cos(self.yaw_t) * np.cos(theta_h2t) * self.dt
        # y_trailer_next = self.y_trailer + self.v * np.sin(self.yaw_t) * np.cos(theta_h2t) * self.dt

        yaw_h_next = self.yaw_h + self.v / self.L_head * math.tan(delta_f) * self.dt
        yaw_t_next = self.yaw_t + self.v / self.L_trailer * math.sin(self.yaw_h - self.yaw_t) * self.dt

        x_trailer_next = self.x - self.L_trailer * math.cos(self.yaw_t)
        y_trailer_next = self.y - self.L_trailer * math.sin(self.yaw_t)

        self.yaw_h = yaw_h_next
        self.yaw_t = yaw_t_next
        self.x_trailer = x_trailer_next
        self.y_trailer = y_trailer_next
        self.v = self.v + self.dt * a
        self.a = a

    def get_state(self):
        x_trailer = self.x - self.L_trailer*math.cos(self.yaw_t)
        y_trailer = self.y - self.L_trailer*math.sin(self.yaw_t)

        x_head_front = self.x + self.L_head * math.cos(self.yaw_h)
        y_head_front = self.y + self.L_head * math.sin(self.yaw_h)


        return self.x, self.y, self.yaw_h, self.v, self.a
        # return [self.x, self.y, x_head_front, y_head_front, self.x_trailer, self.y_trailer, self.yaw_h, self.yaw_t]

    def state_space(self, ref_delta, ref_yaw):
        """将模型离散化后的状态空间表达

        Args:
            ref_delta (_type_): 参考的转角控制量
            ref_yaw (_type_): 参考的偏航角

        Returns:
            _type_: _description_
        """

        A = np.matrix([
            [1.0, 0.0, -self.v * self.dt * math.sin(ref_yaw)],
            [0.0, 1.0, self.v * self.dt * math.cos(ref_yaw)],
            [0.0, 0.0, 1.0]])

        B = np.matrix([
            [self.dt * math.cos(ref_yaw), 0],
            [self.dt * math.sin(ref_yaw), 0],
            [self.dt * math.tan(ref_delta) / self.L,
             self.v * self.dt / (self.L * math.cos(ref_delta) * math.cos(ref_delta))]
        ])

        # C = np.eye(3)
        C = np.zeros(3)
        return A, B, C


class KinematicModel_3:
  """假设控制量为转向角delta_f和加速度a
  """

  def __init__(self, x, y, psi, v, L, dt):
    self.x = x
    self.y = y
    self.psi = psi
    self.v = v
    self.L = L
    self.a = 0
    # 实现是离散的模型
    self.dt = dt

  def update_state(self, a, delta_f):
    self.x = self.x+self.v*math.cos(self.psi)*self.dt
    self.y = self.y+self.v*math.sin(self.psi)*self.dt
    self.psi = self.psi+self.v/self.L*math.tan(delta_f)*self.dt
    self.v = self.v+a*self.dt
    self.a = a

  def get_state(self):
    return self.x, self.y, self.psi, self.v ,self.a

  def state_space(self, ref_delta, ref_yaw):
    """将模型离散化后的状态空间表达

    Args:
        ref_delta (_type_): 参考的转角控制量
        ref_yaw (_type_): 参考的偏航角

    Returns:
        _type_: _description_
    """

    A = np.matrix([
        [1.0, 0.0, -self.v*self.dt*math.sin(ref_yaw)],
        [0.0, 1.0, self.v*self.dt*math.cos(ref_yaw)],
        [0.0, 0.0, 1.0]])

    B = np.matrix([
        [self.dt*math.cos(ref_yaw), 0],
        [self.dt*math.sin(ref_yaw), 0],
        [self.dt*math.tan(ref_delta)/self.L, self.v*self.dt /(self.L*math.cos(ref_delta)*math.cos(ref_delta))]
    ])

    C = np.zeros(3)
    return A, B, C
def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    copied from https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/stanley_control/stanley_control.html
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle
class MyReferencePath:
    def __init__(self,path):
        self.refer_path = path

    def calc_track_error(self, x, y):
        """计算跟踪误差

        Args:
            x (_type_): 当前车辆的位置x
            y (_type_): 当前车辆的位置y

        Returns:
            _type_: _description_
        """
        # 寻找参考轨迹最近目标点
        d_x = [self.refer_path[i, 0]-x for i in range(len(self.refer_path))]
        d_y = [self.refer_path[i, 1]-y for i in range(len(self.refer_path))]
        d = [np.sqrt(d_x[i]**2+d_y[i]**2) for i in range(len(d_x))]
        s = np.argmin(d)  # 最近目标点索引

        yaw = self.refer_path[s, 2]  #航向角
        k = self.refer_path[s, 3]  #曲率
        angle = normalize_angle(yaw - math.atan2(d_y[s], d_x[s])) #航向角误差
        e = d[s]  # 误差  距离误差
        if angle < 0:
            e *= -1

        return e, k, yaw, s

    def calc_ref_trajectory(self, robot_state, dl=1.0):
        """计算参考轨迹点，统一化变量数组，便于后面MPC优化使用
            参考自https://github.com/AtsushiSakai/PythonRobotics/blob/eb6d1cbe6fc90c7be9210bf153b3a04f177cc138/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py
        Args:
            robot_state (_type_): 车辆的状态(x,y,yaw,v)
            dl (float, optional): _description_. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        e, k, ref_yaw, ind = self.calc_track_error(
            robot_state[0], robot_state[1])

        xref = np.zeros((NX, T + 1))
        dref = np.zeros((NU, T))
        ncourse = len(self.refer_path)

        xref[0, 0] = self.refer_path[ind, 0]
        xref[1, 0] = self.refer_path[ind, 1]
        xref[2, 0] = self.refer_path[ind, 2]

        # 参考控制量[v,delta]
        ref_delta = math.atan2(L*k, 1)
        dref[0, :] = robot_state[3]
        dref[1, :] = ref_delta

        travel = 0.0

        for i in range(T + 1):
            travel += abs(robot_state[3]) * dt
            dind = int(round(travel / dl))

            if (ind + dind) < ncourse:
                xref[0, i] = self.refer_path[ind + dind, 0]
                xref[1, i] = self.refer_path[ind + dind, 1]
                xref[2, i] = self.refer_path[ind + dind, 2]

            else:
                xref[0, i] = self.refer_path[ncourse - 1, 0]
                xref[1, i] = self.refer_path[ncourse - 1, 1]
                xref[2, i] = self.refer_path[ncourse - 1, 2]

        return xref, ind, dref
def linear_mpc_control(xref, x0, delta_ref, ugv):
    """
    linear mpc control

    xref: reference point
    x0: initial state
    delta_ref: reference steer angle
    ugv:车辆对象
    returns: 最优的控制量和最优状态
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0  # 代价函数
    constraints = []  # 约束条件
    '''
    预测时域内的cost
    '''
    for t in range(T):
        cost += cvxpy.quad_form(u[:, t]-delta_ref[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(x[:, t] - xref[:, t], Q)

        A, B, C = ugv.state_space(delta_ref[1, t], xref[2, t])
        constraints += [x[:, t + 1]-xref[:, t+1] == A @
                        (x[:, t]-xref[:, t]) + B @ (u[:, t]-delta_ref[:, t])]


    cost += cvxpy.quad_form(x[:, T] - xref[:, T], Qf)

    constraints += [(x[:, 0]) == x0]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_VEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        opt_x = get_nparray_from_matrix(x.value[0, :])
        opt_y = get_nparray_from_matrix(x.value[1, :])
        opt_yaw = get_nparray_from_matrix(x.value[2, :])
        opt_v = get_nparray_from_matrix(u.value[0, :])
        opt_delta = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        opt_v, opt_delta, opt_x, opt_y, opt_yaw = None, None, None, None, None,

    return opt_v, opt_delta, opt_x, opt_y, opt_yaw

def ComputeCurvature(dy, ddy):
    a = ddy
    norm_square = 1+dy*dy
    norm = np.sqrt(norm_square)
    b = norm*norm_square
    return a/b
def plot_vehicle(x, y, yaw,length,width):
    # 定义车辆尺寸
    # length = 4.5
    # width = 3.0

    # 计算车辆的四个角点
    corners = np.array([[-length / 2, -width / 2],
                       [length / 2, -width / 2],
                       [length / 2, width / 2],
                       [-length / 2, width / 2]])

    # 根据偏航角进行旋转变换
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                                [np.sin(yaw), np.cos(yaw)]])
    rotated_corners = np.dot(corners, rotation_matrix.T)

    # 平移车辆位置
    translated_corners = rotated_corners + np.array([x, y])

    # 绘制车辆
    plt.plot(translated_corners[:, 0], translated_corners[:, 1], 'b-')
    plt.plot(translated_corners[[0, 3], 0], translated_corners[[0, 3], 1], 'b-')
    plt.plot(translated_corners[[1, 2], 0], translated_corners[[1, 2], 1], 'b-')

def main():
    road, x_left, y_left, x_right, y_right, ob = define_road()


    car_state = np.array([0, 0, -0.3, 2, 0])
    # ugv = KinematicModel_3(car_state[0], car_state[1], car_state[2], car_state[3], L=2, dt=0.2)
    ugv = semi_truck(car_state[0],car_state[1],L1=1,L2=3,yaw1=car_state[2],yaw2=0,v=car_state[3])
    car_state = ugv.get_state()
    print(car_state)
    frenet_state = road.cartesian_to_frenet3D(car_state[0],car_state[1],car_state[2],car_state[3],car_state[4])
    current_speed = ugv.v
    current_acc = ugv.a
    current_s = frenet_state[0]
    current_d = frenet_state[1]
    current_d_d = frenet_state[3]
    current_d_dd = frenet_state[-1]
    best_path = get_best_path(current_speed,current_acc,current_d,current_d_d,current_d_dd,current_s,road,ob)
    path_c = calc_frenet_paths(current_speed, current_acc, current_d, current_d_d, current_d_dd, current_s)
    calc_global_paths(path_c, road)
    dy = np.gradient(best_path.y, best_path.x)  #
    ddy = np.gradient(dy, best_path.x)
    x_ ,y_ ,yaw_ ,acc_ ,v_ ,x_t,y_t= [] , [] ,[] ,[] , [],[],[]
    for t in range(5000):


        ref_path = np.array([best_path.x, best_path.y, best_path.yaw, best_path.c]).transpose()
        frenet_path = MyReferencePath(ref_path)
        robot_state = np.zeros(4)
        robot_state[0] = ugv.x
        robot_state[1] = ugv.y
        robot_state[2] = ugv.yaw_h
        robot_state[3] = ugv.v
        x0 = robot_state[0:3]

        xref, target_ind, dref = frenet_path.calc_ref_trajectory(
            robot_state)
        opt_v, opt_delta, opt_x, opt_y, opt_yaw = linear_mpc_control(
            xref, x0, dref, ugv)
        # ugv.update_state(0, opt_delta[0])  # 加速度设为0，恒速
        acc = opt_v[0] - ugv.v
        # acc_index = 1
        # acc = best_path
        ugv.update_state(acc, opt_delta[0])
        ugv.a = acc

        x_.append(ugv.x)
        y_.append(ugv.y)
        x_t.append(ugv.x_trailer)
        y_t.append(ugv.y_trailer)
        # yaw_.append(ugv.psi)
        yaw_.append(ugv.yaw_h)
        acc_.append(ugv.a)
        v_.append(ugv.v)

        car_state = [ugv.x, ugv.y, ugv.yaw_h, ugv.v, ugv.a]
        frenet_state = road.cartesian_to_frenet3D(ugv.x, ugv.y, ugv.yaw_h, ugv.v, ugv.a)

        if frenet_state[0] > 70:
            break

        if t % 10 == 0:
            current_speed = ugv.v
            current_acc = ugv.a
            current_s = frenet_state[0]
            current_d = frenet_state[1]
            current_d_d = frenet_state[3]
            current_d_dd = frenet_state[-1]
            path_c = calc_frenet_paths(current_speed,current_acc,current_d,current_d_d,current_d_dd,current_s)
            calc_global_paths(path_c, road)
            best_path = get_best_path(current_speed,current_acc,current_d,current_d_d,current_d_dd,current_s,road,ob)

        x_redraw = ugv.x + ugv.L_head / 2 * math.cos(ugv.yaw_h)
        y_redraw = ugv.y + ugv.L_head / 2 * math.sin(ugv.yaw_h)

        x_t_redraw = ugv.x_trailer + ugv.L_trailer / 2 * math.cos(ugv.yaw_t)
        y_t_redraw = ugv.y_trailer + ugv.L_trailer / 2 * math.sin(ugv.yaw_t)



        plt.cla()
        # plot_vehicle(car_state[0], car_state[1], car_state[2], length = 4, width = 2)
        # plot_car(car_state[0],car_state[1],car_state[2],opt_delta[0],'b')


        plt.plot(ob[:, 0], ob[:, 1], "o", markersize=10, color='black')

        plt.scatter(car_state[0], car_state[1])
        plt.plot(road.r_x, road.r_y)
        plt.plot(x_left,y_left)
        plt.plot(x_right,y_right)
        plt.plot(x_,y_)
        plt.plot(x_t,y_t)
        plt.plot(best_path.x[:], best_path.y[:], "-o", color='g')

        plot_vehicle(x_redraw, y_redraw, ugv.yaw_h, ugv.L_head * 1.3, 1.5)
        plot_vehicle(x_t_redraw, y_t_redraw, ugv.yaw_t, ugv.L_trailer * 1.1, 1.5)
        # i = 0
        # for path_c_i in path_c:
        #     if i % 10 == 0:
        #         plt.plot(path_c_i.x[:], path_c_i.y[:],"--", color='gray')
        #         print(i)
        #     i += 1

        plt.title("v[km/h]:" + str(ugv.v * 3.6)[0:4])
        plt.xlim(ugv.x-50,ugv.x+50)
        plt.ylim(ugv.y-50 , ugv.y+50)
        plt.grid(True)
        plt.pause(0.0001)
        # print(ugv.x, ugv.y, ugv.psi, ugv.v, ugv.a)
        # print(ugv.x, ugv.y, ugv.yaw_h, ugv.v, ugv.a)
    plt.cla()
    plt.plot(road.r_x, road.r_y)
    plt.plot(x_left, y_left)
    plt.plot(x_right, y_right)
    plt.plot(x_, y_)
    plt.plot(x_t, y_t)
    plt.plot(ob[:, 0], ob[:, 1], "o", markersize=10, color='black')
    plt.show()



main()

