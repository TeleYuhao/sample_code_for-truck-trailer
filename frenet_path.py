# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:12:32 2023

@author: Administrator
"""
import numpy as np
import copy
from frenet_spline2d import *
from math import *


MAX_SPEED = 40.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2  # maximum acceleration [m/ss]
MAX_CURVATURE = 3  # maximum curvature [1/m]
ROBOT_RADIUS = 2

MAX_ROAD_WIDTH = 2.5
SAMPLE_WIDTH = 0.5
MIN_SAMPLE_TIME = 6
MAX_SAMPLE_TIME = 8
SAMPLE_TIME = 0.2
SPEED_SAMPLE_LENGTH = 3/3.6
SPEED_SAMPLE_NUM = 1
TARGET_SPEED = 10/3.6

K_J = 0.1
K_T = 0.1
K_D = 1
K_LAT = 1.0
K_LON = 1.0

def plot_circle(x,y,radius):
    theta = np.linspace(0, 2*np.pi, 40)  # 创建角度数组
    x_ =x + radius * np.cos(theta)  # 计算圆上点的x坐标
    y_ =y + radius * np.sin(theta)  # 计算圆上点的y坐标

    # 绘制圆形
    plt.plot(x_, y_,color='black')


class FrenetPath:
    '''frenetic坐标系'''
    def __init__(self):
        self.t = []
        '''d-横向位移'''
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        '''s-纵向位移'''
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        
        self.cd = 0.0 #距离参考线的损失
        self.cv = 0.0 #速度损失函数
        self.cc = 0.0 #碰撞损失
        self.sc = 0.0 #路径切换损失
        
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt
    
class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial
        '''求解4阶多项式'''

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        '''计算一阶项'''
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        '''计算一介导数'''
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        '''计算二阶导数'''
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        '''计算三阶导数'''
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt
        
def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []
    target_speed = TARGET_SPEED
    # if s0 < 170:
    #     target_speed = TARGET_SPEED
    # else:
    #     target_speed = TARGET_SPEED - (s0 - 170)/3
    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, SAMPLE_WIDTH):
        '''di 即道路宽度边界循环  即侧向边界'''

        # Lateral motion planning
        for Ti in np.arange(MIN_SAMPLE_TIME, MAX_SAMPLE_TIME, SAMPLE_TIME):
            '''ti 时间长度循环'''
            fp = FrenetPath()
    #         '''横向曲线'''
            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            
    #         '''生成frenet坐标系下的纵向运动信息'''
            fp.t = [t for t in np.arange(0.0, Ti, SAMPLE_TIME)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]


            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(target_speed - SPEED_SAMPLE_LENGTH * SPEED_SAMPLE_NUM,
                                target_speed + SPEED_SAMPLE_LENGTH * SPEED_SAMPLE_NUM, SPEED_SAMPLE_LENGTH):

                '''速度循环'''
                tfp = copy.deepcopy(fp) 
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel,tv, 0.0, Ti)
                '''纵向曲线'''

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk LONTITUDE
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk LATERAL

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths

def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            # print('over speed')
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            # print('over acc')
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            # print('over kappa')
            continue
        elif not check_collision(fplist[i], ob):
            # print('collision happen')
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]

def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True

def calc_global_paths(fplist, csp):
    '''将frenet坐标系的位置等转换为全局坐标系下的位置和速度等'''
    for fp in fplist:
        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            if fp.s[i] > csp.s[-1]:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])
        fp.c.append(fp.c[-1])
    if fp.s[-1] > 200:

        print('here')

    return fplist


def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob):
    '''求解最有frenetic曲线'''
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    '''将frenet坐标系转换为全局坐标系'''
    fplist = calc_global_paths(fplist, csp)
    '''检查曲线是否符合约束'''
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp
    # if best_path == None:
        # print('here')

    return best_path
def define_road():
    wx = np.array([0.0, 10.0, 20.5, 35.0, 70.5])
    wy = np.array([0.0, -6.0, 5.0, 6.5, 0.0])

    wx = np.array([0.0, 10.0, 20.5, 35.0, 70.5, 100.3, 130, 150])
    wy = np.array([0.0, -6.0, 5.0, 6.5, 0.0, -20, 10, -30])
    
    
    # wx = np.array([0.0, 10.0, 20.5, 35.0, 70.5, 100.3, 130, 150])
    # wy = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -20, 10, -30])

    # wx = np.array([0.0, 10.0, 20.5, 35.0, 70.5,100.3,130,150])
    # wy = np.array([0.0, -6.0, 5.0, -6.5, 0.0,-20,10,-30])
    ob = np.array([
                [9.5,-6.05],
                [10,0],
                   ])
    ob = np.array([
                [8.728,-6.014],
                    [20.0, 2.5],
                    [30.0, 8.0],
                    [40.0, 7.0],
                    [50.0, 3.0]
                    ])
    ob = np.array([
        [8.653822161527355, -6.508718152190319],
        [20.0, 1.5],
        [30.0, 10.0],
        [40.0, 7.0],
        [50.0, 2.0],
        [95.32, -19.24],
        [63.9, 4.2],
        [140.66, 1.33],
        [113.74, -10.35]
    ])
    # ob = np.array([[20.0, 1.5],
    #                [30.0, 10.0],
    #                [40.0, 7.0],
    #                [50.0, 2.0],
    #                [95.32,-19.24],
    #                [140.66,1.33]
    #                ])
    # ob = np.array([
    #                [20.0, 2.5],
    #                [30.0, 10.0],
    #                [40.0, 7.5],
    #                [50.0, 2.0]
    #                ])
    # ob = np.array([[19.04159893557086, 3.707497866391803],
    #                [49.052568626385856, 5.700641708539293],
    #
    #                ])
    # ob = np.array([[20,2.5]])
    road = Spline2D(wx, wy)

    s_left = np.linspace(0, road.s[-1], int(road.s[-1] / 0.1))
    d_left = 2.5
    s_right = np.linspace(0, road.s[-1], int(road.s[-1] / 0.1))
    d_right = -2.5
    x_left, y_left = np.zeros_like(s_left), np.zeros_like(s_left)
    x_right, y_right = np.zeros_like(s_left), np.zeros_like(s_left)

    for i in range(len(s_left)):
        x_left_, y_left_ = road.frenet_to_cartesian1D(s_left[i], d_left)
        x_right_, y_right_ = road.frenet_to_cartesian1D(s_right[i], d_right)
        x_left[i] = x_left_
        y_left[i] = y_left_
        x_right[i] = x_right_
        y_right[i] = y_right_
    return road,x_left,y_left,x_right,y_right,ob


import matplotlib.pyplot as plt


def get_best_path(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0 ,road ,ob):
    path_all = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    calc_global_paths(path_all, road)
    path_check = check_paths(path_all, ob)

    min_cost = float("inf")
    best_path = None
    for fp in path_check:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp
    return best_path

def plot_car(x, y, yaw, steer=0.0, truck_color="-k"):  # pragma: no cover

    # Vehicle parameters
    LENGTH = 0.4*10  # [m]
    WIDTH = 0.2*10  # [m]
    BACK_TO_WHEEL = 0.1*10  # [m]
    WHEEL_LEN = 0.03*10  # [m]
    WHEEL_WIDTH = 0.02 *10 # [m]
    TREAD = 0.07  *10# [m]
    WB = 0.25*10

    outline = np.array(
        [[-BACK_TO_WHEEL, (LENGTH - BACK_TO_WHEEL), (LENGTH - BACK_TO_WHEEL),
          -BACK_TO_WHEEL, -BACK_TO_WHEEL],
         [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array(
        [[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH -
          TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[cos(yaw), sin(yaw)],
                     [-sin(yaw), cos(yaw)]])
    Rot2 = np.array([[cos(steer), sin(steer)],
                     [-sin(steer), cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truck_color)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truck_color)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truck_color)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truck_color)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truck_color)
    plt.plot(x, y, "*")

def main():
    road, x_left, y_left, x_right, y_right,ob = define_road()

    c_speed = 10.0 / 3.6  # current speed [m/s]
    c_accel = 0.0  # current acceleration [m/ss]
    c_d = 0.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position
    i=0
    path_c = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)

    calc_global_paths(path_c,road)
    path_check = check_paths(path_c, ob)
    min_cost = float("inf")
    best_path = None
    for fp in path_check:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    path_b_t = get_best_path(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0 ,road ,ob)
    plt.plot(road.r_x,road.r_y,"--",color ="black")
    plt.plot(x_left,y_left,color ="black")
    plt.plot(x_right,y_right,color ="black")
    plt.plot(ob[:, 0], ob[:, 1], "o", markersize=10, color='black')
    plt.plot(best_path.x[1:], best_path.y[1:], "-o",  color='green')
    plt.plot(path_b_t.x[:],path_b_t.y[:],"-x",color = "red")


    # for path_c_i in path_c:
    for path_c_i in path_check:
        if i % 10 == 0:
            if best_path==path_c_i:
                continue
            plt.plot(path_c_i.x[:], path_c_i.y[:], color='gray')

        i += 1
    print("采样点的个数为：",len(path_c_i.x))
    print(path_c_i.x)
    plt.show()

# main()