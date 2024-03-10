# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:24:34 2023

@author: Administrator
"""
import bisect
import math
from typing import Tuple
from dataclasses import dataclass, field
import numpy as np



class Spline:
    """Use natural spline to construct the curve with a given list a points
       See Numerical Analysis, 9th Edition, Algorithm 3.4, Natural Cubic Spline.
    """

    def __init__(self, x_list: np.ndarray, y_list: np.ndarray) -> None:
        self.x_list = x_list
        n: int = x_list.size
        h = np.diff(x_list)

        main_diag = np.ones(n)
        main_diag[1:-1] = 2 * (h[1:] + h[:-1])
        A = np.diag(main_diag) + np.diag(h, k=1) + np.diag(h, k=-1)
        A[0, 1] = 0
        A[-1, -2] = 0
        b = np.zeros(n)
        b[1:-1] = 3 * (np.divide(np.diff(y_list[1:]), h[1:]) -
                       np.divide(np.diff(y_list[:-1]), h[:-1]))

        self.c = np.linalg.solve(A, b)
        self.a = y_list
        self.d = np.divide(np.diff(self.c), 3 * h)
        self.b = np.divide(np.diff(self.a),
                           h) - np.multiply(h, self.c[1:] + 2 * self.c[:-1]) / 3

    def calculate_approximation(self, pos_x: float) -> float:
        """Given x coordinate, use Spline to approximate y coordinate

        Args:
            pos_x (float): x coordinate

        Returns:
            float: approximated y coordinate

        Remark:
            When x is out of the range, we still use spline to approximate it,
            But the precision is not granted!
        """
        index = bisect.bisect(self.x_list, pos_x) - 1
        index = max(min(index, self.x_list.size - 2), 0)
        dx = pos_x - self.x_list[index]
        return self.a[index] + self.b[index] * dx + \
               self.c[index] * dx**2.0 + self.d[index] * dx**3.0

    def calculate_derivative(self, pos_x: float) -> float:
        """Given x coordinate, use Spline to approximate dy/dx at x

        Args:
            pos_x (float): x coordinate

        Returns:
            float: approximated dy/dx at x

        Remark:
            When x is out of the range, we still use spline to approximate it,
            But the precision is not granted!
        """
        index = bisect.bisect(self.x_list, pos_x) - 1
        index = max(min(index, self.x_list.size - 2), 0)
        dx = pos_x - self.x_list[index]
        return self.b[index] + 2.0 * self.c[index] * dx + 3.0 * self.d[index] * dx**2.0

    def calculate_second_derivative(self, pos_x: float) -> float:
        """Given x coordinate, use Spline to approximate d^2y/dx^2 at x

        Args:
            pos_x (float): x coordinate

        Returns:
            float: approximated d^2y/dx^2 at x

        Remark:
            When x is out of the range, we still use spline to approximate it,
            But the precision is not granted!
        """
        index = bisect.bisect(self.x_list, pos_x) - 1
        index = max(min(index, self.x_list.size - 2), 0)
        dx = pos_x - self.x_list[index]
        return 2.0 * self.c[index] + 6.0 * self.d[index] * dx

    def calculate_third_derivative(self, pos_x: float) -> float:
        """Given x coordinate, use Spline to approximate d^3y/dx^3 at x

        Args:
            pos_x (float): x coordinate

        Returns:
            float: approximated d^3y/dx^3 at x

        Remark:
            When x is out of the range, we still use spline to approximate it,
            But the precision is not granted!
        """
        index = bisect.bisect(self.x_list, pos_x) - 1
        index = max(min(index, self.x_list.size - 2), 0)
        return 6.0 * self.d[index]


class Spline2D:
    """A 2 dimensional Spline with x coordinates and y coordinates are 1d spline 
       corresponding to cumulative distance
    """
    def __init__(self, x_list: np.ndarray, y_list: np.ndarray):
        """Use a list of points [x_list[i], y_list[i]] to construct a 2d spline that
           passes through the list of points.

        Args:
            x_list (np.ndarray): list of x coordinates
            y_list (np.ndarray): list of y coordinates
        """
        dx = np.diff(x_list)
        dy = np.diff(y_list)
        ds = np.hypot(dx, dy)

        self.s = np.zeros_like(x_list)
        self.s[1:] = np.cumsum(ds)
        self.sx = Spline(self.s, x_list)
        self.sy = Spline(self.s, y_list)

        self.x_list = x_list
        self.y_list = y_list
        self.yaw = []
        self.r_x ,self.r_y = self.generate_target_course(x_list,y_list)
        self.s_,self.d_ = [],[]
        for i in range(len(self.r_x)):
            s_i,d_i = self.cartesian_to_frenet1D(self.r_x[i], self.r_y[i])
            self.s_.append(s_i)
            self.d_.append(d_i)
        self.kx = self.get_curvature(self.r_x, self.r_y)
        
        
        
    def get_x_list(self):
        return self.x_list
    
    def get_y_list(self):
        return self.y_list

    def calc_position(self, pos_s: float) -> Tuple[float, float]:
        """Given frenet coordinate (pos_s, 0), compute its cartesian coordinate.

        Args:
            pos_s (float): longitudinal coordinate

        Returns:
            Tuple[float, float]: cartesian coordinates corresponding to (pos_s, 0)
        """
        pos_x = self.sx.calculate_approximation(pos_s)
        pos_y = self.sy.calculate_approximation(pos_s)

        return pos_x, pos_y

    def calc_curvature(self, pos_s: float) -> float:
        """compute the curvature at position (pos_s, 0)

        Args:
            pos_s (float): longitudinal coordinate

        Returns:
            float: curvature (absolute value) at (pos_s, 0)
        """
        dx = self.sx.calculate_derivative(pos_s)
        ddx = self.sx.calculate_second_derivative(pos_s)
        dy = self.sy.calculate_derivative(pos_s)
        ddy = self.sy.calculate_second_derivative(pos_s)
        k = abs(ddy * dx - ddx * dy) / ((dx**2 + dy**2)**1.5)
        return k

    def calc_curvature_derivative(self, pos_s: float) -> float:
        """compute the derivative of curvature at position (pos_s, 0)

        Args:
            pos_s (float): longitudinal coordinate

        Returns:
            float: derivative of curvature at (pos_s, 0)
        """
        dx = self.sx.calculate_derivative(pos_s)
        ddx = self.sx.calculate_second_derivative(pos_s)
        dddx = self.sx.calculate_third_derivative(pos_s)
        dy = self.sy.calculate_derivative(pos_s)
        ddy = self.sy.calculate_second_derivative(pos_s)
        dddy = self.sy.calculate_third_derivative(pos_s)

        a = dx * ddy - dy * ddx
        b = dx * dddy - dy * dddx
        c = dx * ddx + dy * ddy
        d = dx * dx + dy * dy
        kd = (b * d - 3.0 * a * c) / (d * d * d)**(2.5)
        return kd

    def calc_yaw(self, pos_s: float) -> float:
        """compute the yaw angle in frenet coordinate (pos_s, 0)

        Args:
            pos_s (float): longitudinal coordinate  

        Returns:
            float: yaw angle in radians
        """
        dx = self.sx.calculate_derivative(pos_s)
        dy = self.sy.calculate_derivative(pos_s)
        yaw = math.atan2(dy, dx)
        return yaw

    def frenet_to_cartesian1D(self, pos_s: float,
                              pos_d: float) -> Tuple[float, float]:
        """Given the frenet coordinate (pos_s, pos_d), compute its cartesian coordinate

        Args:
            pos_s (float): longitudinal coordinate  
            pos_d (float): lateral coordinate

        Returns:
            Tuple[float, float]: cartesian coordinate of (pos_s, pos_d)
        """
        rx, ry = self.calc_position(pos_s)
        ryaw = self.calc_yaw(pos_s)
        x = rx - math.sin(ryaw) * pos_d
        y = ry + math.cos(ryaw) * pos_d
        return x, y

    def frenet_to_cartesian2D(self, s: float, d: float, s_d: float,
                              d_d: float) -> Tuple[float, float, float, float]:
        x, y = self.frenet_to_cartesian1D(s, d)
        r_yaw, r_kappa = self.calc_yaw(s), self.calc_curvature(s)
        speed = np.hypot(s_d * (1 - r_kappa * d), d_d)
        yaw = np.arctan2(s_d, (1 - r_kappa * d) * d_d) + r_yaw
        yaw = np.fmod(yaw, np.pi)
        return x, y, speed, yaw

    def cartesian_to_frenet1D(self, pos_x: float, pos_y: float) -> Tuple[float, float]:
        """Given the cartesian coordinate (pos_x, pos_y), computes its frenet coordinate

        Args:
            pos_x (float): x coordinate
            pos_y (float): y coordinate

        Returns:
            Tuple[float ,float]: the corresponding frenet coordinate (s, d)
        """
        s = self.find_nearest_rs(pos_x, pos_y)
        rx, ry = self.calc_position(s)
        ryaw = self.calc_yaw(s)

        dx = pos_x - rx
        dy = pos_y - ry
        cross_rd_nd = math.cos(ryaw) * dy - math.sin(ryaw) * dx
        d = math.copysign(math.sqrt(dx * dx + dy * dy), cross_rd_nd)

        return s, d

    def cartesian_to_frenet2D(
            self, x: float, y: float, yaw: float,
            speed: float) -> Tuple[float, float, float, float]:
        
        s, d = self.cartesian_to_frenet1D(x, y)
        r_yaw, r_kappa = self.calc_yaw(s), self.calc_curvature(s)

        s_d = speed * np.cos(yaw - r_yaw) / (1 - r_kappa * d)
        d_d = speed * np.sin(yaw - r_yaw)#横向速度
        # d_d = (1 - r_kappa * d) * np.tan(yaw - r_yaw)
        return s, d, s_d, d_d
    
    
    def cartesian_to_frenet3D(
            self, x: float, y: float, yaw: float,
            speed: float, acceleration: float) -> Tuple[float, float, float, float,float]:
        
        s ,d ,s_d ,d_d = self.cartesian_to_frenet2D(x,y,yaw,speed)
        r_yaw, r_kappa, r_d_kappa = self.calc_yaw(s), self.calc_curvature(s) ,self.calc_curvature_derivative(s)
        
        delta_theta = yaw - r_yaw
        d_dd = acceleration*np.sin(delta_theta)#横向加速度
        index = self.find_nearest_index(x, y)
        k_x = self.kx[index]
        
        d_ = (1 - r_kappa * d) * np.tan(delta_theta)#横向位移对纵坐标的一阶导数
        d__ = -(r_d_kappa*d+r_kappa*d_)*np.tan(delta_theta)+(1-r_kappa*d)/ \
                (np.cos(delta_theta))**2*(k_x*(1-r_kappa*d)/np.cos(delta_theta)-r_kappa)
        
        
        
        
        c_d_theta = (1 - r_kappa*d)/(k_x* np.cos(delta_theta)) - r_kappa
        k_r_d_prime = r_d_kappa*d + r_kappa*d_
        # s_dd = (acceleration * np.cos(delta_theta) - s_d**2 *(d_d* c_d_theta - kappa_r_d_prime)) /(1 - r_kappa*d)
        
        
        # print(s_d**2 *(d_*c_d_theta - k_r_d_prime))
        # print(d_)
        # print(k_x)
        # print(delta_theta)
        # print(c_d_theta)
        # print(r_kappa)
        # print(k_r_d_prime)
        # print(d_*c_d_theta)
        # print(k_r_d_prime)
        s_dd = (acceleration*np.cos(delta_theta) - s_d**2 *(d_*c_d_theta - k_r_d_prime))/(1 - r_kappa*d)
        # s_dd = acceleration*np.cos(delta_theta)
                 
        
        return s,d,s_d,d_d,s_dd,d_dd

    def find_nearest_rs(self, pos_x: float, pos_y: float) -> float:
        """find the closest frenet coordinate on reference line, i.e.
            argmin_{s} (x(s, 0) - pos_x)^2 + (y(s, 0) - pos_y)^2
            where x(s, 0), y(s, 0) are the cartesian coordinates of (s, 0)
            The computation is from coarse to fine to reduce computational burden.

        Args:
            pos_x (float): x coordinate
            pos_y (float): y coordinate

        Returns:
            float: the corresponding s coordinate on reference line
        """
        precision_list = [5, 1, 0.05]
        left, right = self.s[0], self.s[-1]
        for precision in precision_list:
            refined_s = np.arange(left, right + precision, precision)
            positions = np.array([list(self.calc_position(s)) for s in refined_s])
            dists = np.linalg.norm(positions - np.array([pos_x, pos_y]), axis=1)
            ri = np.argmin(dists)
            rs = refined_s[ri]
            
            left, right = rs - precision, rs + precision

        return rs
    
    def find_nearest_index(self, pos_x: float, pos_y: float) -> float:
        """find the closest frenet coordinate on reference line, i.e.
            argmin_{s} (x(s, 0) - pos_x)^2 + (y(s, 0) - pos_y)^2
            where x(s, 0), y(s, 0) are the cartesian coordinates of (s, 0)
            The computation is from coarse to fine to reduce computational burden.

        Args:
            pos_x (float): x coordinate
            pos_y (float): y coordinate

        Returns:
            float: the corresponding s coordinate on reference line
        """
        delta_x = np.array([pos_x]) - self.r_x
        delta_y = np.array([pos_y])- self.r_y
        index = np.argmin(delta_x**2 + delta_y**2)
        return index
    

    
    def generate_target_course(self,x, y):
        '''csp-二次样条曲线插值对象'''
        s = np.arange(0, self.s[-1], 0.1)
    
        rx, ry, ryaw, rk ,rkd = [], [], [], [],[]
        for i_s in s:
            ix, iy = self.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            # ryaw.append(csp.calc_yaw(i_s))
            # rk.append(csp.calc_curvature(i_s))
            # rkd.append(csp.calc_curvature_derivative(i_s))
    
        return rx, ry
        # return rx, ry, ryaw, rk, rkd,csp
    def get_curvature(self,x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5)
        return curvature



@dataclass
class CertesianState:
    t: float = 1.0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    cur: float = 0.0
    vel: float = 0.0
    acc: float = 0.0

@dataclass
class FrenetState:
    t: float = 2.0
    s: float = 0.0  # lane pos
    routeIdx: int = 0
    s_d: float = 0.0
    s_dd: float = 0.0
    s_ddd: float = 0.0
    d: float = 0.0
    d_d: float = 0.0
    d_dd: float = 0.0
    d_ddd: float = 0.0

@dataclass
class State(CertesianState, FrenetState):
    """State of the trajectory."""

    def __post_init__(self):
        if self.vel == 0 and self.s_d != 0:
            self.vel = math.sqrt(self.s_d**2 + self.d_d**2)

    """
    Modified from: https://blog.csdn.net/u013468614/article/details/108748016
    """

    def complete_cartesian2D(self, rx: float, ry: float, ryaw: float,
                             rkappa: float) -> None:
        cos_theta_r = math.cos(ryaw)
        sin_theta_r = math.sin(ryaw)

        self.x = rx - sin_theta_r * self.d
        self.y = ry + cos_theta_r * self.d
        if self.s_d <= 1e-1:
            self.s_d = 1e-1
            self.vel = 0
            self.yaw = None
        else:
            one_minus_kappa_r_d = 1 - rkappa * self.d
            self.vel = math.sqrt(one_minus_kappa_r_d**2 * self.s_d**2 +
                                 self.d_d**2)
            self.yaw = math.asin(self.d_d / self.vel) + ryaw
        return

    def complete_frenet2D(self, rs: float, rx: float, ry: float, ryaw: float,
                          rkappa: float) -> None:
        self.s = rs
        dx = self.x - rx
        dy = self.y - ry

        cos_theta_r = math.cos(ryaw)
        sin_theta_r = math.sin(ryaw)
        cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
        self.d = math.copysign(math.sqrt(dx * dx + dy * dy), cross_rd_nd)

        delta_theta = self.yaw - ryaw
        sin_delta_theta = math.sin(delta_theta)
        cos_delta_theta = math.cos(delta_theta)
        one_minus_kappa_r_d = 1 - rkappa * self.d
        self.s_d = self.vel * cos_delta_theta / one_minus_kappa_r_d
        self.d_d = self.vel * sin_delta_theta
        return
    
# def generate_target_course(x, y):
#     '''csp-二次样条曲线插值对象'''
#     csp = Spline2D(x, y)
#     s = np.arange(0, csp.s[-1], 0.1)

#     rx, ry, ryaw, rk ,rkd = [], [], [], [],[]
#     for i_s in s:
#         ix, iy = csp.calc_position(i_s)
#         rx.append(ix)
#         ry.append(iy)
#         ryaw.append(csp.calc_yaw(i_s))
#         rk.append(csp.calc_curvature(i_s))
#         rkd.append(csp.calc_curvature_derivative(i_s))


#     return rx, ry, ryaw, rk, rkd,csp
def get_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5)
    return curvature

# def main():
#     import matplotlib.pyplot as plt
#     wx = np.array([0.0, 10.0, 20.5, 35.0, 70.5])
#     wy = np.array([0.0, -6.0, 5.0, 6.5, 0.0])
#     road = Spline2D(wx, wy)
#
#     s_left = np.linspace(0,road.s[-1],int(road.s[-1]/0.1))
#     d_left = 2.5
#     s_right = np.linspace(0,road.s[-1],int(road.s[-1]/0.1))
#     d_right = -2.5
#     x_left,y_left = np.zeros_like(s_left),np.zeros_like(s_left)
#     x_right,y_right = np.zeros_like(s_left),np.zeros_like(s_left)
#
#     for i in range(len(s_left)):
#         x_left_,y_left_ = road.frenet_to_cartesian1D(s_left[i], d_left)
#         x_right_,y_right_ = road.frenet_to_cartesian1D(s_right[i], d_right)
#         x_left[i] = x_left_
#         y_left[i] = y_left_
#         x_right[i] = x_right_
#         y_right[i] = y_right_
#
#
#
#
#
#     # s = np.zeros_like(road.x)
#     # for i in range(len(s)):
#     #     s_point = road.calc_position(road.r_x,road.)
#
#     plt.plot(road.r_x,road.r_y)
#     plt.plot(x_left,y_left)
#     plt.plot(x_right,y_right)
#     plt.show()
#
# # main()






