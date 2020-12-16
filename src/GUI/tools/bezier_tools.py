from numpy import array, flipud
from math import sqrt
import cmath
import integration_tools

class Bezier:
    '''A class representing a cubic bezier curve'''
    
    def __init__(self, ctrlPoly):
        '''Create a Bezier object from control points'''
        self.ctrlPoly = array([array(ctrlPt) for ctrlPt in ctrlPoly])
    
    def set(self, i, point):
        '''Change the ith control point of the cubic bezier'''
        self.ctrlPoly[i] = array(point)
    
    def reverse(self):
        self.ctrlPoly = flipud(self.ctrlPoly)
    
    def q(self, t):
        '''Evaluate a cubic bezier at t'''
        return (1.0-t)**3 * self.ctrlPoly[0] + 3*(1.0-t)**2 * t * self.ctrlPoly[1] + 3*(1.0-t)* t**2 * self.ctrlPoly[2] + t**3 * self.ctrlPoly[3]

    def prime(self, t):
        '''Evaluate the first derivative of a cubic bezier at t'''
        return 3*(1.0-t)**2 * (self.ctrlPoly[1]-self.ctrlPoly[0]) + 6*(1.0-t) * t * (self.ctrlPoly[2]-self.ctrlPoly[1]) + 3*t**2 * (self.ctrlPoly[3]-self.ctrlPoly[2])

    def primeprime(self, t):
        '''Evaluate the second derivative of a cubic bezier at t'''
        return 6*(1.0-t) * (self.ctrlPoly[2]-2*self.ctrlPoly[1]+self.ctrlPoly[0]) + 6*(t) * (self.ctrlPoly[3]-2*self.ctrlPoly[2]+self.ctrlPoly[1])

    def speed(self, t):
        '''Evaluate the speed of a cubic bezier at t'''
        q_vel = self.prime(t)
        return sqrt(q_vel[0]**2 + q_vel[1]**2)

    def arc_length(self, t1, t2, num_intervals=100):
        '''Calculate the arc length of a cubic bezier from t1 to t2'''
        return integration_tools.simpson(lambda t: self.speed(t), t1, t2, num_intervals)

    def t(self, s, num_intervals=100):
        '''Calculate the arc length parameterization of a cubic bezier'''
        if all(self.ctrlPoly[0] == self.ctrlPoly[1]):
            # We reparameterize by r = (3 - 2t)t^2
            def r_speed(t):
                r_vel = -1 * self.ctrlPoly[0] + (1 - t / (2 * (1 - t))) * self.ctrlPoly[2] + (t / (2 * (1 - t))) * self.ctrlPoly[3]
                return sqrt(r_vel[0]**2 + r_vel[1] ** 2)
            s_n, r_n = integration_tools.rk4(lambda s, t: 1/r_speed(t), 0, 0, s, num_intervals)
            return r_n[-1]
        if all(self.ctrlPoly[0] == self.ctrlPoly[1]) or all(self.ctrlPoly[2] == self.ctrlPoly[3]):
            raise ValueError('Endpoints and handles of Bezier curve coincide.')
        s_n, t_n = integration_tools.rk4(lambda s, t: 1/self.speed(t), 0, 0, s, num_intervals)
        return t_n[-1]
    
    def split(self, t):
        '''Split a cubic bezier into two new curves at t using De Casteljau's algorithm'''
        A, B, C, D = self.ctrlPoly.copy()
        L = (1.0 - t) * A + t * B
        M = (1.0 - t) * B + t * C
        N = (1.0 - t) * C + t * D
        P = (1.0 - t) * L + t * M
        Q = (1.0 - t) * M + t * N
        R = (1.0 - t) * P + t * Q
        
        bez1 = Bezier([A, L, P, R])
        bez2 = Bezier([R, Q, N, D])
        return [bez1, bez2]
        
    def split_all(self, *ts):
        '''Split a cubic bezier at multiple t values'''
        result = []
        current_curve = self
        old_t = 0
        
        for t in ts:
            if t < old_t:
                raise ValueError('The provided values are not in increasing order.')
            
            new_t = (t - old_t) / (1 - old_t) if old_t != 1 else 1
            bez1, bez2 = current_curve.split(new_t)
            result.append(bez1)
            current_curve = bez2
            old_t = t
        result.append(current_curve)
        return result 
                    
    def slice_old(self, t):
        '''Split a cubic bezier into two new cubic beziers at t'''
        x1, y1 = self.ctrlPoly[0]
        x2, y2 = self.ctrlPoly[1]
        x3, y3 = self.ctrlPoly[2]
        x4, y4 = self.ctrlPoly[3]

        x12 = (x2-x1)*t+x1
        y12 = (y2-y1)*t+y1

        x23 = (x3-x2)*t+x2
        y23 = (y3-y2)*t+y2

        x34 = (x4-x3)*t+x3
        y34 = (y4-y3)*t+y3

        x123 = (x23-x12)*t+x12
        y123 = (y23-y12)*t+y12

        x234 = (x34-x23)*t+x23
        y234 = (y34-y23)*t+y23

        x1234 = (x234-x123)*t+x123
        y1234 = (y234-y123)*t+y123
        
        bez1 = Bezier(array([[x1, y1], [x12, y12], [x123, y123], [x1234, y1234]]))
        bez2 = Bezier(array([[x1234,y1234],[x234,y234],[x34,y34],[x4,y4]]))
        return [bez1, bez2]

class BezierChain:
    '''A class representing a chain of Bezier curves.'''
    
    def __init__(self, ctrlPolys):
        self.curves = [Bezier(ctrlPoly) for ctrlPoly in ctrlPolys]
    
    def reverse(self):
        for curve in self.curves:
            curve.reverse()
        self.curves = self.curves[::-1]
    
    def cycle(self, i):
        '''Set the ith curve to be the first in the chain'''
        self.curves = self.curves[i:] + self.curves[:i]
    
    def q(self, i, t):
        '''Evaluate the ith Bezier in the chain at t'''
        return self.curves[i].q(t)
    
    def remove(self, i, t1, j, t2):
        '''Remove a portion of the Bezier chain between two t values
        on the ith and jth curve'''
        if (i == j):
            if t1 < t2:
                new_ith_curve, discard, new_next_curve = self.curves[i].split_all(t1, t2)
                self.curves = [new_next_curve] + self.curves[i + 1:] + self.curves[:i] + [new_ith_curve]
            elif t2 < t1:
                print("This shouldn't have happened")
                first_discard, new_ith_curve, second_discard = self.curves[i].split_all(t1, t2)
                self.curves = new_ith_curve
        else:
            new_ith_curve, discard = self.curves[i].split(t1)
            discard, new_jth_curve = self.curves[j].split(t2)
            if i < j:
                self.curves = [new_jth_curve] + self.curves[j + 1:] + self.curves[:i] + [new_ith_curve]
            else:
                self.curves = [new_jth_curve] + self.curves[j + 1:i] + [new_ith_curve]
            #check edge cases
    
    def search_control_points(self, target):
        '''Searches the Bezier chain for a control point, returning the index of the curve along
        the chain and the index of the control point'''
        for i, curve in enumerate(self.curves):
            for j, point in enumerate(curve.ctrlPoly):
                if (abs(target[0] - point[0]) < 10) and (abs(target[1] - point[1]) < 10):
                    return (i, j)
        return (-1, -1) 
    
    def search_all_points(self, target, num_samples=100, tolerance=1):
        '''Searches the Bezier chain for a point, returning the index of the curve along
        the chain and the t-value of the point'''
        for i, curve in enumerate(self.curves):
            for j in range(num_samples + 1):
                point = curve.q(j / num_samples)
                if (abs(target[0] - point[0]) <= tolerance) and (abs(target[1] - point[1]) <= tolerance):
                    return (i, j / num_samples)
        return (-1, -1)
    
    def signed_area(self):
        '''Compute the signed area enclosed by the Bezier chain'''
        signed_area = 0
        l = len(self.curves)
        for i, curve in enumerate(self.curves):
            x1, y1 = curve.q(0)
            x2, y2 = self.curves[(i + 1) % l].q(0)
            signed_area += (x1 * y2 - x2 * y1)
        return signed_area / 2
    
    def set(self, old_point, new_point):
        '''Change the control point of a Bezier in the chain'''
        i, j = search_control_points(old_point)
        if (i, j) != (-1, -1):
            self.curves[i].set(j, new_point)
    
    def length(self):
        '''Returns the total length of the Bezier chain'''
        if self.len is None:
            self.len = sum(curve.arc_length(0, 1) for curve in self.curves)
        return self.len
    
    def even_points(self, num_points):
        '''Returns n + 1 evenly spaced points along the Bezier chain'''
        curve_lengths = [curve.arc_length(0, 1) for curve in self.curves]
        spacing = sum(curve_lengths) / num_points
        
        i = 0
        traveled_dist = 0
        curve_dist = curve_lengths[0]
        for _ in range(num_points):
            while traveled_dist > curve_dist:
                i += 1
                curve_dist = sum(curve_lengths[:i + 1])
            curve = self.curves[i]
            s = traveled_dist - (curve_dist - curve_lengths[i])
            yield curve.q(curve.t(s))
            traveled_dist += spacing
        yield curve.q(1)