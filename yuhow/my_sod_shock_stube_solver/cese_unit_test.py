"""
This cese_unit_test.py provides known analytic solution
, cese solution and also code for unit test.

Basically, both analytic solution and cese solution are
extracted extracted from Prof. Sin-Chung Chang's paper:

Journal of Computational Physics 119, 295-324 (1995)
The Method of Space-Time Conservation Element and Solution
Element -- A New Approach for Solving the Navier-Stokes and 
Eluer Equations

Sod tube condition:
    time: 0.2
    x range: 102 points btw -0.505 ~ 0.505

    driven section (-0.505 ~ 0):
        density  = 1.0
        pressure = 1.0
        velocity = 0.0

    working section (0 ~ 0.505)
        density  = 0.125
        pressure = 0.1
        velocity = 0.0
"""

import sys

class TestManager(object):

    def __init__(self):
        print 'Loading test manager...'

    @staticmethod
    def get_sod_shock_tube_analytic_solution_time_2ds():
        """
        return the specific analytic solution of sod shock tube when t = 0.2
        """
        file = open('analytic_100.dat', 'r')
    
        position = []
        density = []
        velocity = []
        pressure = []
        
        while True:
            line = file.readline().split()
            if not line: break
   
            position.append(float(line[0]))
            density.append(float(line[1]))
            velocity.append(float(line[2]))
            pressure.append(float(line[3]))

        return [position, density, velocity, pressure]

    @staticmethod
    def get_your_sod_shock_tube_cese_solution_time_2ds():
        """
        return your cese solution of sod shock tube when t = 0.2
        """
        file = open('100.dat', 'r')
        
        position = []
        density = []
        velocity = []
        pressure = []
        
        while True:
            line = file.readline().split()
            if not line: break
        
            position.append(float(line[0]))
            density.append(float(line[1]))
            velocity.append(float(line[2]))
            pressure.append(float(line[3]))
        
        return [position, density, velocity, pressure]

    @staticmethod
    def get_standard_sod_shock_tube_cese_solution_time_2ds():
        """
        return the standard cese solution of sod shock tube when t = 0.2
        """
        file = open('standard_cese_100.dat', 'r')
        
        position = []
        density = []
        velocity = []
        pressure = []
        
        while True:
            line = file.readline().split()
            if not line: break
        
            position.append(float(line[0]))
            density.append(float(line[1]))
            velocity.append(float(line[2]))
            pressure.append(float(line[3]))
        
        return [position, density, velocity, pressure]

    def deviation_calculator(self, a, b):
        """
        return |a - b|/max(|a|, |b|)
        """
        if a == 0 and b == 0:
            return 0
        else:
            if b == 0:
                return (a - 0) / a
            else:
                return abs((a - b) / b)

    def test_get_deviation_of_porting(self, delta_precision = 0.001):
        """
        delta_precision: a float number to claim two floating number value 
                         are equal.

        If the deviation of solutions is smaller than delta_precision in the
        same mesh point, these two solutions will be regarded as the same. 

        relative deviation = |solution_a - solution_b|/max(|solution_a|, |solution_b|)

        The value of delta_precision is set to be 0.001, which is a classical value of
        deviation that we can ignore.
        """

        solution_a = self.get_your_sod_shock_tube_cese_solution_time_2ds()
        solution_b = self.get_standard_sod_shock_tube_cese_solution_time_2ds()

        all_difference = []

        if len(solution_a) == 0 or len(solution_b) == 0:
            if len(solution_a) == 0:
                print 'The first input solution is empty. Please check it.'
            if len(solution_b) == 0:
                print 'The second input solution is empty. Please check it.'
            sys.exit()

        if len(solution_a[0]) != len(solution_b[0]):
            print 'Two solutions have different corresponding mesh points. Please check it.'
            sys.exit()
        
        is_different = False

        for i in range(0, len(solution_a[0])):
            difference = []
            dev_rho = self.deviation_calculator(solution_a[1][i], solution_b[1][i])
            dev_vel = self.deviation_calculator(solution_a[2][i], solution_b[2][i])
            dev_p   = self.deviation_calculator(solution_a[3][i], solution_b[3][i])

            if dev_rho >= delta_precision:
                is_different = True
                difference.append(dev_rho)
            else:
                difference.append(0)
                    
            if dev_vel >= delta_precision:
                is_different = True
                difference.append(dev_vel)
            else:
                difference.append(0)
                
            if dev_p >= delta_precision:
                is_different = True
                difference.append(dev_p)
            else:
                difference.append(0)

            all_difference.append(difference)

        if is_different:
            print 'Your solution is changed after porting.'
            return all_difference
        else:
            print 'Your solution is unchanged after porting.'
            return None
