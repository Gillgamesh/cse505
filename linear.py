import math
from typing import List
import z3
import numpy as np

from z3 import Real, Int, solve, RealVector, Solver, Or, And



def mat_transpose(m1):
    return np.array(m1).transpose()

def mat_mul(m1, m2):
    '''
    NxM matrix by MxK matrix multiplication
    '''
    N = len(m1)
    assert(N>=1)
    M = len(m1[0])
    assert(M>=1)
    assert(len(m2)==M)
    K = len(m2[0])
    '''
    return i'th row dot the j'th column for each entry i,j
    '''
    return_matrix = np.zeros((N,K)).tolist()
    for i in range(N):
        for j in range(K):
            return_matrix[i][j] = dot_product(
                m1[i],
                [m2[l][j] for l in range(M)]
            )
    return return_matrix

def transform_vector(mat, vec):
    col_vector = mat_mul(mat, mat_transpose([vec]))
    return mat_transpose(col_vector)[0]

def dot_product(x, y):
    '''
    z3-compatible dot product
    '''
    return sum([x[i] * y[i] for i in range(len(x))])

def wedge_product(v1, v2):
    '''
    Given 2 origin-centered vectors, calculate their signed area.
    '''
    x1, y1 = v1
    x2, y2 = v2
    return x1*y2 - x2*y1


def left_test(p1, p2, p3):
    '''
    Check if p3 is to the left (or on) of the line defined by p1, p2
    '''
    # treat p1 as origin point:
    # let (v:= p2-p1, w:=p3-p1)
    v = (p2[0]-p1[0], p2[1]-p1[1])
    w = (p3[0]-p1[0], p3[1]-p1[1])
    return wedge_product(v, w) >= 0


def right_test(p1, p2, p3):
    '''
    Check if p3 is to the right (or on) of the line defined by p1, p2
    '''
    # treat p1 as origin point:
    # let (v:= p2-p1, w:=p3-p1)
    v = (p2[0]-p1[0], p2[1]-p1[1])
    w = (p3[0]-p1[0], p3[1]-p1[1])
    return wedge_product(v, w) <= 0


class ConvexPolygon():
    '''
    A convex polygon defined by a set of adjacent coordinates.
    '''
    coordinates = []
    base_name: str

    def __init__(self, coords, base_name="shape"):
        self.coordinates = coords
        self.base_name = base_name

    # def add_constraints(self, solver: Solver):
    #     x = Int(f'{self.base_name}__{i}_{j}')
    #     pass

    def constrain_in_shape(self, var, solver: Solver):
        N = len(self.coordinates)
        x, y = var
        for i in range(N):
            p1, p2, p3 = (
                self.coordinates[i], 
                self.coordinates[(i + 1) % N],
                self.coordinates[(i + 2) % N]
            )
            if left_test(p1, p2, p3):
                solver.add(left_test(p1, p2, (x, y)))
            else:
                solver.add(right_test(p1, p2, (x, y)))

    def constrain_not_in_shape(self, var, solver:Solver):    
        clauses = []
        N = len(self.coordinates)
        x, y = var
        for i in range(N):
            p1, p2, p3 = (
                self.coordinates[i], 
                self.coordinates[(i + 1) % N],
                self.coordinates[(i + 2) % N]
            )
            if left_test(p1, p2, p3):
                clauses.append(right_test(p1, p2, (x, y)))
            else:
                clauses.append(left_test(p1, p2, (x, y)))
        solver.add(Or(clauses))

    def transform(self, func)->'ConvexPolygon':
        return ConvexPolygon(
            coords=[func(coord) for coord in self.coordinates],
            base_name=self.base_name+"_t"
        )

def matrix_equality(m1, m2):
    N = len(m1)
    # assert(N>=1)
    M = len(m1[0])
    # assert(M>=1)
    # assert(len(m2)==M)
    # K = len(m2[0])
    clauses = []
    for i in range(N):
        for j in range(M):
            clauses.append(m1[i][j]==m2[i][j])
    return And(clauses)
    



if __name__ == "__main__":
    x = Real('x')
    # solve((x+1) * (x+2) == 0)
    # triangle = ConvexPolygon(
    #     [
    #         [0,0],
    #         [1,0],
    #         [0,1]
    #     ]
    # )
    # solver = Solver()
    # x = Real('x')
    # y = Real('y')
    # triangle.constrain_in_shape([x,y], solver)
    # solver.push()
    # print(solver.check())
    # solver.pop()
    solver = Solver()
    input_shape = ConvexPolygon(
        [
            [-5, -5],
            [5, -5],
            [5, 5],
            [-5, 5]
        ]
    )
    x = Real('x')
    y = Real('y')
    transformation = [
        [1,1],
        [-1,1]
    ]
    output_constraint_box = input_shape.transform(
        lambda p: transform_vector(1.3*np.eye(2), p)
    )
    input_shape.constrain_in_shape([x,y], solver)
    output_constraint_box.constrain_not_in_shape(
        transform_vector(transformation, [x, y]),
        solver
    )
    if solver.check() == z3.sat:
        print(solver.model())
    else:
        print("unsat")
    # mat = [RealVector('x', 2), RealVector('y', 2)]
    # solve(matrix_equality(mat_mul(mat, start), iden))


