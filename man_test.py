from manim import *
import manim

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

class NonlinearTransformSeries(manim.LinearTransformationScene):
    def __init__(self):
        LinearTransformationScene.__init__(
            self,
            show_coordinates=True,
            leave_ghost_vectors=True
        )
    def construct(self):
        matrix = [
            [1,-1],
            [1,1]
        ]
        self.apply_matrix(matrix)
        # self.wait(0)
        # self.apply_nonlinear_transformation(relu)
        # self.wait(0)
        # self.apply_matrix(matrix)
        self.wait()

class ApplyMatrixExample(manim.Scene):
    def construct(self):
        matrix = [
            [1,-1],
            [1,1]
        ]
        self.add(
            manim.NumberPlane(),
            manim.Square(side_length=2*1.3, color=manim.utils.color.RED),
            # manim.Vector([1,1]),
            # manim.Vector([-1,1]),
        )
        violation_vector = manim.Vector([1, -3/10], color=manim.utils.color.PINK)
        # input_square = manim.Square()
        input_grid = manim.NumberPlane(
            x_length=2,
            y_length=2,
            # x_range=(-1, 5, 1),
            # y_range
        )
        # self.play(
        #     # manim.ApplyMatrix(matrix, input_square),
        #     # manim.ApplyPointwiseFunction(relu, manim.Square()),
        #     # manim.ApplyMatrix(matrix, manim.Vector([1,0])),
        #     # manim.ApplyMatrix(matrix, manim.Vector([0,1]))
        #     )
        self.wait(0.5)
        self.play(
            manim.ApplyMatrix(matrix, input_grid),
            manim.ApplyMatrix(matrix, violation_vector)
        )
        input_grid.prepare_for_nonlinear_transform()
        self.play(
            manim.ApplyPointwiseFunction(relu, input_grid),
            manim.ApplyPointwiseFunction(relu, violation_vector),
        )
        self.play(
            manim.ApplyMatrix(matrix, input_grid),
            manim.ApplyMatrix(matrix, violation_vector)
        )
        
        # self.play(
        #     manim.Apply(matrix, input_square)
        # )
        


# ApplyMatrixExample().construct()
# NonlinearTransformSeries()