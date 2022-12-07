from unittest.mock import DEFAULT
from manim import *
import manim
import torch

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
            [1, -1],
            [1, 1],
        ]
        self.add(
            manim.NumberPlane(),
            manim.Square(side_length=2*1.3, color=manim.utils.color.RED),
            # manim.Vector([1,1]),
            # manim.Vector([-1,1]),
        )
        violation_vector = manim.Vector([-3/10, -1], color=manim.utils.color.PINK)
        # input_square = manim.Square()
        input_grid = manim.NumberPlane(
            x_length=2,
            y_length=2,
            background_line_style={
                "stroke_color": manim.utils.color.TEAL
            }
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

class BasicSequentialModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2,20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2,2),
        )
    def forward(self, x):
        return self.layers(x)
        

def weird_polar(points):
    x, y = points[0], points[1]
    points = np.array([x * np.sqrt(1 - y * y/2), y * np.sqrt(1-x * x/2), 0])
    points = np.tan(points) / np.tan(1)
    return points


def train_model(X, y, num_epochs=20):
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    learning_rate = 0.001
    layers = BasicSequentialModel()
    layers.train()
    optimizer = torch.optim.Adam(layers.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        y_pred = layers(X)
        loss_func = torch.nn.MSELoss()
        loss = loss_func(y, y_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch {epoch}: loss {loss}" )
    return layers

class NonlinearNoApproxExample(manim.ZoomedScene):
    def construct(self):
        circle = manim.Circle(color=manim.utils.color.RED)
        thinner_circle = manim.Circle(color=manim.utils.color.RED)
        thinner_circle.set_stroke(width=1)
        self.add(
            manim.NumberPlane(),
            circle
            # manim.Square(side_length=2*1.3, color=manim.utils.color.RED),
            # manim.Vector([1,1]),
            # manim.Vector([-1,1]),
        )
        # violation_vector = manim.Vector([1, -3/10], color=manim.utils.color.PINK)
        # TODO - get this programatically
        violation_coords = [
            375404693077277226889351051194570845062176041542286876927382358940864192835634050736150645301754565570 /
            375887829275949276828963180752582952524793046740097912556650658071049565356526290125072559512401061289,
            -375880577512710485513406111182162340662741698738972977513757779076591962585076227602096877692486666753 /
            375887829275949276828963180752582952524793046740097912556650658071049565356526290125072559512401061289,
            0
        ]
        violation_vector = manim.Point(
            violation_coords,
            color=manim.utils.color.PINK,
        ).set_stroke_width(10)
        violation_vector2 = manim.Point(
            violation_coords,
            color=manim.utils.color.RED,
        ).set_stroke_width(10)
        # input_square = manim.Square()
        input_grid = manim.NumberPlane(
            x_length=2,
            y_length=2,
            background_line_style={
                "stroke_color": manim.utils.color.TEAL
            }
            # x_range=(-1, 5, 1),
            # y_range
        )
        input_grid2 = manim.NumberPlane(
            x_length=2,
            y_length=2,
            background_line_style={
                "stroke_color": manim.utils.color.GREEN
            }
        )
        # self.play(
        #     # manim.ApplyMatrix(matrix, input_square),
        #     # manim.ApplyPointwiseFunction(relu, manim.Square()),
        #     # manim.ApplyMatrix(matrix, manim.Vector([1,0])),
        #     # manim.ApplyMatrix(matrix, manim.Vector([0,1]))
        #     )
        input_grid.prepare_for_nonlinear_transform()
        input_grid2.prepare_for_nonlinear_transform()
        model = BasicSequentialModel()
        model.load_state_dict(torch.load("network.tm"))
        def run_model(points):
            with torch.no_grad():
                x, y = np.array(model(torch.FloatTensor(points)[:2]))
                return np.array([x,y,0])
        self.play(
            manim.ApplyPointwiseFunction(run_model, input_grid2 ),
            manim.ApplyPointwiseFunction(weird_polar, input_grid),
            manim.ApplyPointwiseFunction(run_model, violation_vector2),
            manim.ApplyPointwiseFunction(weird_polar, violation_vector),
        )
        # wait and zoom in
        self.wait()
        self.play(
            self.camera.frame.animate.move_to(run_model(violation_coords)).set(width=(2)),
            manim.ReplacementTransform(input_grid, input_grid.copy().set_stroke(width=1)),
            manim.ReplacementTransform(input_grid2, input_grid2.copy().set_stroke(width=1)),
            manim.ReplacementTransform(circle, thinner_circle),
            manim.ReplacementTransform(violation_vector, violation_vector),
            manim.ReplacementTransform(violation_vector2, violation_vector2)
            )
        self.wait(1)
        
        # self.play(
        #     manim.Apply(matrix, input_square)
        # )

# ApplyMatrixExample().construct()
# NonlinearTransformSeries()

if __name__=="__main__":
    num_points = 5000
    sample_points = np.array(
        list(zip(np.random.uniform(-1,1,num_points), np.random.uniform(-1,1,num_points)))
    )
    mapped_to = np.apply_along_axis(weird_polar, 1, sample_points)[:, :2]
    print(mapped_to)
    network = train_model(sample_points, mapped_to, num_epochs=2000)
    print(network)
    torch.save(network.state_dict(), "network.tm")