from ..problem import Problem


class Rosenbrock(Problem):
    def __init__(self) -> None:
        self.dims = 2

    def fitness(self, x):
        """
        The fitness function. This is the function that will be optimized.
        https://en.wikipedia.org/wiki/Rosenbrock_function
        """
        a = 1
        b = 100
        return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

    def get_bounds(self):
        """
        The bounds of the problem. This is used to define the search space.
        """
        return [-5, -5], [10, 10]
