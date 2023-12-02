class Problem:
    """
    Following the same pattern as pygmo, we define a Problem class that will be used by the optimizer.
    """

    def __init__(self) -> None:
        pass

    def fitness(self, x):
        """
        The fitness function. This is the function that will be optimized.
        """
        raise NotImplementedError

    def get_bounds(self):
        """
        The bounds of the problem. This is used to define the search space.
        """
        raise NotImplementedError
