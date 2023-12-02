class BaseEAOptimizer:
    def __init__(self) -> None:
        pass

    def run(self):
        raise NotImplementedError

    def get_best_solution(self):
        raise NotImplementedError

    def get_best_fitness(self):
        raise NotImplementedError
