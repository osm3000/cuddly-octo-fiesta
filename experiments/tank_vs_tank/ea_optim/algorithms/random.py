import numpy as np
import uuid
import json
import datetime
import os
from ..problem import Problem


class RandomOptimizer:
    """
    A simple Random Optimizer. It will generate a population of random solution, try all of them, and return the best one.
    """

    def __init__(self, population_size: int, problem: Problem) -> None:
        self.population_size = population_size
        self.problem = problem
        self.best_solution = None
        self.best_fitness = None
        self.verbose = False

        # Logistics for saving the results
        folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results_folder = f"./results/{folder_name}"
        # Create the folder
        os.makedirs(self.results_folder)

    def run(self):
        """
        Run the optimization process.
        """
        # Generate the population
        population = []
        for _ in range(self.population_size):
            population.append(
                np.random.uniform(
                    self.problem.get_bounds()[0], self.problem.get_bounds()[1]
                ).tolist()
            )

        # Evaluate the population
        fitnesses = []
        for solution_idx, solution in enumerate(population):
            fitnesses.append(self.problem.fitness(solution))

            if self.verbose:
                if solution_idx % 100 == 0:
                    # Get best fitness so far
                    best_fitness_so_far = np.min(fitnesses)
                    print(
                        f"Solution {solution_idx + 1}/{len(population)} - Best fitness so far: {best_fitness_so_far}"
                    )

                    # Save a checkpoint
                    current_best_solution = population[np.argmin(fitnesses)]

                    # Data to save
                    data = {
                        "best_fitness_so_far": best_fitness_so_far,
                        "current_best_solution": current_best_solution,
                    }

                    # Save the current population to a file
                    with open(
                        f"{self.results_folder}/best_solution_{solution_idx}.json", "w"
                    ) as f:
                        json.dump(data, f, indent=4)

        # Find the best solution - the one with the lowest fitness, since we are minimizing
        self.best_solution = population[np.argmin(fitnesses)]
        self.best_fitness = np.min(fitnesses)

        return self.best_solution, self.best_fitness

    def get_best_solution(self):
        """
        Return the best solution found.
        """
        return self.best_solution

    def get_best_fitness(self):
        """
        Return the fitness of the best solution found.
        """
        return self.best_fitness

    def set_verbose(self, verbose: bool):
        """
        Set the verbosity of the optimizer.
        """
        self.verbose = verbose


class SimpleEvolutionaryStrategies(RandomOptimizer):
    def __init__(
        self, population_size: int, problem: Problem, elite_size: int, nb_of_gen: int
    ) -> None:
        super().__init__(population_size, problem)
        self.elite_size = elite_size
        self.best_solution = None
        self.best_fitness = None
        self.verbose = True
        self.nb_of_gen = nb_of_gen

        self.population: np.ndarray = None
        self.elites: np.ndarray = None
        self.population_fitnesses: np.ndarray = None
        self.elites_fitnesses: np.ndarray = None

        self._sigmas: np.ndarray = None
        self._means: np.ndarray = None

        self.generation_nb = 0

        self.algo_name = "SimpleEvolutionaryStrategies"

    def _init_population(self):
        """
        Generate the initial population.
        """
        self.population = []
        for _ in range(self.population_size):
            self.population.append(
                np.random.uniform(
                    self.problem.get_bounds()[0],
                    self.problem.get_bounds()[
                        1
                    ],  # TODO: Review this logic again, it's not clear
                ).tolist()
            )

        # Convert to numpy array
        self.population = np.array(self.population)

    def setup(self):
        """
        Setup the optimizer.
        """
        self._init_population()

        ########################################################
        # Initialize the means and sigmas
        ############################
        # For each dimension of the population, calculate the mean and sigma
        self._means = np.mean(self.population, axis=0)
        self._sigmas = np.std(self.population, axis=0)

    def step(self):
        if self.generation_nb == 0:
            self.setup()

            # Evaluate the population
            fitnesses = []
            for solution in self.population:
                fitnesses.append(self.problem.fitness(solution))

            sorted_fitnesses_indices = np.argsort(
                fitnesses
            )  # Ascending order: lowest fitness first
            # Get the elites
            self.elites = self.population[sorted_fitnesses_indices[: self.elite_size]]
            self.elites_fitnesses = np.array(fitnesses)[
                sorted_fitnesses_indices[: self.elite_size]
            ]

        else:
            # Estimate the means and sigmas
            self._means = np.mean(self.elites, axis=0)
            self._sigmas = np.std(self.elites, axis=0)

            # Generate a single offspring --> Does it have to be a single offspring only?
            new_offspring = np.random.normal(self._means, self._sigmas)

            # print("new_offspring: ", new_offspring)
            # print("new_offspring.shape: ", new_offspring.shape)

            # Evaluate the offspring
            offspring_fitness = self.problem.fitness(new_offspring.tolist()[0])

            # Update the elites
            # Check if the offspring is better than the worst elite
            if offspring_fitness < np.max(self.elites_fitnesses):
                # Replace the worst elite with the offspring
                worst_elite_idx = np.argmax(self.elites_fitnesses)
                self.elites[worst_elite_idx] = new_offspring
                self.elites_fitnesses[worst_elite_idx] = offspring_fitness

        # Extract the best solution and fitness
        best_solution_idx = np.argmin(self.elites_fitnesses)
        self.best_solution = self.elites[best_solution_idx].tolist()[0]
        self.best_fitness = self.elites_fitnesses[best_solution_idx].tolist()[0][0]

        # Save the best solution and fitness
        data = {
            "best_fitness_so_far": self.best_fitness,
            "current_best_solution": self.best_solution,
            "algorithm": self.algo_name,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }

        # Save the current population to a file
        # print(data)
        with open(
            f"{self.results_folder}/best_solution_{self.generation_nb}.json", "w"
        ) as f:
            json.dump(data, f)
        # Increment the generation counter
        self.generation_nb += 1

        if self.verbose:
            print(
                f"Generation: {self.generation_nb} - Best fitness: {self.best_fitness}"
            )

        return self.best_solution, self.best_fitness

    def run(self):
        for _ in range(self.nb_of_gen):
            self.step()

    def get_name(self):
        return self.algo_name
