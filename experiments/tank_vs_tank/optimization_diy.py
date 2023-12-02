import logging
import multiprocessing
import signal
import pygmo as pg

import numpy as np

import ea_optim
import diy_game_engine

USE_PYGMO = False

logger = logging.getLogger("DIY_OPTIMIZATION")
logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler("diy_optimization.log")
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# NB_OF_RUNS = multiprocessing.cpu_count()
NB_OF_RUNS = 16


def run_parallel_workers():
    global NB_OF_RUNS
    all_outputs = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        all_outputs = pool.map(diy_game_engine.worker, range(NB_OF_RUNS))
        # arcade.exit()
    return all_outputs


def run_sequential_workers():
    global NB_OF_RUNS
    all_outputs = []
    for i in range(NB_OF_RUNS):
        all_outputs.append(diy_game_engine.worker(i))
    return all_outputs


class SimulationProblem(ea_optim.problem.Problem):
    def __init__(self):
        self.nb_dims = diy_game_engine.POLICY_NET.nb_of_param

    def fitness(self, x):
        global NB_OF_RUNS

        # Make sure Random policy is not used
        diy_game_engine.USE_RANDOM_POLICY = False
        # Set the weights of the policy
        diy_game_engine.POLICY_NET.set_weight(x)

        # Run the simulation 10 times, and average the total reward
        total_reward = []
        all_outputs = []
        # all_outputs = run_parallel_workers()
        all_outputs = run_sequential_workers()

        total_time = []
        for outputs in all_outputs:
            total_reward.append(outputs["game_won"])
        mean_total_reward = np.mean(total_reward)
        # print("mean_total_reward: ", mean_total_reward)
        # logger.info(
        #     "mean_total_reward: %s",
        #     mean_total_reward,
        # )

        # Since this is a maximization problem, but the framework is for minimization, return the negative of the mean total reward
        return [-mean_total_reward]

    def get_bounds(self):
        return ([-1] * self.nb_dims, [1] * self.nb_dims)

    # def gradient(self, x):
    #     return pygmo.estimate_gradient_h(lambda x: self.fitness(x), x)

    def get_nobj(self):
        return 1

    def get_name(self):
        return "BattleTankAI"


def main():
    if USE_PYGMO:
        prob = pg.problem(SimulationProblem())
        algo = pg.algorithm(pg.sea(gen=100_000))
        # algo = pg.algorithm(pg.cmaes(gen=1000))
        # print("algo: ", algo.verbose)
        algo.set_verbosity(level=1)
        pop = pg.population(prob, 1000)
        print(type(pop))
        print(type(algo))
        print("------------------" * 10)
        pop = algo.evolve(pop)

        print(pop.champion_f)  # best fitness

        with open("best_fitness.txt", "w") as f:
            f.write(str(pop.champion_f))

        # Save the best solution to a file
        np.save("best_weights.npy", pop.champion_x)
        print("Saved the best solution to best_weights.npy")

    else:
        prob = SimulationProblem()

        # prob = ea_optim.benchmarks.Rosenbrock()
        # # Set the optimization algorithm
        algo = ea_optim.algorithms.RandomOptimizer(100_000, prob)
        # algo = ea_optim.algorithms.SimpleEvolutionaryStrategies(
        #     population_size=1000,
        #     problem=prob,
        #     elite_size=100,
        #     nb_of_gen=10000,
        # )
        algo.set_verbose(True)

        try:
            algo.run()
        except KeyboardInterrupt:
            print("User interrupted the optimization process.")

        # Get the best solution and fitness
        best_solution = algo.get_best_solution()
        best_fitness = algo.get_best_fitness()

        print("best_fitness: ", best_fitness)

        # with open("best_fitness_diy.txt", "w") as f:
        #     f.write(str(best_fitness))

        # # Save the best solution to a file
        # np.save("best_weights_diy.npy", best_solution)
        # print("Saved the best solution to best_weights.npy")


if __name__ == "__main__":
    main()
