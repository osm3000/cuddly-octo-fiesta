"""
This is a FAST DIY game engine, mainly for the battle-tank game.
"""
from copy import deepcopy
import numpy as np
import policy
import random
import torch
import json
from PIL import Image
from typing import Dict, List
import imageio

DRAW_FLAG = True

SENSOR_RANGE = 1
NB_OF_STATES = ((SENSOR_RANGE * 2) + 1) ** 2 - 1
NB_OF_SENSOR_DIMS = (SENSOR_RANGE * 2) + 1

DIR_ACTIONS = ["right", "left", "up", "down"]
ALL_ACTIONS = DIR_ACTIONS + ["fire"]

POLICY_NET = policy.Policy(
    nb_of_inputs=NB_OF_STATES,  #
    nb_of_outputs=len(ALL_ACTIONS),
    nb_of_hidden_neurons=2,
)

USE_RANDOM_POLICY = False


def map_reader(path):
    """
    This function reads the map file and returns a list of lists.
    """
    with open(path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


class MovingObject:
    def __init__(self, symbol, position: tuple) -> None:
        self.symbol = symbol

        self.current_x = position[0]
        self.current_y = position[1]
        self.next_x = position[0]
        self.next_y = position[1]

        self._command = None

        # Object state
        self.is_alive = True
        self.health = 100
        self.speed = 1
        self._direction = None  # 0: up, 1: right, 2: down, 3: left

    def on_update(self):
        self.current_x = self.next_x
        self.current_y = self.next_y

    @property
    def command(self):
        return self._command

    @command.setter
    def command(self, value):
        if value in ALL_ACTIONS:
            self._command = value
        else:
            raise ValueError("Invalid command")

    @property
    def direction(self):
        return self._direction

    def __repr__(self) -> str:
        return f"""
        Position: ({self.current_x}, {self.current_y})
        Direction: {self._direction}
        Command: {self._command}
        Next position: ({self.next_x}, {self.next_y})
        """


class Tank(MovingObject):
    def __init__(self, name: str, position: tuple) -> None:
        super().__init__(symbol="@", position=position)
        self.name = name

        # Initial command
        self._command = "right"

        # Tank state
        self.health = 100

        # Tank turret state
        self._turret_direction = "left"

        # Weapon state
        self.weapon = "cannon"
        self.weapon_ammo = 10

        self._nb_of_downs = 0

    def reason(self, sensor_data: dict):
        global POLICY_NET, ALL_ACTIONS, DIR_ACTIONS, USE_RANDOM_POLICY

        numerical_sensor_data = sensor_data["sensor_data"]

        if USE_RANDOM_POLICY:
            POLICY_NET.set_weight(np.random.rand(POLICY_NET.nb_of_param).tolist())

        decision_id = POLICY_NET.forward(torch.Tensor(numerical_sensor_data))

        self._command = ALL_ACTIONS[decision_id]

        # Demo logic
        # if sensor_data["frontal_wall"]:
        # if self._command == "right":
        #     self._command = "down"
        #     self._nb_of_downs += 1
        # elif self._command == "down":
        #     self._command = "left"
        # elif self._command == "left":
        #     self._command = "up"
        # elif self._command == "up":
        #     self._command = "right"

        #     if self._command == "right":
        #         self._command = "down"
        #         self._nb_of_downs += 1

        # if self._nb_of_downs == 2:
        #     self._command = "fire"
        #     self._nb_of_downs = 0

        # Random logic
        # self._command = random.choice(ALL_ACTIONS)

        if self._command in DIR_ACTIONS:
            self._direction = self._command
            self._turret_direction = self._command

        if self._command == "fire":
            self.weapon_ammo -= 1

        assert self._command in ALL_ACTIONS

        # if self._command == "down":
        #     self._nb_of_downs += 1

        return self._command

    @property
    def turret_direction(self):
        return self._turret_direction

    def can_fire(self):
        return self.weapon_ammo > 0


class Bullet(MovingObject):
    def __init__(self, position, tank_name: str) -> None:
        super().__init__(symbol="*", position=position)
        self._tank_name = tank_name

    @property
    def tank_name(self):
        return self._tank_name


class GameEngine:
    def __init__(self):
        self.map = None
        self.walls = {}
        self.tanks: Dict[str, Tank] = {"player": None, "enemy": None}
        self.bullets = []

        self.frame_buffer = []

    def setup(self):
        self.map = map_reader("diy_map_0.map")
        self.walls = self._get_walls(self.map)
        self.empty_tiles = self._get_empty_tiles(self.map)

        # Create the tanks, and assign each to a random empty tile
        random.shuffle(self.empty_tiles)
        self.tanks["player"] = Tank(name="player", position=self.empty_tiles.pop())
        self.tanks["enemy"] = Tank(name="enemy", position=self.empty_tiles.pop())

    def _get_empty_tiles(self, map: list) -> List[tuple]:
        """
        This function returns a dictionary of empty tiles.
        """
        empty_tiles = []
        for y, line in enumerate(map):
            for x, symbol in enumerate(line):
                if symbol == " ":
                    empty_tiles.append((x, y))
        return empty_tiles

    def _move_element(self, element: MovingObject):
        if element.command == "right":
            element.next_x = element.current_x + 1
            element.next_y = element.current_y
        elif element.command == "left":
            element.next_x = element.current_x - 1
            element.next_y = element.current_y
        elif element.command == "down":
            element.next_x = element.current_x
            element.next_y = element.current_y + 1
        elif element.command == "up":
            element.next_x = element.current_x
            element.next_y = element.current_y - 1

    def collision_with_wall(self, element: MovingObject):
        if (element.next_x, element.next_y) in self.walls:
            return True
        return False

    def sensor_surroundings(self, element: MovingObject):
        """
        Get the value of the surrounding (wall, tank, bullet) tiles around the element
        """
        global SENSOR_RANGE, NB_OF_STATES, NB_OF_SENSOR_DIMS
        template_reading = np.zeros((NB_OF_SENSOR_DIMS, NB_OF_SENSOR_DIMS))
        if SENSOR_RANGE == 1:
            # Get the value of the surrounding tiles
            for y in range(-1, 2):
                for x in range(-1, 2):
                    # Read the walls
                    if (element.current_x + x, element.current_y + y) in self.walls:
                        template_reading[y + 1, x + 1] = 1
                    elif (  # Read the tanks
                        element.current_x + x,
                        element.current_y + y,
                    ) in self.tanks.values():
                        template_reading[y + 1, x + 1] = 2
                    elif (  # Read the bullets
                        element.current_x + x,
                        element.current_y + y,
                    ) in self.bullets:
                        template_reading[y + 1, x + 1] = 3
                    else:  # Read the empty tiles
                        template_reading[y + 1, x + 1] = 0

            # Flatten the template reading and convert to a list
            template_reading = template_reading.flatten().tolist()

            # Remove the center tile
            template_reading.pop(4)
        else:
            raise NotImplementedError("Only sensor range of 1 is implemented for now.")

        return template_reading

    def _convert_template_reading_to_numerical(self, sensor_reading: list):
        """
        Convert the template reading to a numerical vector
        """
        global SENSOR_RANGE, NB_OF_STATES
        # Convert the template reading to a numerical value
        num_vector = []

        for item in sensor_reading:
            if item == "#":
                num_vector.append(1)
            elif item == "@":
                num_vector.append(2)
            elif item == "*":
                num_vector.append(3)
            else:
                num_vector.append(0)

        return num_vector

    def sensor_frontal_wall_detection(self, element: MovingObject):
        # Check if there is a wall in front of the tank, from any direction
        if element.direction == "right":
            if (element.current_x + 1, element.current_y) in self.walls:
                return True
        elif element.direction == "left":
            if (element.current_x - 1, element.current_y) in self.walls:
                return True
        elif element.direction == "down":
            if (element.current_x, element.current_y + 1) in self.walls:
                return True
        elif element.direction == "up":
            if (element.current_x, element.current_y - 1) in self.walls:
                return True
        return False

    def on_update(self):
        ############################
        # Tank logic
        ############################
        # Check for movement command
        for tank_name, tank in self.tanks.items():
            # Get the sensor data
            # sensor_data = {
            #     "frontal_wall": self.sensor_frontal_wall_detection(tank),
            # }
            if not tank.is_alive:
                continue

            sensor_data = self.sensor_surroundings(tank)
            sensor_package = {
                "sensor_data": sensor_data,
            }

            # Get the command from the tank
            tank.reason(sensor_package)

            # Check for firing command
            if tank.command == "fire":
                if tank.can_fire():
                    self.bullets.append(
                        Bullet(
                            position=(tank.current_x, tank.current_y),
                            tank_name=tank_name,
                        )
                    )
                    # Set the direction of the bullet
                    self.bullets[-1].command = tank.turret_direction

            else:  # Check for movement command
                self._move_element(tank)

            # Check for collision with walls
            if self.collision_with_wall(tank):
                tank.next_x = tank.current_x
                tank.next_y = tank.current_y

        ############################
        # Bullet logic
        ############################
        # Move the bullets
        for bullet in self.bullets:
            self._move_element(bullet)

        # Check for collision with walls
        bullet_idx_to_delete = set()
        for bullet_idx, bullet in enumerate(self.bullets):
            if self.collision_with_wall(bullet):
                # bullet_idx_to_delete.append(bullet_idx)
                bullet_idx_to_delete.add(bullet_idx)

        # Check for collision with tanks
        for tank_name, tank in self.tanks.items():
            for bullet_idx, bullet in enumerate(self.bullets):
                if (tank_name != bullet.tank_name) and (
                    (bullet.current_x, bullet.current_y)
                    == (tank.current_x, tank.current_y)
                ):
                    # bullet_idx_to_delete.append(bullet_idx)
                    bullet_idx_to_delete.add(bullet_idx)
                    tank.is_alive = False

        # Delete the bullets that hit the wall
        # Looping in reverse order to avoid the index change problem: https://stackoverflow.com/a/11303234/2863057
        for index in sorted(list(bullet_idx_to_delete), reverse=True):
            del self.bullets[index]

        ############################
        # Update the game state
        ############################
        # Update the tanks
        for tank_name, tank in self.tanks.items():
            tank.on_update()

        # Update the bullets
        for bullet in self.bullets:
            bullet.on_update()

    def draw(self):
        current_state_map = deepcopy(self.map)
        for wall in self.walls:
            x, y = wall
            current_state_map[y] = (
                current_state_map[y][:x] + "#" + current_state_map[y][x + 1 :]
            )

        for tank in self.tanks.values():
            if not tank.is_alive:
                continue
            current_state_map[tank.current_y] = (
                current_state_map[tank.current_y][: tank.current_x]
                + tank.symbol
                + current_state_map[tank.current_y][tank.current_x + 1 :]
            )

        for bullet in self.bullets:
            current_state_map[bullet.current_y] = (
                current_state_map[bullet.current_y][: bullet.current_x]
                + bullet.symbol
                + current_state_map[bullet.current_y][bullet.current_x + 1 :]
            )

        self.frame_buffer.append(current_state_map)
        # for line in current_state_map:
        #     print(line)

    def _get_walls(self, map: list):
        """
        This function returns a dictionary of walls.
        """
        walls = {}
        for y, line in enumerate(map):
            for x, symbol in enumerate(line):
                if symbol == "#":
                    walls[(x, y)] = symbol
        return walls

    def collision_with_wall(self, element: MovingObject):
        if (element.next_x, element.next_y) in self.walls:
            return True
        return False

    def game_over_condition_0(self):
        """
        If any of the tanks is dead, then game over.
        """
        for tank in self.tanks.values():
            if not tank.is_alive:
                return True
        return False


# Define your mapping from strings to colors
color_dict = {
    "#": (200, 100, 200),
    "@": (255, 0, 0),
    "*": (0, 255, 0),
    " ": (255, 255, 255),
}


def array_to_image(array, color_dict):
    # Convert the array of strings to an array of RGB tuples
    # rgb_array = np.vectorize(color_dict.get)(array)  # [0]
    # print(rgb_array)
    # print(array)

    # Convert the array of RGB tuples to a PIL Image object
    # image = Image.fromarray(np.uint8(rgb_array))
    image = Image.fromarray(np.uint8(array))

    image = image.resize((array.shape[1] * 10, array.shape[0] * 10))

    return image


def worker(item):
    global DRAW_FLAG
    engine = GameEngine()
    engine.setup()

    for clk in range(100):
        engine.on_update()
        if DRAW_FLAG:
            engine.draw()
        # print("-//-" * 30)
        if engine.game_over_condition_0():
            break

    # Dump the frame buffer to a file
    if DRAW_FLAG:
        all_frames = []
        with open("frame_buffer.txt", "w") as f:
            for frame_idx, frame in enumerate(engine.frame_buffer):
                segmented_frame = []
                for line in frame:
                    f.write(line + "\n")
                    segmented_frame.append(list(line))
                f.write("-//-" * 30 + "\n")

                # Save the current frame to a file
                # with open(f"./images/frame_{frame_idx}.txt", "w") as f1:
                #     for line in frame:
                #         f1.write(line + "\n")

                numeric_segmented_frame = []
                for line in segmented_frame:
                    new_line = []
                    for item in line:
                        if item == "#":
                            new_line.append([0, 0, 0])
                        elif item == "@":
                            new_line.append([255, 0, 0])
                        elif item == "*":
                            new_line.append([0, 255, 0])
                        else:
                            new_line.append([255, 255, 255])
                    numeric_segmented_frame.append(new_line)
                # # Convert the current frame to an image
                # segmented_frame = np.array(segmented_frame)
                numeric_segmented_frame = np.array(numeric_segmented_frame).astype(
                    np.uint8
                )
                print(numeric_segmented_frame.shape)
                # print(segmented_frame.shape)
                image = array_to_image(numeric_segmented_frame, color_dict)
                # # Save the image to a file
                # image.save(f"./images/{frame_idx + 1}.png")

                # all_frames.append(numeric_segmented_frame)
                all_frames.append(image)
            # Save the frames to a gif file
            # print(np.array(all_frames).shape)
            imageio.mimsave("./movie.gif", all_frames)
            # # Convert the frame buffer to a numpy array
            # frame_buffer = np.array(engine.frame_buffer)
            # # Convert the numpy array a sequence of images
            # images = []
            # for frame in frame_buffer:
            #     images.append(np.array([list(line) for line in frame]))
            # Save the images to a file

    game_end_state = {
        "game_won": engine.game_over_condition_0(),
    }

    return game_end_state


def load_specific_weights(path):
    if ".npy" in path:
        return np.load(path).tolist()
    elif ".json" in path:
        with open(path, "r") as f:
            data = json.load(f)
        print(f'Fitness: {data["best_fitness_so_far"]}')
        return data["current_best_solution"]
    else:
        raise ValueError("Invalid file type")


def try_solution(path):
    global POLICY_NET

    # Load the weights
    POLICY_NET.set_weight(load_specific_weights(path))

    # POLICY_NET.set_weight(np.load("best_weights.npy").tolist())
    # Create and run the game
    worker(1)


def main():
    # worker(1)
    # try_solution("results/2023-11-22_08-35-04/best_solution_1800.json")
    try_solution("results/2023-11-22_10-23-03/best_solution_1700.json")
    # try_solution("best_weights_diy.npy")


if __name__ == "__main__":
    main()
