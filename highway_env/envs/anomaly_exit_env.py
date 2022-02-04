import numpy as np
from gym.envs.registration import register
from typing import Tuple
from sqlalchemy import false

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.utils import near_split
from highway_env.vehicle.behavior import AggressiveVehicle, DefensiveVehicle


class AnomalyExitEnv(AbstractEnv):

    """
    A highway environment that spawns either (1) an anomaly scenario or (2) an exit scenario

    (1) The ego-vehicle is driving on a highway behind other aggressive and defensive vehicles. 
        A car is crashed ahead in the left lane of the highway, non-ego cars merge right randomly
        to avoid the crashed vehicle, and the ego-vehicle needs to learn to similarly avoid the 
        crashed car

    (2) The ego-vehicle is driving on a highway behind other aggressive and defensive vehicles. 
        An exit ramp is approaching in the right-most lane and some non-ego cars merge right to 
        exit. The ego-car needs to stay in the left-most lane without crashing to receive the 
        largest reward.

    The goal of this environment is to learn a policy that learns when an anomaly has occured based
    on the behavior changes of non-ego cars on the highway. Thus, in the anomaly scenario, the ego
    car should learn to merge right to avoid the crash, and in the exit scenario, the ego vehicle 
    should learn to stay in the left-most lane since no crash has occured.
    """

    @classmethod
    def default_config(cls) -> dict:
        """
        Set the default configuration of this environment.
        TODO: Figure out which settings are necessary here and which aren't.
        """

        cfg = super().default_config()
        cfg.update({
            # TODO: Change default rewards in config
            # TODO: Choose observation space of ego vehicle
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
            "observations": {
                "observation1": {
                    "type": "Kinematics",
                    "vehicles_count": 1,
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "absolute": False,
                    "include_human_disp": False
                },
                "observation2": {
                    "type": "LidarObservation",
                    "cells": 180,
                    "normalize": False
                }
            },
            "action": {                         # action space is discrete 
                "type": "DiscreteMetaAction",
                "longitudinal": True,           # ego car chooses how to speed up or slow down
                "lateral": True                 # ego car chooses when to switch lanes and how
            },
            "anomaly_probability": 0.5,         # probability of two-lane anomaly scenario occuring
            "vehicles_count": 7,                # number of moving vehicles including ego-vehicle
            "initial_lane_id": 0,               # initial spawn lane of ego vehicle. TODO: Make it random?
            "screen_height": 500,               # rendering screen height
            "screen_width": 1500,               # rendering screen width
            "simulation_frequency": 25,         # [Hz]
            "policy_frequency": 5               # [Hz]
                    })
        return cfg


    def _reward(self, action: int) -> float:
        """
        TODO: Set reward of ego-vehicle. (IGNORE THE BELOW REWARDS FOR NOW)

        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """

        action_reward = {0: self.config["lane_change_reward"],
                         1: 0,
                         2: self.config["lane_change_reward"],
                         3: 0,
                         4: 0}
        reward = 0
        # reward = self.config["collision_reward"] * self.vehicle.crashed \
        #     + self.config["right_lane_reward"] * self.vehicle.lane_index[2] / 1 \
        #     + self.config["high_speed_reward"] * self.vehicle.speed_index / (self.vehicle.SPEED_COUNT - 1)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
                reward += self.config["merging_speed_reward"] * \
                          (vehicle.target_speed - vehicle.speed) / vehicle.target_speed

        return utils.lmap(action_reward[action] + reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])


    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the ego-vehicle has passed the end of the exit."""

        return self.vehicle.crashed or self.vehicle.position[0] > sum(self.ends)


    def _reset(self) -> None:
        """Called to reset environment and choose between the anomaly or exit scenario."""

        # Lengths of sections of highway
        # TODO: change lengths of sections of highway randomly?
        self.ends = [10., 80., 40., 80., 40., 80.]  # [ego spawn area, before, start split, exit, diverge, after]

        # Choose between anomaly and exit scenario for _make_road()
        self.anomaly_scenario = self.np_random.rand() <= self.config["anomaly_probability"]

        # Make scenario
        self._make_road()
        self._make_vehicles()


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Called every timestep to step the environment forward."""

        obs, reward, done, info = super().step(action)  # step env
        self._clear_vehicles()                          # remove non-ego cars that have passed the end of the env
        # TODO: Force non-ego cars to merge right in anomaly scenario when close to crashed car
        return obs, reward, done, info


    def _make_road(self) -> None:
        """
        Make a road composed of either a straight highway or a straight highway with an exit lane.
        """

        net = RoadNetwork()                         # graph to be filled for wither scenario

        ends = self.ends                            # access highway section lengths

        # Lane design variables
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE # paint of lanes
        y = [0, StraightLane.DEFAULT_WIDTH]                                 # y-location of highway lanes

        ###########################
        # Create anomaly scenario #
        ###########################
        if self.anomaly_scenario:
            line_type = [[c, s], [n, c]]    # left lane and right lane paint styles for anomaly 2-lane scenario

            # Highway lanes
            # First iteration creates left lane and second iteration creates right lane
            for i in range(2):
                net.add_lane("s", "a", StraightLane([0, y[i]], [ends[0], y[i]], line_types=line_type[i]))
                net.add_lane("a", "b", StraightLane([ends[0], y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
                net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type[i]))
                net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends[:4]), y[i]], line_types=line_type[i]))
                net.add_lane("d", "e", StraightLane([sum(ends[:4]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

            # Generate road from graph
            road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
            
        ###############################
        # Create normal exit scenario #
        ###############################
        else:
            line_type = [[c, s], [n, c]]        # left lane and right lane paint styles for two lanes outside of the exit section
            line_type_merge = [[c, s], [n, s]]  # left lane and right lane paint styles for two lanes inside of the exit section
            
            # Highway lanes
            # First iteration creates left lane and second iteration creates right lane
            for i in range(2):
                net.add_lane("s", "a", StraightLane([0, y[i]], [ends[0], y[i]], line_types=line_type[i]))
                net.add_lane("a", "b", StraightLane([ends[0], y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
                net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
                net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends[:4]), y[i]], line_types=line_type_merge[i]))
                net.add_lane("d", "e", StraightLane([sum(ends[:4]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

            # Diverging exit lane
            lcd = StraightLane([sum(ends[:3]), 2*StraightLane.DEFAULT_WIDTH], [sum(ends[:4]), 2*StraightLane.DEFAULT_WIDTH],    # linear exit third lane
                                line_types=[s, c])
            ldj = StraightLane(lcd.position(ends[3], 0.), lcd.position(sum(ends[3:5]), StraightLane.DEFAULT_WIDTH),             # diagonal section diverging from two lanes
                                line_types=[c, c])
            ljk = StraightLane(ldj.position(ends[4], 0.), ldj.position(ends[4], 0.) + [ends[5], 0.],                            # linear exit completely disconnected from two lanes
                                line_types=[c, c])
            lbc = StraightLane([sum(ends[:2]), y[1]], lcd.position(0., 0.), line_types=[n, c])                                  # diagonal section generating third lane from two lanes

            # Add exit sections to graph
            net.add_lane("c", "d", lcd)
            net.add_lane("d", "j", ldj)
            net.add_lane("j", "k", ljk)
            net.add_lane("b", "c", lbc)

            # Generate road from graph
            road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        
        # Set the road of the environment for this scenario
        self.road = road


    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.9) -> None:
        """Helper function to spawn car at specific location with some probability and behavior"""

        # chance that car will not be spawned
        if self.np_random.rand() > spawn_probability:
            return

        # randomly decide the starting lane
        route = self.np_random.choice(range(2), size=1, replace=True)

        # decide behavior of car to be spawned
        # vehicle_type = DefensiveVehicle if self.np_random.rand() > 0.5 else AggressiveVehicle
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])

        # create car on on of the two lanes at given position with some deviation on speed
        vehicle = vehicle_type.make_on_lane(self.road, ("a", "b", route[0]),
                                            longitudinal=longitudinal,
                                            speed=25+self.np_random.randn()*speed_deviation)

        # if the new car is too close to any existing cars, don't add it
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 10:
                return

        # randomize delta in IDM (how close do we want the actual velocity to be with desired velocity)
        vehicle.randomize_behavior()

        # Add car to env's vehicles list
        self.road.vehicles.append(vehicle)
        return vehicle


    def _clear_vehicles(self) -> None:
        """Remove non-ego vehicles that are at the end of the env."""

        is_leaving = lambda vehicle: vehicle.position[0] > sum(self.ends)

        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle == self.vehicle or not is_leaving(vehicle)]


    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway, as well as an ego-vehicle.
        """

        # generate ego-vehicle
        controlled_vehicle = self.action_type.vehicle_class(self.road, [0,0]).make_on_lane(self.road, ("s", "a", self.config["initial_lane_id"]),
                                        longitudinal=5,
                                        speed=30)
        self.vehicle = controlled_vehicle
        self.road.vehicles.append(controlled_vehicle)

        # set noisy initial positions of non-ego vehicles before exit section ends
        non_ego_spawn_positions = np.linspace(self.ends[0], sum(self.ends[:4]), self.config["vehicles_count"])
        non_ego_spawn_positions += np.random.normal(size = non_ego_spawn_positions.shape)

        # spawn non-ego vehicles 
        for t in range(self.config["vehicles_count"] - 1):
            curr_vehicle = self._spawn_vehicle(non_ego_spawn_positions[t])  # current lateral spawn position

            # force all spawned cars in anomaly scenario to merge into right lane to avoid crashed car in left lane
            if curr_vehicle and self.anomaly_scenario:
                curr_vehicle.plan_route_to("e")
                graph = self.road.network.graph
                curr_vehicle.route = [(li[0], li[1], len(graph[li[0]][li[1]])-1) for li in curr_vehicle.route]

            # in normal exit scenario
            elif curr_vehicle and not self.anomaly_scenario:

                # setting route of non-ego car to exit the highway
                if curr_vehicle.position[0] < sum(self.ends[:3]) and self.np_random.rand() <= 0.5:
                    curr_vehicle.plan_route_to("k")
                    graph = self.road.network.graph
                    curr_vehicle.route = [(li[0], li[1], len(graph[li[0]][li[1]])-1) for li in curr_vehicle.route]

                # setting route of non-ego car to not switch lanes and go straight
                else:
                    curr_vehicle.enable_lane_change = False
                    curr_vehicle.plan_route_to("e")
                    graph = self.road.network.graph
                    curr_vehicle.route = [(li[0], li[1], curr_vehicle.route[0][2]) for li in curr_vehicle.route]

        # add crashed car to left lane if anomaly scenario
        if self.anomaly_scenario:
            vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])

            # spawn location of crashed car is in front of all other vehicles 
            vehicle = vehicle_type.make_on_lane(self.road, ("s", "a", 0),
                                                longitudinal=sum(self.ends[:4])+self.np_random.normal(sum(self.ends[4:])/2.),
                                                speed=0)
            vehicle.crashed = True  # sets car to be crashed and stopped on road
            self.road.vehicles.append(vehicle)


register(
    id='anomaly-exit-v0',
    entry_point='highway_env.envs:AnomalyExitEnv',
)
