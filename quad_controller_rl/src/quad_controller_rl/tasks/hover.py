"""Hover task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Hover(BaseTask):
    """Simple task where the goal is to lift off the ground and hover at a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([-cube_size / 2, -cube_size / 2, 0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([cube_size / 2, cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        #print("Hover(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Hover(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 5.0  # secs
        self.max_error_position = 8.0
        self.target_position = np.array([0.0, 0.0, 10.0])  # Make the agent hover at 10 units above ground
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # zero velocity when hovering
        self.position_weight = 0.6
        self.orientation_weight = 0.0
        self.velocity_weight = 0.4

        # self.reset()

    def reset(self):
        self.last_position = None
        self.last_timestamp = None

        p = self.target_position + np.random.normal(0.5, 0.1, size=3)

        # Return initial condition
        return Pose(
                position=Point(*p),  # drop off from target height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )


    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        # Calculate velocity
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03)

        state = np.concatenate([position, orientation, velocity])
        self.last_position = position
        self.last_timestamp = timestamp

        # Compute reward / penalty and check if this episode is complete
        done = False
        error_position = np.linalg.norm(self.target_position - position)
        error_orientation = np.linalg.norm(self.target_orientation - orientation)
        error_velocity = np.linalg.norm(self.target_velocity - velocity)
        reward = - (self.position_weight * error_position +
                    self.orientation_weight * error_orientation +
                    self.velocity_weight * error_velocity)

        if error_position > self.max_error_position:
            reward -= 50.0
            done = True
        elif timestamp > self.max_duration:
            reward += 50.0
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state[:7], reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done

    def __repr__(self):
        return self.__class__.__name__

