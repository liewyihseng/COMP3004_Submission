import numpy as np
from gym import ObservationWrapper
from gym.spaces import Box
import cv2

'''
Converts the image observation from RGB to Red channel

Input Parameter:
    ObservationWrapper : A predefined class from gym
    
Return:
    observation: The extracted red channel
'''
class RObservation(ObservationWrapper):
    def __init__(self, env, keep_dim=False):
        super(RObservation, self).__init__(env)
        self.keep_dim = keep_dim

        assert (
                len(env.observation_space.shape) == 3
                and env.observation_space.shape[-1] == 3
        )

        obs_shape = self.observation_space.shape[:2]
        if self.keep_dim:
            self.observation_space = Box(
                low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
            )
        else:
            self.observation_space = Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )

    def observation(self, observation):
        observation = observation[:, :, 0]
        # cv2.imshow("Red Channel", observation)
        if self.keep_dim:
            observation = np.expand_dims(observation, -1)
        return observation


'''
Converts the image observation from RGB to Green channel

Input Parameter:
    ObservationWrapper : A predefined class from gym

Return:
    observation: The extracted green channel
'''
class GObservation(ObservationWrapper):
    def __init__(self, env, keep_dim=False):

        super(GObservation, self).__init__(env)
        self.keep_dim = keep_dim

        assert (
                len(env.observation_space.shape) == 3
                and env.observation_space.shape[-1] == 3
        )

        obs_shape = self.observation_space.shape[:2]
        if self.keep_dim:
            self.observation_space = Box(
                low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
            )
        else:
            self.observation_space = Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )

    def observation(self, observation):
        observation = observation[:, :, 1]
        # cv2.imshow("Green Channel", observation)
        if self.keep_dim:
            observation = np.expand_dims(observation, -1)
        return observation


'''
Converts the image observation from RGB to Blue channel

Input Parameter:
    ObservationWrapper : A predefined class from gym

Return:
    observation: The extracted blue channel
'''
class BObservation(ObservationWrapper):
    def __init__(self, env, keep_dim=False):
        super(BObservation, self).__init__(env)
        self.keep_dim = keep_dim

        assert (
                len(env.observation_space.shape) == 3
                and env.observation_space.shape[-1] == 3
        )

        obs_shape = self.observation_space.shape[:2]
        if self.keep_dim:
            self.observation_space = Box(
                low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
            )
        else:
            self.observation_space = Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )

    def observation(self, observation):
        observation = observation[:, :, 2]
        # cv2.imshow("Blue Channel", observation)
        if self.keep_dim:
            observation = np.expand_dims(observation, -1)
        return observation
