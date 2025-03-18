import random
import numpy as np

from models import *

#
# Add your Filtering / Smoothing approach(es) here
#
class HMMFilter:
    def __init__(self, probs, tm, om, sm):
        self.__tm = tm
        self.__om = om
        self.__sm = sm
        self.__f = probs
        
    # sensorR is the sensor reading (index!), self._f is the probability distribution resulting from the filtering    
    def filter(self, sensorR : int) -> np.array :
        # Predict step: apply the transition model
        predicted_belief = np.dot(self.__tm.get_T().T, self.__f)

        # Update step: apply the observation model
        if sensorR is not None:
            observation_matrix = self.__om.get_o_reading(sensorR)
            updated_belief = observation_matrix @ predicted_belief
        else:
            updated_belief = predicted_belief

        # Normalize the belief
        #self.__f = updated_belief / np.sum(updated_belief)
        self.__f = updated_belief / np.sum(updated_belief) if np.sum(updated_belief) > 0 else np.ones_like(updated_belief) / len(updated_belief)
        return self.__f


class HMMSmoother:
    def __init__(self, tm, om, sm, window_size=5):
        self.__tm = tm
        self.__om = om
        self.__sm = sm
        
        # things for sliding window
        self.window_size = window_size
        self.window = []

    def update_window(self, new_reading):
        self.window.append(new_reading)
        if len(self.window) > self.window_size:
            self.window.pop(0)

    def smooth(self, f_k : np.array) -> np.array:
        """
        f_k is the filtered result (f_vector) for step k
        fb is the smoothed result (fb_vector)
        """
        # window has closed, no more data
        if not self.window:
            return f_k

        # Initialize the backward message
        b = np.ones(f_k.shape)
        
        # Iterate over the sensor readings in reverse order
        for sensorR in reversed(self.window):
            # Update the backward message
            observation_matrix = self.__om.get_o_reading(sensorR)
            b = np.dot(self.__tm.get_T(), observation_matrix @ b)
            b_sum = np.sum(b)
            if 0 <= b_sum <= 1e-3:
                b = np.full(f_k.shape, 1e-10) # Fill b with tiny, uniform probabilities (Prevents zero-probability states while maintaining small influence.)
            else:
                b /= b_sum  # Normalize

        # Compute the smoothed belief
        fb = f_k * b
        fb_sum = np.sum(fb)
        if fb_sum == 0:
            fb = np.ones(f_k.shape) / f_k.size  # Reset to uniform distribution if sum is zero
        else:
            fb /= fb_sum  # Normalize
        return fb