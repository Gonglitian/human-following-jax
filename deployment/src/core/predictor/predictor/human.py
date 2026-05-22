from collections import deque
import numpy as np

class Human:
    def __init__(self, id, x=15.0, y=15.0):
        self.id = id
        self.past_locations = deque(maxlen=5)
        self.past_timesteps = deque(maxlen=5)

        self.past_predictions = deque(maxlen=5)
        self.past_prediction_validity = deque(maxlen=5)

        self._x = x
        self._y = y


    def get_position(self):
        return np.array([self._x, self._y])

    def store_predictions(self, predictions, is_valid, inlcude_current=True):
        # stack the current position (2, ) in the front of the predictions (5, 2), to form (6, 2)
        if inlcude_current:
            predictions = np.vstack([self.get_position(), predictions])
        self.past_predictions.append(predictions)
        self.past_prediction_validity.append(is_valid)

    def set_attributes(self, x, y, t):
        self._x = x
        self._y = y
        self._t = t
        self.past_locations.append(np.array([x, y]))
        self.past_timesteps.append(t)
