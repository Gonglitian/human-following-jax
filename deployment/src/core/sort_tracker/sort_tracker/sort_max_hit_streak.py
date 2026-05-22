from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two batches of bounding boxes in the form [x1, y1, x2, y2].
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) *
              (bb_test[..., 3] - bb_test[..., 1]) +
              (bb_gt[..., 2] - bb_gt[..., 0]) *
              (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    """
    Converts a bounding box in the form [x1, y1, x2, y2] to [x, y, s, r].
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h    # scale is area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Converts a bounding box in the form [x, y, s, r] to [x1, y1, x2, y2].
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bounding boxes.
    Using a constant velocity model (no explicit acceleration states).
    State vector: [x, y, s, r, dx, dy, ds, dr]
    Measurements: [x, y, s, r]
    """
    def __init__(self, bbox, tracker_id):
        """
        Initializes a tracker using initial bounding box and assigned ID.
        """
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 0.15  # Time step

        # State Transition Matrix F
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1]
        ])

        # Measurement Matrix H
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0, 0],  # s
            [0, 0, 0, 1, 0, 0, 0, 0]   # r
        ])

        # Measurement Noise Covariance R
        self.kf.R = np.eye(4)
        self.kf.R[0:2, 0:2] *= 10.0
        self.kf.R[2:, 2:] *= 10.0

        # Process Noise Covariance Q
        self.kf.Q = np.eye(8)
        self.kf.Q[0:4, 0:4] *= 2.0
        self.kf.Q[4:8, 4:8] *= 1.0

        # Initial State Covariance Matrix P
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # Initialize State Vector x
        self.kf.x = np.zeros((8, 1))
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = tracker_id
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.max_hit_streak = 0  # Track the maximum hit streak achieved
        self.age = 0
        self.total_timesteps = 0
        self.detected_timesteps = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bounding box.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        # Update max_hit_streak if current hit_streak is a new maximum
        if self.hit_streak > self.max_hit_streak:
            self.max_hit_streak = self.hit_streak
        self.detected_timesteps += 1
        self.total_timesteps += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1
        self.total_timesteps += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def get_detect_ratio(self):
        if self.total_timesteps == 0:
            return 0
        return self.detected_timesteps / (self.total_timesteps+0.000001)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked objects (both represented as bounding boxes).
    Returns:
        matches, unmatched_detections, unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:,1]:
            unmatched_trackers.append(t)

    # Filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if len(matches) == 0:
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.next_id = 0
        self.available_ids = []

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1, y1, x2, y2, score], ...]
        Requires: this method must be called once for each frame even with empty detections
                  (use np.empty((0, 5)) for frames without detections).
        Returns:
          A similar array, where the last column is the object ID.
        """
        self.frame_count += 1
        # Get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :4] = pos
            trks[t, 4] = trk.id
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.available_ids.append(self.trackers[t].id)
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks[:, :4], self.iou_threshold)

        # Update matched trackers with assigned detections
        for m in matched:
            t = m[1]
            d = m[0]
            self.trackers[t].update(dets[d, :])

        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            if self.available_ids:
                tracker_id = min(self.available_ids)
                self.available_ids.remove(tracker_id)
            else:
                tracker_id = self.next_id
                self.next_id += 1
            trk = KalmanBoxTracker(dets[i, :], tracker_id)
            self.trackers.append(trk)

        # Filter trackers using max_hit_streak instead of current hit_streak
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            d = trk.get_state()[0]
            if (trk.time_since_update < self.max_age) and \
               ((trk.get_detect_ratio() > 0.9 or trk.hit_streak >= self.min_hits) or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.available_ids.append(trk.id)
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def update_with_valid_unmatched_trks(self, dets=np.empty((0, 5))):
        """
        Similar to the `update` method, but also ensures that unmatched trackers 
        which still meet the validity criteria (i.e., haven't exceeded max_age and 
        meet the detection ratio/max_hit_streak conditions) are included in the returned 
        results along with the matched ones.
        """

        self.frame_count += 1
        # Get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :4] = pos
            trks[t, 4] = trk.id
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Remove trackers with NaN predictions
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.available_ids.append(self.trackers[t].id)
            self.trackers.pop(t)

        # Associate detections with trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks[:, :4], self.iou_threshold
        )

        # Update matched trackers with assigned detections
        for m in matched:
            t = m[1]
            d = m[0]
            self.trackers[t].update(dets[d, :])

        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            if self.available_ids:
                tracker_id = min(self.available_ids)
                self.available_ids.remove(tracker_id)
            else:
                tracker_id = self.next_id
                self.next_id += 1
            trk = KalmanBoxTracker(dets[i, :], tracker_id)
            self.trackers.append(trk)

        # Include unmatched trackers if they meet conditions, using max_hit_streak
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            d = trk.get_state()[0]
            if (trk.time_since_update < self.max_age) and \
               ((trk.get_detect_ratio() > 0.9 or trk.max_hit_streak >= self.min_hits) or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            # Remove dead tracklets
            if trk.time_since_update > self.max_age:
                self.available_ids.append(trk.id)
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display',
                        help='Display online tracker output (slow) [False]', action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args
