"""
Copyright (c) 2016 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>
"""

import numpy as np
import sys # Required for floating point epsilon
import copy


class Timeseries(object):
    """A class which encapsulates a timeseries.

    This class is useful to ensapsulate a timeseries with a a set
    sampling rate. That is, rather than storing the entire timeseries,
    we store only the start, stop, and delta_t which defines the timeseries.
    It is important that any timeseries which reference this timeseries
    correctly interpolated to the timeseries associated with this timeseries.
    """

    def __init__(self, start, stop, dt):
        """Initialize a new timeseries.

        Initializes a new timeseries object. This function takes the first point,
        the final point, and the delta_t which defines the timeseries.
        :param start The time at t[0]
        :param stop The time at t[-1] (i.e., the last timepoint). This point is
        included in the timeseries (goes from star to stop, inclusive).
        :param dt The change in time associated with each timestep.
        """
        self.start = float(start)
        self.stop = float(stop)
        self.dt = float(dt)
        if self.start > self.stop:
            raise RuntimeError('Endpoint of timeseries is greater than start point')
        self.n = int(np.round((stop - start) / dt))
        self.stop = self.start + self.dt * self.n

    def timeseries(self):
        """Returns the complete timeseries associated with this element"""
        return np.arange(self.start, self.stop - self.dt/2, self.dt)

    def indices(self):
        """Returns the indices associated with this timeseries

        The indices are a simple array going from 0 to N-1 (where N is the
        length of the timeseries)
        """
        return np.arange(0, self.n)

    def find_index(self, value):
        """Finds the index associated with a certain value

        Note that this function merely returns the "closest" index to the
        given value. It is not guaranteed that the value is actually represented
        in the timeseries. If the value is not found, None is returned
        """
        if value < self.start or value > self.stop:
            return None
        index = int(round((value - self.start) / self.dt))
        return index

    def find_index_range(self, start, stop):
        """Returns a range of indices corresponding to a time window"""
        if start > np.floor(self.stop) - 1:
            return None
        if stop < np.ceil(self.start) + 1:
            return None
        t1 = self.find_index(start)
        if t1 is None:
            t1 = 0
        t2 = self.find_index(stop)
        if t2 is None:
            t2 = self.n
        return np.arange(t1, t2)

    def __len__(self):
        """Override the length of this timeseries"""
        return self.n

    def __getitem__(self, index):
        """Overrides the [] operator for this timeseries"""
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = self.n if index.stop is None else index.stop
            step = 1 if index.step is None else index.step
            return Timeseries(start * self.dt + self.start,
                              stop * self.dt + self.start,
                              self.dt * step)
        elif isinstance(index, list) or isinstance(index, tuple) or \
            (isinstance(index, np.ndarray) and index.ndim == 1):
            return [self[int(i)] for i in index]
        else:
            if index > self.n or index < -self.n:
                raise IndexError
            if index >= 0:
                return int(index) * self.dt + self.start
            else:
                return self.stop + self.dt * (int(index) + 1)

    def __add__(self, y):
        """Adds an offset to the timeseries"""
        return Timeseries(self.start + y, self.stop + y, self.dt)

    def __sub__(self, y):
        """Subtracts an offset from the timeseries"""
        return Timeseries(self.start - y, self.stop - y, self.dt)

    def __repr__(self):
        """Defines the printing function for this class"""
        return "<Timeseries ({:f}:{:f}:{:f}, length: {:d})>".format(self.start, self.dt, self.stop, self.n)

    def __mul__(self, m):
        """Defines multiplication of a timeseries object"""
        return Timeseries(self.start * m, self.stop * m, self.dt * m)

    def __truediv__(self, d):
        """Floating point division"""
        return Timeseries(self.start / d, self.stop / d, self.dt / d)

    def __lt__(self, f):
        """Less than operator"""
        return self.timeseries() < f

    def __le__(self, f):
        """Less than or equal operator"""
        return self.timeseries() <= f

    def __gt__(self, f):
        """Greater than operator"""
        return self.timeseries() > f

    def __ge__(self, f):
        """Greater than or equal operator"""
        return self.timeseries() >= f

    def __array__(self):
        """Converts the timeseries into a numpy array"""
        return self.timeseries()

    def __copy__(self):
        """Returns a completely new copy"""
        return copy.deepcopy(self)

    @classmethod
    def is_regular(cls, timeseries, eps=0.10):
        """Given a list of times, determine if the times are "regular"

        Given a list of times, this function determines if the times
        can be accurately stored in a timeseries object. That is,
        it determines if there is a regular delta_t thoughout the timeseries.
        Given floating point error, we use an epilson of 10% of the average
        delta_t to determine if the timeseries is regular. If the timeseries
        is not regular, then the original timeseries must be interpolated
        before storage in a timeseries object.
        :param timeseries: The original series of timepoints (np array)
        :param eps: A fractional percentage of the average delta_t to
        account for floating point errors
        """
        mean_dt = np.mean(np.diff(timeseries))

        # Ensure that all the diffs are less than the epsilon
        if np.any(np.abs(np.diff(timeseries) - mean_dt) > mean_dt * eps):
            return False # Not a valid timeseries
        return True

    @classmethod
    def convert_to_timeseries(cls, t):
        """Convert a floating point timeseries to a timeseries object"""
        if isinstance(t, Timeseries):
            return t
        dt = np.mean(np.diff(t))
        return Timeseries(t[0], t[0] + len(t) * dt, dt)
