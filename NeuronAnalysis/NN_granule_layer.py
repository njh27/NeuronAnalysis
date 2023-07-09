import numpy as np
from random import choices



def relu(x, knee):
    return np.maximum(0., x - knee)

def relu_refl(x, knee):
    return np.maximum(0., knee - x)


class MossyFiber(object):
    """ Very simple mossy fiber class that defines a 2D response profile from the relu functions
    """
    def __init__(self, h_knee, v_knee, h_refl, v_refl, h_weight=1., v_weight=1.):
        """ Construct response function given the input parameters. """
        self.h_knee = h_knee
        self.v_knee = v_knee
        self.h_refl = h_refl
        self.v_refl = v_refl
        self.h_weight = h_weight
        self.v_weight = v_weight
        self.h_fun = relu_refl if self.h_refl else relu
        self.v_fun = relu_refl if self.v_refl else relu

    def response(self, h, v):
        """ Returns the response of this mossy fiber given vectors of horizontal and vertical inputs. """
        output = self.h_fun(h, self.h_knee) * self.h_weight
        output += self.v_fun(v, self.v_knee) * self.v_weight
        return output
    

class GranuleCell(object):
    """ Granule cell class that gets inputs from mossy fibers and computes activation response
    """
    def __init__(self, mossy_fibers, mf_weights, activation="relu"):
        """ Construct response function given the input parameters. """
        if activation == "relu":
            self.act_fun = relu
        self.mfs = mossy_fibers
        # Normalize the sum of mf weights to 1 to make sure granule cell can be activated
        self.mf_weights = mf_weights / np.sum(mf_weights)

    def response(self, h, v):
        """ Returns the response of this granule cell given vectors of horizontal and vertical inputs
        by summing the response over its mossy fiber inputs. """
        output = np.zeros(h.shape[0])
        for mf_ind, mf in enumerate(self.mfs):
            output += mf.response(h, v) * self.mf_weights[mf_ind]
        output = self.act_fun(output, 0.)
        return output
    

def make_mossy_fibers(N, knee_win=[-30, 30]):
    """ Makes N mossy fibers for both the horizontal and vertical axis with random response parameters
    and an even distribution of positive and negative sloped activation functions.
    """
    mossy_fibers = []
    # Get random numbers first for speed
    knees = np.random.uniform(knee_win[0], knee_win[1], N)
    is_refl = np.random.choice([True, False], size=N)
    # weights = np.random.uniform(size=(N, 2))
    for n_mf in range(0, N):
        # Make a horizontal and vertical mossy fiber each iteration
        mossy_fibers.append(MossyFiber(knees[n_mf], 0., is_refl[n_mf], False, h_weight=1., v_weight=0.))
        mossy_fibers.append(MossyFiber(0., knees[n_mf], False, is_refl[n_mf], h_weight=0., v_weight=1.))

    return mossy_fibers


def make_granule_cells(N, mossy_fibers):
    """ Make granule cells by choosing 3-5 mossy fibers from "mossy_fibers" and combining them
    with random weights. """
    granule_cells = []
    # Get some random numbers up front
    n_mfs = np.random.randint(3, 6, size=N)
    for n_gc in range(0, N):   
        # Choose a set of mossy fibers and some random weights
        mf_in = choices(mossy_fibers, k=n_mfs[n_gc])
        mf_weights = np.random.uniform(size=len(mf_in))
        granule_cells.append(GranuleCell(mf_in, mf_weights))

    return granule_cells
