import numpy as np

def rng_check(rng=None):
	
	if rng is None:
		rng = np.random.RandomState(42)
	
	return rng

