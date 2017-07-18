import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from deepgp import AEPDGP_net
from deepbo.models import gpr
from deepbo.models import dgpr
from deepbo.acquisition.bo import ei
from deepbo.tasks.synthetic import branin
from deepbo.tasks.synthetic import hartmann
from deepbo.tasks.synthetic import sin
from deepbo.tasks.synthetic import rosenbrock
from deepbo.tasks.synthetic import goldstein_price
from deepbo.optimisers import lbfgs_search