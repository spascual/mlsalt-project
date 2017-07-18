import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deepgp import AEPDGP_net
from deepbo.models import gpr
from deepbo.models import dgpr
from deepbo.acquisition.bo import ei
from deepbo.acquisition.active import exploration
from deepbo.tasks import base_task
from deepbo.tasks.synthetic import branin
from deepbo.tasks.synthetic import hartmann
from deepbo.optimisers import grid_search
from deepbo.optimisers import lbfgs_search 
from deepbo.optimisers import pool_search