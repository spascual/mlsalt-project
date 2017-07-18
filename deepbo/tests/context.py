import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deepgp import AEPDGP_net
from deepbo.models import gpr
from deepbo.models import dgpr
from deepbo.acquisition.bo import ei
from deepbo.tasks import base_task
from deepbo.optimisers import grid_search 