import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../deepbo')))

# print sys.path[0]

from deepbo.acquisition.bo import ei
from deepbo.acquisition.active import exploration
from deepbo.models import dgpr
from deepbo.models import gpr
from deepbo.optimisers import lbfgs_search
from deepbo.optimisers import pool_search
from deepbo.tasks import base_task

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../geepee')))

import geepee.aep_models as aep
# import geepee.vfe_models as vfe
# import geepee.ep_models as ep
from geepee.kernels import compute_kernel, compute_psi_weave
import geepee.config as config

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from thesis_work import metrics
from thesis_work.data_utils import load_data
from thesis_work.data_utils import delete_blocks
from thesis_work.data_utils import start_df
from thesis_work.data_utils import save_df

