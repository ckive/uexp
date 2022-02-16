END_HORIZON = 60

# Train on past X amount of data, adjusting horizon --> END_HORIZON

import pandas as pd
import numpy as np
from ue.uexp.models.dense_multi_variate import MMLP_v1
