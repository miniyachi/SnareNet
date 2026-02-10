from datasets.cvxqp.cvxqp_problem import CvxQPProblem
from datasets.cvx_qcqp.cvxqcqp_problem import QCQP
from datasets.cbf.cbf_problem import SafeControl
from datasets.discrete_cbf.discrete_cbf import DiscSafeControl
from datasets.noncvx.noncvx_problem import NonCvxProblem

from models.snarenet import SnareNet
from models.hardnetaff import HardNetAff
from models.dc3 import DC3
from models.optnet import OptNet

PROBTYPE_TO_CLASS = {
    'cvxqp': CvxQPProblem,
    'cvx_qcqp': QCQP,
    'cbf': SafeControl,
    'discrete_cbf': DiscSafeControl,
    'noncvx': NonCvxProblem,
}

MODELNAME_TO_CLASS = {
    'hardnetaff': HardNetAff,
    'dc3': DC3,
    'snarenet': SnareNet,
    'optnet': OptNet,
}


HISTORY_DF_COLS = [
    '_step', 
    'epoch', 
    'train/loss', 
    'valid/eval', 
    'valid/opt_gap_max', 
    'valid/opt_gap_gmean', 
    'valid/ineq_err_max', 
    'valid/ineq_err_gmean', 
    'valid/ineq_err_nviol', 
    'valid/eq_err_max', 
    'valid/eq_err_gmean', 
    'valid/eq_err_nviol', 
    'valid/n_solved', 
    'valid/last_iter_taken', 
]


TEST_METRICS_DICT_KEYS = [
    '_step',  
    'test/eval', 
    'test/opt_gap_max', 
    'test/opt_gap_gmean', 
    'test/ineq_err_max', 
    'test/ineq_err_gmean', 
    'test/ineq_err_nviol', 
    'test/eq_err_max', 
    'test/eq_err_gmean', 
    'test/eq_err_nviol',
    'test/n_feasible_1e-1',
    'test/n_feasible_1e-2',
    'test/n_feasible_1e-4', 
    'test/n_solved', 
    'test/time', 
]