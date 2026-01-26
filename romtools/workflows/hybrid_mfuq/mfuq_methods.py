from math import prod, floor, comb
from itertools import combinations

import numpy as np
from scipy.optimize import minimize,least_squares,NonlinearConstraint
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

# import autograd.numpy as np
# from autograd import grad
import scipy.stats

def pearsonr_with_axis(x, y, axis=0):
    """
    Compute the Pearson correlation coefficient between two arrays along a specified axis.

    Parameters:
    x : array_like
        Input array.
    y : array_like
        Input array.
    axis : int, optional
        Axis along which to compute the correlation. Default is 0.

    Returns:
    statistic : float
        Pearson correlation coefficient.
    pvalue : float
        Two-tailed p-value.
    """
    # Ensure x and y are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Calculate means
    mean_x = np.mean(x, axis=axis, keepdims=True)
    mean_y = np.mean(y, axis=axis, keepdims=True)

    # Calculate the numerator and denominator for Pearson correlation
    numerator = np.sum((x - mean_x) * (y - mean_y), axis=axis)
    denominator = np.sqrt(np.sum((x - mean_x) ** 2, axis=axis) * np.sum((y - mean_y) ** 2, axis=axis))

    # Calculate Pearson correlation coefficient
    statistic = numerator / denominator

    # Calculate the p-value
    n = np.sum(~np.isnan(x) & ~np.isnan(y), axis=axis)  # Count non-NaN pairs
    df = n - 2  # Degrees of freedom
    t_stat = statistic * np.sqrt(df / (1 - statistic ** 2))  # t-statistic
    pvalue = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stat), df))  # Two-tailed p-value

    return statistic, pvalue


def compute_correlations(X ,y, type, num_folds=None, seed=2025):
    '''
    Compute the correlations between models from data. 
    X is [n_data, n_models, n_pilot]
    y is [n_data] (hifi) or X.shape (lofi)
    type computes either corr(Q0,Qi) or corr(Qi,Qj)
    '''
    if num_folds == None:
        num_folds = X.shape[0]

    if type == "hifi":
        y = y[:,None,None]
    elif type == "lofi":
        X = X[:,:,None,:]
        y = y[:,None,:,:]
    else:
        print('correlation type not implemented!  ' \
               'Options are "hifi" and "lofi"')
        return None
    
    # Initialize KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    # Perform cross-validation
    if type == "hifi":
        corrs = np.zeros((num_folds, X.shape[1], X.shape[2]))
    elif type =="lofi":
        corrs = np.zeros((num_folds, X.shape[1], y.shape[2], X.shape[3]))

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        corrs[i] = pearsonr(X_train, y_train, axis=0)[0]

    return np.mean(corrs, axis=0)


class Pilot():
    '''
    Abstract class which handles pilot sampling.
    Includes strategy to estimate correlations/costs.
    '''
    def __init__(self, s_list, num_pilot, random_seed=2025):
        self.s_list = s_list
        self.num_pilot = num_pilot
        self.rng = np.random.default_rng(random_seed)
    
    def set_train_and_test_labels(self, max_groups=int(1e6)):
        '''
        Given a length-n_pilot list of s values and an integer Np of pilot samples,
        Create lists of train- and test- group indices enabling maximum data re-use

        For each i,s_i in enumerate(s_list):
            Train (resp. test) labels[i] is size [NpCs, s_i] (resp. [NpCs_i, Np-s_i])
            Each of the NpCs_i groups indexes a disjoint splitting of the Np samples
                into s_i training points and Np-s_i testing points
        '''

        pilot_set    = set(range(self.num_pilot))
        train_labels = [0 for s in self.s_list] 
        test_labels  = [0 for s in self.s_list]
        for i,s in enumerate(self.s_list):
            NpCs = comb(self.num_pilot, s)
            if NpCs <= max_groups:
                train_labels[i] = [*map(list,list(combinations(range(self.num_pilot),s)))]
            else:
                train_labels[i] = []
                seen = set()
                while len(train_labels[i]) < max_groups:
                    label = sorted(self.rng.choice(range(self.num_pilot), s, replace=False))
                    tup = tuple(label)
                    if tup not in seen:
                        seen.add(tup)
                        train_labels[i].append(label)
            test_labels[i]  = [list(pilot_set-set(train_labels[i][comb])) 
                               for comb in range(len(train_labels[i]))]
        self.train_labels = train_labels
        self.test_labels = test_labels

    def split_data_using_labels(self, data_list):
        '''
        Given index sets train_labels,test_labels and [N_FOM,...] arrays 
        Split the FOM trajectories x and QoI data y according to labels

        train_labels (resp. test_labels) is a list of n_pilot lists 
            of size [NpCs, s] (resp. [NpCs, Np-s]) indexing the splitting    
        Each element of x,y_train (resp. x,y_test) is an [NpCs,s,...]
            (resp. [NpCs,Np-s,...]) array
        Note that only the first Np << N_FOM samples are used for groups  
        '''
        train_data, test_data = [], []
        for data in data_list:
            train_data_i, test_data_i = [], []
            for (train, test) in zip(self.train_labels, self.test_labels):
                train_data_i.append(data[train]), test_data_i.append(data[test])
            train_data.append(train_data_i), test_data.append(test_data_i)
        return train_data, test_data

    def estimate_FOM_correlations(self, X_test_list, y_test_list):
        '''
        Given pilot lists of model QoI evaluations (X ROM, y FOM),
        Estimate the correlations between the models with an average

        Both X_list, y_list are lists of length n_pilot
        Each X in X_list is an array of shape [NpCs, Np-s, ...]
            representing QoI evals at the Np-s samples NOT used 
            to build the corresponding ROM of size s
        Each y in y_list is an array of shape [NpCs, Np-s, ...]
            representing FOM QoI evals at these same samples
        For each of the n_pilot s values, correlations are computed
            as an average across NpCs groups

        WARNING: pearsonr doesn't do vector QoIs I think, since
                it multiplies elementwise
        '''
        corr_s = []
        # for (X, y) in zip(X_test_list, y_test_list):
        #     out = pearsonr(X, y, alternative='two-sided', method=None, axis=1)
        #     corr_s.append(np.mean(out.statistic, axis=0).squeeze())
        for (X, y) in zip(X_test_list, y_test_list):
            out, _ = pearsonr_with_axis(X, y, axis=1)
            corr_s.append(np.mean(out, axis=0).squeeze())
        return np.array(corr_s)

    def set_ROM_correlation_labels(self, seed=2025):
        '''
        Given pilot list of test_labels (for each s_i, size [NpCs_i, Np-s_i]),
        Compute shared labels usable for ROM correlations (j<i)

        Both model i and model j are trained on different samples
            (with potential overlap)
        This routine computes the sample indices which are not used by either

        Loop through labels[i] for i,s_i in enumerate(pilot)
            Loop through labels[j] for j<i
                Only m = min(NpCs_i,NpCs_j) sample groups are possible; randomize choice
                For each group, can evaluate corr only on intersection of test indices
        Output is nested list with entry[i][j] of size [m, overlap]
            where the size of overlap depends on i,j and the random group index
        '''
        np.random.seed(seed)
        indices = []
        for i,label_i in enumerate(self.test_labels):  #label_i is [nCsi, Np-si]
            indices_i = []
            NpCs_i = len(label_i)
            for j,label_j in enumerate(self.test_labels[:i]):
                NpCs_j = len(label_j)
                random_ij = np.random.permutation(min(NpCs_i, NpCs_j))  # random list of indices
                indices_i.append([list(set(label_i[k]) & set(label_j[k])) for k in random_ij])  # min_{ij}(NpCs), Np - si U sj
            indices.append(indices_i)
        self.ROM_correlation_indices = indices


class MFMC():
    '''
    Abstract class containing instances of the MFMC problem.
    Problem is solved in terms of correlations, not covariances.
    '''
    def __init__(self, budget, type, hybrid: bool = True, n_active: int = 1):
        '''
        Initialize the MFMC object with budget and type parameters.
        Budget is a scalar in units of HF evaluations.
        Type options are "ACV-MF" or "ACV-IS".
        n_active: number of models with nonzero s values (last n_active models)
        '''
        
        self.budget = budget  # computational budget
        self.fac = int(hybrid)  # toggle for hybrid approach
        self.n_active = n_active  # number of active s entries

        # Define MFMC strategy
        if type == 'MF' or type == 'ACV-MF':
            self.type = 'ACV-MF'
        elif type == 'IS' or type == 'ACV-IS':
            self.type = 'ACV-IS'
        else:
            print('Type not implemented!  Defaulting to ACV-MF...')
            self.type = 'ACV-MF'

    def expand_s(self, s_active):
        '''
        Expand compressed s vector to full length by padding with zeros.
        s_active contains only the nonzero entries (last n_active models).
        '''
        n_lofi = len(self.hf_corr_list)
        if self.n_active is None or self.n_active == n_lofi:
            return s_active
        
        # Pad with zeros at the beginning
        s_full = np.zeros(n_lofi)
        s_full[-self.n_active:] = s_active
        return s_full

    def build_F(self, r):
        '''
        Build sampling matrix F(r) given strategy vector r.
        See Gorodetsky et al. 2020 for definitions.
        '''

        if self.type=='ACV-MF':
            xr, yr = np.meshgrid(r,r)
            mins   = np.minimum(xr,yr)
            F      = (mins-1) / mins

        elif self.type=='ACV-IS':
            v = (r-1) / r
            F = v.reshape(-1,1) * v.reshape(1,-1) 
            np.fill_diagonal(F,v)

        return F


    def set_corrs_and_costs(self, hf_corr_list, lf_corr_list, cost_list):
        '''
        Assimilate lists of correlation and cost functions into MFMC object.
        hf_corr_list contains corr(Q_0,Q_i) for all i>0.
        lf_corr_list contains corr(Q_i,Q_j), i>j, 
          in lexicographic order for each i+j=k.
        cost_list contains cost(Q_i)/cost(Q_0) for all i>0.
        '''
        self.hf_corr_list = hf_corr_list  # length n_lofi
        self.lf_corr_list = lf_corr_list  # length n_lofi_choose_2
        self.cost_list = cost_list  # length n_lofi
        
        # Set default n_active if not specified
        if self.n_active is None:
            self.n_active = len(self.hf_corr_list)

    def build_C(self, s_active):
        '''
        Build correlation matrix C(s) given sampling vector s_active.
        s_active contains only nonzero entries.
        '''
        try: 
            self.lf_corr_list
        except AttributeError: 
            print("Correlation functions are not set!")
            return None

        # Expand to full s vector
        s = self.expand_s(s_active)
        
        # Corr funcs must accept entire s as input
        unique_entries = np.array([corr_func(s) for corr_func in self.lf_corr_list])
        
        tmp    = np.sqrt(8*unique_entries.shape[0]+1) + 1
        n      = floor(tmp/2)  # solve quadratic eqn for n_lofi
        idx    = np.tril_indices(n,k=-1)
        C      = np.diag(np.ones(n))  # diagonal is all ones (corr)
        C[idx] = unique_entries
        C     += np.tril(C,k=-1).T
        return C

    def set_objective_and_constraint(self, log=True, bounds=[]):
        '''
        Set objective and constraint functions defining MFMC optimization.
        '''
        try:
            self.cost_list
            self.lf_corr_list
            self.hf_corr_list
        except AttributeError:
            print("Cost and correlation functions are not set!")
            return None
        
        n_lofi = len(self.hf_corr_list)
        
        # Bounds to accommodate known variable ranges
        if bounds==[]:
            state_dim = n_lofi + 1 + self.n_active  # N + r + s_active
            self.bounds = [(None,None) for i in range(state_dim)]
        else:
            self.bounds = bounds

        # toggle on/off log of objective
        self.log = log

        def objective(x):
            '''
            Optimization objective given sampling x = (N, r, s_active).
            Value is the ratio var(Q_MFMC)/var(Q_0).
            '''
            N = x[0]  # num of hifi evaluations
            r = x[1:n_lofi+1]  # oversampling ratios
            s_active = x[n_lofi+1:]  # active entries of s
            
            F = self.build_F(r)
            C = self.build_C(s_active)

            # Expand s to full vector for corr_func
            s = self.expand_s(s_active)
            c = np.array([corr_func(s) for corr_func in self.hf_corr_list])

            vec = np.diag(F)*c
            R2 = np.linalg.solve(F*C, vec).dot(vec)
            if self.log:
                return np.log((1 - R2) / N)
            else:
                return (1 - R2) / N

        def constraint(x):
            '''
            Optimization constraint given sampling x = (N, r, s_active).
            Ensures that cost does not exceed self.budget.
            '''
            N = x[0]  # num of hifi evaluations
            r = x[1:n_lofi+1]  # oversampling ratios
            s_active = x[n_lofi+1:]  # active entries of s
            
            # Expand s to full vector
            s = self.expand_s(s_active)

            # Cost funcs must accept entire s as input
            w = np.array([cost_func(s) for cost_func in self.cost_list])
            return N * (1 + np.dot(r,w) + self.fac*np.sum(s_active))

        self.objective = objective
        self.constraint = constraint

    def solve(self):
        '''
        Solve the MFMC optimization problem with SLSQP.
        '''
        try:
            self.constraint
            self.objective
        except AttributeError:
            print("Objective and constraint not set!")
            return None

        n_lofi = len(self.cost_list)
        state_dim = 1 + n_lofi + self.n_active  # N + r + s_active
        
        x0 = np.random.uniform(1, self.bounds[-1][1], state_dim)  # random initialization
        # Set budget constraint and solve with SciPy
        nlc = NonlinearConstraint(self.constraint, -np.inf, self.budget)
        options = {'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-5}
        result = minimize(self.objective, x0, method='SLSQP', jac='3-point',
                          constraints=nlc, bounds=self.bounds, options=options)

        self.result = result