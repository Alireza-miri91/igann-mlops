import time
import torch
import torch.nn as nn
import warnings
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import cvxpy as cp

warnings.simplefilter("once", UserWarning)


class LinearModel:
    """Picklable container for a linear model's coefficients and intercept."""
    def __init__(self, coef_=None, intercept_=None):
        self.coef_ = coef_
        self.intercept_ = intercept_


class torch_Ridge:
    def __init__(self, alpha, device):
        self.coef_ = None
        self.alpha = alpha
        self.device = device

    def fit(self, X, y):
        self.coef_ = torch.linalg.solve(
            X.T @ X + self.alpha * torch.eye(X.shape[1]).to(self.device), X.T @ y
        )

    def predict(self, X):
        return X.to(self.device) @ self.coef_


def cvxpy_logistic_monotonic(X, y, mon_inc_signs, lambda_reg=1e-3):
    """
    Solve logistic regression with monotonic constraints using CVXPY.

    Parameters:
        X         : numpy array of shape (n_samples, n_features).
        y         : numpy array of shape (n_samples,), with values in {-1, 1}.
        mon_inc_signs: list of (int, int)
        List of (feature_index, sign) pairs. sign=1 for increasing, sign=-1 for decreasing.
        If sign is 0, the feature is not constrained.
        lambda_reg: regularization strength.

    Returns:
        beta      : Fitted coefficient vector.
        b         : Fitted intercept.
    """
    n_samples, n_features = X.shape
    beta = cp.Variable(n_features)
    b = cp.Variable()

    # Logistic loss: log(1+exp(-y_i*(x_i^T beta + b))) for each sample i.
    logistic_loss = cp.sum(cp.logistic(-cp.multiply(y, X @ beta + b)))
    reg_term = 0.5 * lambda_reg * cp.sum_squares(beta)
    objective = cp.Minimize(logistic_loss + reg_term)

    # Enforce nonzero coefficients for the specified monotonic features.
    constraints = []
    for j, sign in mon_inc_signs:
        if sign == 1:
            constraints.append(beta[j] >= 0)
        else:
            constraints.append(beta[j] <= 0)

    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL)
        beta_opt = beta.value
        b_opt = b.value
    except Exception as e:
        warnings.warn(f"CVXPY solver failed in cvxpy_logistic_monotonic: {e}. Using zeros for beta and b.")
        beta_opt = np.zeros(n_features)
        b_opt = 0.0
    return beta_opt, b_opt

class ShiftedReLU(nn.Module):
    def __init__(self, shift=1.0):
        super(ShiftedReLU, self).__init__()
        self.shift = np.random.rand() - 1.0
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.shift)

class ELM_Regressor:
    """
    Single hidden layer neural network for regression.
    Trainable parameters are only the output weights.
    Hidden weights are sampled once.
    If a list of monotonic features is provided via 'mon_features'
    (which are indices relative to the numerical features),
    the output-layer weights are optimized via CVXPY such that
    the derivative (of the numerical part) plus a cumulative offset is nonnegative.
    """

    def __init__(
        self,
        n_input,
        n_categorical_cols,
        n_hid,
        seed=0,
        elm_scale=10,
        elm_alpha=0.0001,
        act="elu",
        device="cpu",
    ):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.n_numerical_cols = n_input - n_categorical_cols
        self.n_categorical_cols = n_categorical_cols
        # Random weights for the numerical part (not optimized)
        self.hidden_list = torch.normal(
            mean=torch.zeros(self.n_numerical_cols, self.n_numerical_cols * n_hid),
            std=elm_scale,
        ).to(device)
        mask = torch.block_diag(*[torch.ones(n_hid)] * self.n_numerical_cols).to(device)
        self.hidden_mat = self.hidden_list * mask  # shape: (n_numerical_cols, n_numerical_cols*n_hid)
        self.output_model = None
        self.n_input = n_input

        self.n_hid = n_hid
        self.elm_scale = elm_scale
        self.elm_alpha = elm_alpha
        if act == "elu":
            self.act = torch.nn.ELU()
            #self.act = ShiftedReLU()
            self.act_name = "elu"
        elif act == "relu":
            self.act = torch.nn.ReLU()
            self.act_name = "relu"
        else:
            self.act = act
            self.act_name = "custom"
        self.device = device

    def _act_deriv(self, preact):
        """
        Compute derivative of the activation function given pre-activation values.
        Here we implement for ReLU and ELU.
        preact: torch tensor.
        Returns: same shape as preact.
        """
        if self.act_name == "relu":
            return (preact > 0).float()
        elif self.act_name == "elu":
            # For ELU, derivative is 1 if preact >= 0, and exp(preact) if preact < 0.
            return torch.where(preact >= 0, torch.ones_like(preact), torch.exp(preact))
        else:
            # Fallback: numerical derivative (not recommended for production)
            eps = 1e-4
            return (self.act(preact + eps) - self.act(preact - eps)) / (2 * eps)

    def get_hidden_values(self, X):
        """
        Computes the hidden layer output.
        For the numerical features, computes preactivation and applies the activation.
        Then, concatenates the (untouched) categorical features.
        """
        X_num = X[:, : self.n_numerical_cols]  # shape: (n_samples, n_numerical_cols)
        preact = X_num @ self.hidden_mat  # shape: (n_samples, n_numerical_cols*n_hid)
        X_hid_num = self.act(preact)  # activated numerical part
        if self.n_categorical_cols > 0:
            X_cat = X[:, self.n_numerical_cols :]
            X_hid = torch.hstack((X_hid_num, X_cat))
        else:
            X_hid = X_hid_num
        return X_hid, preact

    def predict(self, X, hidden=False):
        """
        Full prediction with the model for input X.
        """
        if hidden:
            X_hid = X
        else:
            X_hid, _ = self.get_hidden_values(X)
        out = X_hid @ self.output_model.coef_
        return out

    def predict_single(self, x, i):
        """
        Partial output of one base function for feature i.
        """
        x_in = x.reshape(len(x), 1)
        if i < self.n_numerical_cols:
            x_in = x_in @ self.hidden_mat[i, i * self.n_hid : (i + 1) * self.n_hid].unsqueeze(0)
            x_in = self.act(x_in)
            out = x_in @ self.output_model.coef_[i * self.n_hid : (i + 1) * self.n_hid].unsqueeze(1)
        else:
            start_idx = self.n_numerical_cols * self.n_hid + (i - self.n_numerical_cols)
            out = x_in @ self.output_model.coef_[start_idx : start_idx + 1].unsqueeze(1)
        return out
    
    '''
    def compute_importance(self, X):
        """
        Compute the average importance (effect) for each numerical feature for this regressor.
        
        Parameters:
            X: torch.Tensor of shape (n_samples, n_features) after preprocessing.
            It is assumed that the first self.n_numerical_cols columns correspond to numerical features.
        
        Returns:
            importance: numpy array of shape (self.n_numerical_cols,)
                        containing the average absolute contribution for each numerical feature.
        """
        # Total number of features in the preprocessed space.
        total_features = self.n_numerical_cols + self.n_categorical_cols
        importance = np.zeros(total_features, dtype=np.float32)

        # Ensure that X is on the same device as the model.
        X = X.to(self.device)

        # Loop over all features in the processed feature space.
        for i in range(total_features):
            # Extract the column for feature i.
            # For numerical features, predict_single aggregates the n_hid block internally.
            # For categorical features, predict_single uses the corresponding one-hot column.
            x_i = X[:, i].unsqueeze(1)
            contribution = self.predict_single(x_i, i)
            # Average the absolute contribution over samples.
            importance[i] = torch.mean(torch.abs(contribution)).item()
            
        return importance
    '''
    def fit(self, X, y, mult_coef, mon_features=None, mon_sign=None, cumulative_deriv=None, 
        #cumulative_importance=None, importance=None,
        boost_rate=1.0, numerical_cols=None,
        feature_to_indices=None):
        if numerical_cols is not None:
            self.numerical_cols = numerical_cols
        else:
            # Fallback: if not provided, assume features are indexed 0...n_numerical_cols-1 as strings.
            self.numerical_cols = [str(i) for i in range(self.n_numerical_cols)]

        # Compute hidden layer outputs and preactivations.
        X_hid, preact = self.get_hidden_values(X)
        X_hid_mult = X_hid * mult_coef

        # Determine whether any constraints (monotonic or importance) are provided.
        has_monotonic = mon_features is not None and len(mon_features) > 0
        #has_importance = importance is not None and len(importance) > 0

        if not has_monotonic:
            # No constraints provided: use simple ridge regression.
            m = torch_Ridge(alpha=self.elm_alpha, device=self.device)
            m.fit(X_hid_mult, y)
            self.output_model = m
            derivative_contrib = None
            return X_hid, derivative_contrib

        X_hid_np = X_hid_mult.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        n_samples, n_features = X_hid_np.shape
        beta = cp.Variable(n_features)
        objective = cp.Minimize(cp.sum_squares(X_hid_np @ beta - y_np) + self.elm_alpha * cp.sum_squares(beta))
        constraints = []

        # Existing monotonic constraints:
        X_num = X[:, : self.n_numerical_cols]  
        preact_np = (X_num @ self.hidden_mat).detach().cpu().numpy()  
        preact_tensor = torch.from_numpy(preact_np).to(self.device)
        act_deriv = self._act_deriv(preact_tensor).detach().cpu().numpy()  

        if mon_features is not None and mon_sign is not None:
            for idx, (feat, sign) in enumerate(zip(mon_features, mon_sign)):
                if feat >= self.n_numerical_cols:
                    warnings.warn(f"Monotonic constraint requested for feature {feat} which is not numerical; skipping.")
                    continue
                col_start = feat * self.n_hid
                col_end = (feat + 1) * self.n_hid
                hidden_block = self.hidden_mat[feat, col_start:col_end].detach().cpu().numpy()
                beta_block = beta[col_start:col_end]
                A_i = act_deriv[:, col_start:col_end] * hidden_block
                deriv_current = A_i @ beta_block  
                if sign == 1:
                    constraints.append(cumulative_deriv[:, idx] + boost_rate * deriv_current >= 0)
                else:
                    constraints.append(cumulative_deriv[:, idx] + boost_rate * deriv_current <= 0)

        '''
        if importance is not None:
            for (f_i, f_j) in importance:
                # Look up the grouped indices for each feature from the passed mapping.
                group_i = feature_to_indices.get(f_i, [])
                group_j = feature_to_indices.get(f_j, [])
                if not group_i or not group_j:
                    warnings.warn(f"Importance constraint for features '{f_i}' and/or '{f_j}' could not be mapped; skipping.")
                    continue

                # For each group, sum the contribution from all corresponding columns.
                expr_i_total = 0
                for orig_idx in group_i:
                    if orig_idx < self.n_numerical_cols:
                        # For a numerical feature: contribution comes from its n_hid block.
                        col_start = orig_idx * self.n_hid
                        col_end = (orig_idx + 1) * self.n_hid
                        activated = X_hid[:, col_start:col_end].detach().cpu().numpy()
                        expr = activated @ beta[col_start:col_end]
                    else:
                        # For a categorical feature: its columns are appended after the numerical block.
                        # Compute the corresponding index in X_hid.
                        cat_col = (orig_idx - self.n_numerical_cols) + self.n_numerical_cols * self.n_hid
                        activated = X_hid[:, cat_col].detach().cpu().numpy()  # single column per one-hot
                        expr = beta[cat_col] * activated
                    expr_i_total += expr

                expr_j_total = 0
                for orig_idx in group_j:
                    if orig_idx < self.n_numerical_cols:
                        col_start = orig_idx * self.n_hid
                        col_end = (orig_idx + 1) * self.n_hid
                        activated = X_hid[:, col_start:col_end].detach().cpu().numpy()
                        expr = activated @ beta[col_start:col_end]
                    else:
                        cat_col = (orig_idx - self.n_numerical_cols) + self.n_numerical_cols * self.n_hid
                        activated = X_hid[:, cat_col].detach().cpu().numpy()
                        expr = beta[cat_col] * activated
                    expr_j_total += expr

                # Average the total expression over samples.
                #avg_expr_i = cp.sum(expr_i_total) / n_samples
                #avg_expr_j = cp.sum(expr_j_total) / n_samples

                avg_expr_i = cp.sum(cp.abs(expr_i_total)) / n_samples
                avg_expr_j = cp.sum(cp.abs(expr_j_total)) / n_samples

                # For cumulative importance, aggregate over the group (here we take the mean).
                if cumulative_importance is not None:
                    cum_imp_i = np.mean([cumulative_importance[idx].item() for idx in group_i])
                    cum_imp_j = np.mean([cumulative_importance[idx].item() for idx in group_j])
                else:
                    cum_imp_i, cum_imp_j = 0.0, 0.0

                constraints.append(cum_imp_i + boost_rate * avg_expr_i <= cum_imp_j + boost_rate * avg_expr_j)
        '''
        
        # Solve the problem
        try:
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.CLARABEL)
            beta_opt = beta.value
        except Exception as e:
            warnings.warn(f"CVXPY solver failed in ELM_Regressor.fit: {e}. Using zeros for beta.")
            beta_opt = np.zeros(n_features)

        m = torch_Ridge(alpha=self.elm_alpha, device=self.device)
        m.coef_ = torch.tensor(beta_opt, dtype=torch.float32).to(self.device)
        self.output_model = m

        if mon_features is None:
            derivative_contrib = None
        else:
            derivative_contrib = np.zeros((n_samples, len(mon_features)))
            for idx, (feat, sign) in enumerate(zip(mon_features, mon_sign)):
                if feat >= self.n_numerical_cols:
                    continue
                col_start = feat * self.n_hid
                col_end = (feat + 1) * self.n_hid
                hidden_block = self.hidden_mat[feat, col_start:col_end].detach().cpu().numpy()
                contrib = (act_deriv[:, col_start:col_end] * hidden_block) @ beta_opt[col_start:col_end]
                derivative_contrib[:, idx] = boost_rate * sign * contrib
            derivative_contrib = torch.tensor(derivative_contrib, dtype=torch.float32).to(self.device)
        return X_hid, derivative_contrib

class IGANN:
    def __init__(
        self,
        task="classification",
        n_hid=10,
        n_estimators=5000,
        boost_rate=0.1,
        init_reg=1,
        elm_scale=1,
        elm_alpha=1,
        act="elu",
        early_stopping=50,
        device="cpu",
        random_state=1,
        verbose=0,
        monotonicity=None,       # already there: monotonic features.
        #importance=None,    # NEW: list of tuples for importance constraints.
    ):
        self.task = task
        self.n_hid = n_hid
        self.elm_scale = elm_scale
        self.elm_alpha = elm_alpha
        self.init_reg = init_reg
        self.act = act
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.boost_rate = boost_rate
        self.monotonicity = monotonicity
        #self.importance = importance  # NEW: store the importance constraints
        self.target_remapped_flag = False

    def _clip_p(self, p):
        if torch.max(p) > 100 or torch.min(p) < -100:
            warnings.warn(
                "Cutting prediction to [-100, 100]. Did you forget to scale y? Consider higher regularization elm_alpha."
            )
            return torch.clip(p, -100, 100)
        else:
            return p

    def _clip_p_numpy(self, p):
        if np.max(p) > 100 or np.min(p) < -100:
            warnings.warn(
                "Cutting prediction to [-100, 100]. Did you forget to scale y? Consider higher regularization elm_alpha."
            )
            return np.clip(p, -100, 100)
        else:
            return p

    def _loss_sqrt_hessian(self, y, p):
        """
        This function computes the square root of the hessians of the log loss or the mean squared error.
        """
        if self.task == "classification":
            return 0.5 / torch.cosh(0.5 * y * p)
        else:
            return torch.sqrt(torch.tensor([2.0]).to(self.device))

    def _get_y_tilde(self, y, p):
        if self.task == "classification":
            return y / torch.exp(0.5 * y * p)
        else:
            return torch.sqrt(torch.tensor(2.0).to(self.device)) * (y - p)

    def _reset_state(self):
        self.regressors = []
        self.boosting_rates = []
        self.train_scores = []
        self.val_scores = []
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.regressor_predictions = []

    def _preprocess_feature_matrix(self, X, fit_transform=True):
        """
        Preprocesses the feature matrix using a ColumnTransformer that:
        One-hot encodes categorical columns.
        Ensures that numerical columns appear first in the transformed matrix,
        preserving their order.
        """
        if not isinstance(X, pd.DataFrame):
            warnings.warn("Please provide a pandas DataFrame as input for X. Processing stopped.")
            return

        X.columns = [str(c) for c in X.columns]
        # Preserve the original order for numerical columns.
        self.categorical_cols = list(X.select_dtypes(include=["category", "object", "string"]).columns)
        self.numerical_cols = [col for col in X.columns if col not in self.categorical_cols]

        transformers = []
        if len(self.categorical_cols) > 0:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_cols,
                )
            )

        if fit_transform:
            self.column_transformer = ColumnTransformer(
                transformers=transformers,
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas")
            X_transformed = self.column_transformer.fit_transform(X)
        else:
            X_transformed = self.column_transformer.transform(X)

        new_feature_names = []
        if len(self.numerical_cols) > 0:
            new_feature_names.extend(self.numerical_cols)
        if len(self.categorical_cols) > 0:
            one_hot_encoder = self.column_transformer.named_transformers_.get("cat", None)
            if one_hot_encoder and one_hot_encoder != "drop":
                cat_feature_names = list(one_hot_encoder.get_feature_names_out(self.categorical_cols))
                new_feature_names.extend(cat_feature_names)
        self.feature_names = new_feature_names
        self.n_numerical_cols = len(self.numerical_cols)
        self.n_categorical_cols = len(self.feature_names) - self.n_numerical_cols

        missing = [col for col in self.feature_names if col not in X_transformed.columns]
        if missing:
            raise ValueError(f"DataFrame is missing columns needed for ordering: {missing}")

        # After setting self.feature_names, build a mapping for all features.
        self.feature_to_indices = {}
        # For numerical features: each name appears exactly once.
        for i, col in enumerate(self.numerical_cols):
            # Find the index of this column in self.feature_names.
            # (They are expected to appear in the same order.)
            if col in self.feature_names:
                self.feature_to_indices[col] = [self.feature_names.index(col)]
            else:
                warnings.warn(f"Numerical column '{col}' not found in feature_names.")
        # For categorical features: group one-hot columns that start with the original name.
        for cat in self.categorical_cols:
            # For one-hot encoding, the new names typically have a prefix (e.g. "cat_test_A", "cat_test_B", ...)
            indices = [i for i, name in enumerate(self.feature_names) if name.startswith(cat + "_")]
            if indices:
                self.feature_to_indices[cat] = indices
            else:
                # If for some reason the categorical column name appears exactly (e.g. if only one column was produced)
                if cat in self.feature_names:
                    self.feature_to_indices[cat] = [self.feature_names.index(cat)]
                else:
                    warnings.warn(f"Categorical column '{cat}' not found in feature_names.")
        # Return the final processed tensor as before.
        X_final = X_transformed[self.feature_names]
        return torch.tensor(X_final.to_numpy(), dtype=torch.float32)

    def fit(self, X, y, val_set=None):
        """
        Fits the model on training data.
        Supports monotonic constraints by accepting a dictionary mapping feature names to direction (+1 for increasing, -1 for decreasing).
        For classification, if monotonicity is provided, a CVXPY-based logistic regression
        with nonnegative (increasing) or nonpositive (decreasing) coefficients for the specified features is used.
        For regression, if monotonicity is provided, a CVXPY-based nonnegative/ nonpositive least squares
        solver is used for those features.
        Example:
            monotonicity = {'0': 1, '2': -1}  # Feature 0 increasing, feature 2 decreasing
        """
        indices = np.arange(len(X))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=0.15,
            stratify=y if self.task == "classification" else None,
            random_state=self.random_state,
        )
        self.raw_X = X.copy()
        self.raw_X_train = X.iloc[train_indices]
        self.raw_X_val = X.iloc[val_indices]
        if isinstance(y, (pd.Series, pd.DataFrame)):
            self.raw_y_train = y.iloc[train_indices]
            self.raw_y_val = y.iloc[val_indices]
        self._reset_state()

        X_proc = self._preprocess_feature_matrix(X)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        y = torch.from_numpy(y.squeeze()).float()

        if self.task == "classification":
            if torch.min(y) != -1:
                self.target_remapped_flag = True
                y = 2 * y - 1

        # Map monotonic constraint feature names to indices (relative to numerical features)
        mon_idx = []
        mon_sign = []
        if self.monotonicity is not None:
            for f, direction in self.monotonicity.items():
                if f in self.numerical_cols:
                    mon_idx.append(self.numerical_cols.index(f))
                    mon_sign.append(1 if direction in ["+1", +1] else -1)
                else:
                    warnings.warn(f"Monotonic constraint requested for feature '{f}' which is not numerical; skipping.")
        self.mon_idx = mon_idx
        self.mon_sign = mon_sign

        if self.task == "classification":
            if self.mon_idx:
                X_np = X_proc.cpu().numpy()
                y_np = y.cpu().numpy()
                beta, b = cvxpy_logistic_monotonic(X_np, y_np, list(zip(self.mon_idx, self.mon_sign)), lambda_reg=self.init_reg)
                self.linear_model = LinearModel()
                self.linear_model.coef_ = beta
                self.linear_model.intercept_ = b
            else:
                self.linear_model = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=1 / self.init_reg,
                    random_state=self.random_state,
                )
                self.linear_model.fit(X_proc, y)
            self.criterion = lambda prediction, target: torch.nn.BCEWithLogitsLoss()(
                prediction, torch.nn.ReLU()(target)
            )
        elif self.task == "regression":
            # Use CVXPY if either monotonic or importance constraints are provided.
            if self.mon_idx is not None:
                X_np = X_proc.cpu().numpy()
                y_np = y.cpu().numpy()
                n_samples, n_features = X_np.shape
                beta = cp.Variable(n_features)
                b = cp.Variable(1)
                reg_term = 0.5 * self.init_reg * cp.sum_squares(beta)
                objective = cp.Minimize(0.5 * cp.sum_squares(X_np @ beta + b - y_np) + reg_term)
                constraints = []
                
                # Enforce monotonicity if provided
                if self.mon_idx:
                    for i, sign in zip(self.mon_idx, self.mon_sign):
                        if i < self.n_numerical_cols:
                            if sign == 1:
                                constraints.append(beta[i] >= 0)
                            else:
                                constraints.append(beta[i] <= 0)
                constraints.append(b >= -0.1)
                constraints.append(b <= 0.1)
                '''
                
                # Enforce importance constraints if provided.
                if self.importance is not None:
                    for (f_i, f_j) in self.importance:
                        try:
                            idx_i = self.numerical_cols.index(f_i)
                            idx_j = self.numerical_cols.index(f_j)
                        except Exception:
                            warnings.warn(f"Importance constraint for features '{f_i}' and '{f_j}' could not be mapped to numerical indices; skipping.")
                            continue
                        # Enforce that the coefficient for f_i is no larger than that for f_j.
                        constraints.append(beta[idx_i] <= beta[idx_j])
                '''
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.CLARABEL)
                coef = beta.value
                intercept = b.value
                self.linear_model = LinearModel()
                self.linear_model.coef_ = coef
                self.linear_model.intercept_ = (intercept[0] if isinstance(intercept, np.ndarray) else intercept)
            else:
                from sklearn.linear_model import Lasso
                self.linear_model = Lasso(alpha=self.init_reg)
                self.linear_model.fit(X_proc, y)
            self.criterion = torch.nn.MSELoss()

        else:
            warnings.warn("Task not implemented. Can be classification or regression")

        if val_set is None:
            X_train = X_proc[train_indices]
            X_val = X_proc[val_indices]
            y_train = y[train_indices]
            y_val = y[val_indices]
        else:
            X_train = X_proc
            y_train = y
            X_val, y_val = val_set

        if self.task == "classification":
            y_hat_train = (torch.squeeze(torch.from_numpy(self.linear_model.coef_.astype(np.float32)) @ X_train.T)
                           + float(self.linear_model.intercept_))
            y_hat_val = (torch.squeeze(torch.from_numpy(self.linear_model.coef_.astype(np.float32)) @ X_val.T)
                         + float(self.linear_model.intercept_))
        else:
            coef = torch.from_numpy(self.linear_model.coef_.astype(np.float32)).to(X_train.device)
            intercept = float(self.linear_model.intercept_)
            y_hat_train = X_train @ coef + intercept
            y_hat_val = X_val @ coef + intercept

        self.X_min = list(X_proc.min(axis=0))
        self.X_max = list(X_proc.max(axis=0))
        self.unique = [torch.unique(X_proc[:, i]) for i in range(X_proc.shape[1])]
        self.hist = [torch.histogram(X_proc[:, i]) for i in range(X_proc.shape[1])]

        if self.verbose >= 1:
            print("Training shape: {}".format(X_proc.shape))
            print("Validation shape: {}".format(X_val.shape))
            print("Regularization: {}".format(self.init_reg))

        train_loss_init = self.criterion(y_hat_train, y_train)
        val_loss_init = self.criterion(y_hat_val, y_val)
        if self.verbose >= 1:
            print("Train: {:.4f} Val: {:.4f} {}".format(train_loss_init, val_loss_init, "init"))

        X_train, y_train, y_hat_train, X_val, y_val, y_hat_val = (
            X_train.to(self.device),
            y_train.to(self.device),
            y_hat_train.to(self.device),
            X_val.to(self.device),
            y_val.to(self.device),
            y_hat_val.to(self.device),
        )

        if self.mon_idx:
            cumulative_deriv = np.zeros((X_train.shape[0], len(self.mon_idx)), dtype=np.float32)
            for idx, (feat, sign) in enumerate(zip(self.mon_idx, self.mon_sign)):
                if feat < self.n_numerical_cols:
                    cumulative_deriv[:, idx] = self.linear_model.coef_[feat]
                else:
                    cumulative_deriv[:, idx] = 0.0
            cumulative_deriv = torch.tensor(cumulative_deriv, dtype=torch.float32).to(self.device)
        else:
            cumulative_deriv = None
            
        '''
        if self.importance is not None:
            linear_coef = torch.from_numpy(self.linear_model.coef_.astype(np.float32)).to(X_train.device)
            # Compute baseline importance for every column in X_train
            baseline_importance = torch.mean(torch.abs(X_train * linear_coef), dim=0)
            cumulative_importance = baseline_importance.clone().detach()
        else:
            cumulative_importance = None
        '''

        self._run_optimization(
            X_train, y_train, y_hat_train, X_val, y_val, y_hat_val, val_loss_init, cumulative_deriv, #cumulative_importance
        )
        return
    
    def get_params(self, deep=True):
        return {
            "task": self.task,
            "n_hid": self.n_hid,
            "elm_scale": self.elm_scale,
            "elm_alpha": self.elm_alpha,
            "init_reg": self.init_reg,
            "act": self.act,
            "n_estimators": self.n_estimators,
            "early_stopping": self.early_stopping,
            "device": self.device,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "boost_rate": self.boost_rate,
            "monotonicity": self.monotonicity,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if not hasattr(self, parameter):
                raise ValueError(
                    "Invalid parameter %s for estimator %s. Check the list of available parameters with `estimator.get_params().keys()`."
                    % (parameter, self)
                )
            setattr(self, parameter, value)
        return self

    def score(self, X, y, metric=None):
        predictions = self.predict(X)
        if self.task == "regression":
            # if these is no metric specified use default regression metric "mse"
            if metric is None:
                metric = "mse"
            metric_dict = {"mse": mean_squared_error, "r_2": r2_score}
            return metric_dict[metric](y, predictions)
        else:
            if metric is None:
                # if there is no metric specified use default classification metric "accuracy"
                metric = "accuracy"
            metric_dict = {
                "accuracy": accuracy_score,
                "precision": precision_score,
                "recall": recall_score,
                "f1": f1_score,
            }
            return metric_dict[metric](y, predictions)

    def _run_optimization(self, X, y, y_hat, X_val, y_val, y_hat_val, best_loss, cumulative_deriv, #cumulative_importance
                          ):
        counter_no_progress = 0
        best_iter = 0

        for counter in range(self.n_estimators):
            #print(cumulative_importance)
            hessian_train_sqrt = self._loss_sqrt_hessian(y, y_hat)
            y_tilde = torch.sqrt(torch.tensor(0.5).to(self.device)) * self._get_y_tilde(y, y_hat)
            
            regressor = ELM_Regressor(
                n_input=X.shape[1],
                n_categorical_cols=self.n_categorical_cols,
                n_hid=self.n_hid,
                seed=counter,
                elm_scale=self.elm_scale,
                elm_alpha=self.elm_alpha,
                act=self.act,
                device=self.device,
            )
            
            # Call regressor.fit passing both sets of parameters.
            # Note: even if mon_features is empty (or None), you pass the importance parameters.
            X_hid, deriv_contrib = regressor.fit(
                X,
                y_tilde,
                torch.sqrt(torch.tensor(0.5).to(self.device)) * self.boost_rate * hessian_train_sqrt[:, None],
                mon_features=self.mon_idx if (self.mon_idx is not None and len(self.mon_idx) > 0) else None,
                cumulative_deriv=cumulative_deriv if (self.mon_idx is not None and len(self.mon_idx) > 0) else None,
                #cumulative_importance=cumulative_importance,
                #importance=self.importance,
                boost_rate=self.boost_rate,
                numerical_cols=self.numerical_cols,
                feature_to_indices=self.feature_to_indices,   # new argument
                mon_sign=self.mon_sign # new argument
            )
            
            # Only update the monotonic cumulative derivative if mon_features were used.
            if self.mon_idx is not None and len(self.mon_idx) > 0:
                cumulative_deriv = cumulative_deriv + deriv_contrib
            
            '''
            # Independently update cumulative importance if importance constraints are provided.
            if self.importance is not None:
                new_importance = regressor.compute_importance(X)
                cumulative_importance = cumulative_importance + self.boost_rate * new_importance
            '''
            
            train_regressor_pred = regressor.predict(X_hid, hidden=True).squeeze()
            val_regressor_pred = regressor.predict(X_val).squeeze()

            self.regressor_predictions.append(train_regressor_pred)

            y_hat = y_hat + self.boost_rate * train_regressor_pred
            y_hat_val = y_hat_val + self.boost_rate * val_regressor_pred

            y_hat = self._clip_p(y_hat)
            y_hat_val = self._clip_p(y_hat_val)

            train_loss = self.criterion(y_hat, y)
            val_loss = self.criterion(y_hat_val, y_val)

            self.regressors.append(regressor)
            self.boosting_rates.append(self.boost_rate)
            self.train_losses.append(train_loss.cpu())
            self.val_losses.append(val_loss.cpu())

            counter_no_progress += 1
            if val_loss < best_loss:
                best_iter = counter + 1
                best_loss = val_loss
                counter_no_progress = 0

            if self.verbose >= 1:
                print(f"Iteration {counter}: Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f} | No progress: {counter_no_progress}")

            if counter_no_progress > self.early_stopping and self.early_stopping > 0:
                break

        if self.early_stopping > 0:
            if self.verbose > 0:
                print(f"Cutting at {best_iter}")
            self.regressors = self.regressors[:best_iter]
            self.boosting_rates = self.boosting_rates[:best_iter]
        return best_loss

    def predict_proba(self, X):
        """
        Similarly to sklearn, this function returns a matrix of the same length as X and two columns.
        The first column denotes the probability of class -1, and the second column denotes the
        probability of class 1.
        """
        if self.task == "regression":
            warnings.warn(
                "The call of predict_proba for a regression task was probably incorrect."
            )

        pred = self.predict_raw(X)
        pred = self._clip_p_numpy(pred)
        pred = 1 / (1 + np.exp(-pred))

        ret = np.zeros((len(X), 2), dtype=np.float32)
        ret[:, 1] = pred
        ret[:, 0] = 1 - pred

        return ret

    def predict(self, X):
        """
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, it returns the binary target values in a 1-d np.array, it can hold -1 and 1.
        """
        if self.task == "regression":
            return self.predict_raw(X)
        else:
            pred_raw = self.predict_raw(X)
            # detach and numpy pred_raw
            pred = np.where(
                pred_raw < 0,
                np.ones_like(pred_raw) * -1,
                np.ones_like(pred_raw),
            ).squeeze()

            if self.target_remapped_flag:
                pred = np.where(pred == -1, 0, 1)

            return pred

    def predict_raw(self, X):
        """
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, it returns the raw logit values.
        """
        X = self._preprocess_feature_matrix(X, fit_transform=False).to(self.device)

        pred_nn = torch.zeros(len(X), dtype=torch.float32).to(self.device)
        for boost_rate, regressor in zip(self.boosting_rates, self.regressors):
            pred_nn += boost_rate * regressor.predict(X).squeeze()
        pred_nn = pred_nn.detach().cpu().numpy()
        X = X.detach().cpu().numpy()
        pred = (
            pred_nn
            + (self.linear_model.coef_.astype(np.float32) @ X.transpose()).squeeze()
            + self.linear_model.intercept_
        )

        return pred

    def _flatten(self, l):
        return [item for sublist in l for item in sublist]

    def _split_long_titles(self, l):
        return "\n".join(l[p : p + 22] for p in range(0, len(l), 22))

    def _get_pred_of_i(self, i, x_values=None):
        if x_values == None:
            feat_values = self.unique[i]
        else:
            feat_values = x_values[i]
        if self.task == "classification":
            pred = self.linear_model.coef_[0, i] * feat_values
        else:
            pred = self.linear_model.coef_[i] * feat_values
        feat_values = feat_values.to(self.device)
        for regressor, boost_rate in zip(self.regressors, self.boosting_rates):
            pred += (
                boost_rate
                * regressor.predict_single(feat_values.reshape(-1, 1), i).squeeze()
            ).cpu()
        return feat_values, pred

    def get_shape_functions_as_dict(self, x_values=None):
        shape_functions = []
        for i, feat_name in enumerate(self.feature_names):
            # set datatype to numerical if i is smaller than the number of numerical columns
            datatype = "numerical" if i < self.n_numerical_cols else "categorical"
            feat_values, pred = self._get_pred_of_i(i, x_values)

            if datatype == "numerical":
                shape_functions.append(
                    {
                        "name": feat_name,
                        "datatype": datatype,
                        "x": feat_values.cpu().numpy(),
                        "y": pred.numpy(),
                        "avg_effect": float(torch.mean(torch.abs(pred))),
                        "hist": {
                            # make this list for eaysier handling and plotting
                            "counts": self.hist[i].hist.cpu().tolist(),
                            "edges": self.hist[i].bin_edges.cpu().tolist(),
                        },
                    }
                )
            else:
                class_name = feat_name.split("_")[-1]
                shape_functions.append(
                    {
                        "name": feat_name.rsplit("_", 1)[0],
                        "datatype": datatype,
                        "x": [class_name],
                        "y": [pred.numpy()[1]],
                        "avg_effect": float(torch.mean(torch.abs(pred))),
                        "hist": {
                            "counts": [self.hist[i][0][-1].cpu().tolist()],
                            "classes": [class_name],
                        },
                    }
                )

        final_shape_functions = {}
        for shape_function in shape_functions:
            name = shape_function["name"]
            # if the feature is cateogrical we need to add the dropped class to the existing shape function
            if name in final_shape_functions.keys():
                final_shape_functions[name]["x"].extend(shape_function["x"])
                final_shape_functions[name]["y"].extend(shape_function["y"])
                final_shape_functions[name]["avg_effect"] += shape_function[
                    "avg_effect"
                ]
                final_shape_functions[name]["hist"]["counts"].extend(
                    shape_function["hist"]["counts"]
                )
                final_shape_functions[name]["hist"]["classes"].extend(
                    shape_function["hist"]["classes"]
                )
            # if the feature is numerical or categorical and not in the dict yet we just add it to the final shape functions
            else:
                final_shape_functions[name] = shape_function

        # if we have dropped features we need to add them to the final shape functions
        num_rows = self.raw_X.shape[0]
        for name, function in final_shape_functions.items():
            if final_shape_functions[name]["datatype"] == "categorical":
                #class_name = str(self.dropped_features[name])
                #final_shape_functions[name]["x"].append(class_name)
                #final_shape_functions[name]["y"].append(0)  # droped class effect is 0

                final_shape_functions[name]["hist"]["counts"].append(
                    num_rows - np.sum(final_shape_functions[name]["hist"]["counts"])
                )
                final_shape_functions[name]["hist"]["classes"].append(class_name)

        return final_shape_functions

    def plot_single(
        self,
        plot_by_list=None,
        show_n=5,
        scaler_dict=None,
        max_cat_plotted=4,
        max_plots_per_row=3,
    ):
        """ """
        # get shapefunctions
        shape_functions_raw = self.get_shape_functions_as_dict()

        # get names/keys to extract
        if plot_by_list is not None:
            show_n = len(plot_by_list)
            keys = plot_by_list
        else:
            keys = shape_functions_raw.keys()

        # convert to list
        shape_function_list = [shape_functions_raw[name] for name in keys]

        # sort shape functions by effect strength
        sorted_shape_functions = sorted(
            shape_function_list, reverse=True, key=lambda x: x["avg_effect"]
        )

        # redeuce list of shape function to required size
        top_k = sorted_shape_functions[:show_n]

        # set up a grid
        n_rows = int(np.ceil(len(top_k) / max_plots_per_row))
        n_cols = min(len(top_k), max_plots_per_row)

        # So the actual total rows = 2 * n_rows
        total_rows = 2 * n_rows
        total_cols = n_cols

        # create height ratios for the grid
        height_ratios = [4, 1] * n_rows  # shape is 4 histogram is 1

        # set up figure
        plt.close(fig="shape functions")
        fig, axs = plt.subplots(
            total_rows,
            total_cols,
            figsize=(12, 4 * n_rows),  # tune as you like
            gridspec_kw={
                "height_ratios": height_ratios,
                "hspace": 0.5,
                "wspace": 0.4,
            },
            # gridspec_kw={"height_ratios": [5, 1]},
            num="Shape functions",
        )

        # Force axs to be 2D if it is not already
        axs = axs.reshape(total_rows, total_cols)

        def _inverse_transform_x_if_needed(shape_func, scaler_dict):
            """
            Inversely transform the shape function's x and y values and histogram edges
            if:
            1) shape_func is numeric, AND
            2) shape_func['name'] is in scaler_dict
            """
            # If no scaler_dict is provided, just return as-is
            if scaler_dict is None:
                return shape_func

            # if y is in scaler_dict, inverse-transform
            if "y" in scaler_dict:
                scaler_func = scaler_dict["y"]
                shape_func["y"] = np.array(
                    scale_func(np.array(shape_func["y"]).reshape(-1, 1))
                )

            # Check if in scaler_dict
            if shape_func["name"] in scaler_dict:
                scaler_func = scaler_dict[shape_func["name"]]

                # Inverse-transform x-values
                x_arr = np.array(shape_func["x"]).reshape(-1, 1)
                x_inv = scaler_func(x_arr)
                shape_func["x"] = np.array(x_inv).ravel()

                # Inverse-transform histogram edges
                edges_arr = np.array(shape_func["hist"]["edges"]).reshape(-1, 1)
                edges_inv = scaler_func(edges_arr)
                shape_func["hist"]["edges"] = np.array(edges_inv).ravel()

            return shape_func

        # helper function to plot numerical shape
        def _plot_numeric(ax_top, ax_bottom, shape_function):
            shape_function = _inverse_transform_x_if_needed(shape_function, scaler_dict)
            # print(shape_function["x"])
            # print(shape_function["y"])
            # print(shape_function["hist"]["edges"])
            sns.lineplot(
                x=shape_function["x"],
                y=shape_function["y"],
                ax=ax_top,
                linewidth=3,
                color="darkblue",
            )
            ax_top.axhline(y=0, color="grey", linestyle="--")
            ax_bottom.bar(
                shape_function["hist"]["edges"][:-1],
                shape_function["hist"]["counts"],
                width=1,
                color="darkblue",
            )
            ax_bottom.get_xaxis().set_visible(False)

        # helper function to plot categorical shape
        def _plot_categorical(ax_top, ax_bottom, shape_function):
            shape_function = _inverse_transform_x_if_needed(shape_function, scaler_dict)
            ax_top.bar(
                x=shape_function["x"],
                height=shape_function["y"],
                width=1,
                color="darkblue",
            )
            ax_top.axhline(y=0, color="grey", linestyle="--")

            # set xticks and labels
            ax_top.set_xticks(np.arange(len(shape_function["x"])))
            ax_top.set_xticklabels(shape_function["x"], rotation=70)

            ax_bottom.bar(
                x=shape_function["hist"]["classes"],
                height=shape_function["hist"]["counts"],
                width=1,
                color="darkblue",
            )
            ax_bottom.get_xaxis().set_visible(False)

        # main loop
        # print(f"n_rows: {n_rows}, n_cols: {n_cols}")
        for i, shape_function in enumerate(top_k):
            # determine postion
            row = i // n_cols
            col = i % n_cols

            # print(f"i:{i}, row: {row}, col: {col}")

            # get axes
            ax_top = axs[2 * row, col]
            ax_bottom = axs[2 * row + 1, col]

            if shape_function["datatype"] == "numerical":
                _plot_numeric(ax_top, ax_bottom, shape_function)
                # Align x-axes for numeric only
                ax_bottom.set_xlim(ax_top.get_xlim())
            else:
                _plot_categorical(ax_top, ax_bottom, shape_function)

            # add a title
            ax_top.set_title(
                f"{shape_function['name']}: {shape_function['avg_effect']:.2f}"
            )

        # remove empty axes
        for i in range(show_n, n_cols * n_rows):
            row = i // n_cols
            col = i % n_cols
            axs[row * 2, col].axis("off")
            axs[row * 2 + 1, col].axis("off")

        plt.show()

    def plot_learning(self):
        """
        Plot the training and the validation losses over time (i.e., for the sequence of learned
        ELMs)
        """
        fig, axs = plt.subplots(1, 1, figsize=(16, 8))
        fig.axes[0].plot(
            np.arange(len(self.train_losses)), self.train_losses, label="Train"
        )
        fig.axes[0].plot(np.arange(len(self.val_losses)), self.val_losses, label="Val")
        if len(self.test_losses) > 0:
            fig.axes[0].plot(
                np.arange(len(self.test_losses)), self.test_losses, label="Test"
            )
        plt.legend()
        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import make_circles, make_regression
    X, y = make_regression(1000, 4, n_informative=4, noise=10., random_state=42)
    #y = X[:,0]**2 # Quadratic (Non-monotonic) Orginal code
    #y = np.sin(X[:, 3]) # Sinusoidal (Non-monotonic)
    y = -3 * X[:, 1] + 5 # monotonic decreasing
    #y = (X[:, 0] > 0).astype(float) # monotonic increasing step function
    #y = np.random.randn(X.shape[0]) # random noise
    X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    #X["cat_test"] = np.random.choice(
    #    ["A", "B", "C", "D"], X.shape[0], p=[0.2, 0.2, 0.1, 0.5]
    #)
    #X["cat_test_2"] = np.random.choice(
    #    ["E", "F", "G", "H"], X.shape[0], p=[0.8, 0.1, 0.05, 0.05]
    #)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    start = time.time()
    m = IGANN(task="regression", 
              n_estimators=5, 
              monotonicity={"1":1,"2": -1}, 
              verbose=1)
    m.fit(pd.DataFrame(X_train), y_train)
    end = time.time()
    print(end - start)
    m.plot_single(show_n=6, max_cat_plotted=4)
