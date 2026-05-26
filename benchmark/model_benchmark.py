#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhysRes vs Baseline Models - Complete Benchmark
包含: Linear AR, MLP, ESN, LSTM 对比
"""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
try:
    import matplotlib.pyplot as plt  # Optional; benchmark training does not require plotting.
except ModuleNotFoundError:
    plt = None
try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ModuleNotFoundError:
    class StandardScaler:
        def fit(self, X):
            X = np.asarray(X, dtype=float)
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.scale_[self.scale_ < 1e-12] = 1.0
            return self

        def transform(self, X):
            X = np.asarray(X, dtype=float)
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)

        def inverse_transform(self, X):
            X = np.asarray(X, dtype=float)
            return X * self.scale_ + self.mean_

    class Ridge:
        def __init__(self, alpha=1e-5, fit_intercept=True):
            self.alpha = alpha
            self.fit_intercept = fit_intercept

        def fit(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            if self.fit_intercept:
                X_aug = np.column_stack([X, np.ones(len(X))])
                reg = np.eye(X_aug.shape[1]) * self.alpha
                reg[-1, -1] = 0.0
            else:
                X_aug = X
                reg = np.eye(X_aug.shape[1]) * self.alpha
            A = X_aug.T @ X_aug + reg
            b = X_aug.T @ y
            try:
                weights = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                weights = np.linalg.pinv(A) @ b
            if self.fit_intercept:
                self.coef_ = weights[:-1]
                self.intercept_ = float(weights[-1])
            else:
                self.coef_ = weights
                self.intercept_ = 0.0
            return self

        def predict(self, X):
            X = np.asarray(X, dtype=float)
            return X @ self.coef_ + self.intercept_

    def mean_squared_error(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return np.mean((y_true - y_pred) ** 2)

    def mean_absolute_error(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return np.mean(np.abs(y_true - y_pred))

    def r2_score(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ==================== 配置 ====================
TRAIN_RATIO = 0.7
WARMUP = 10
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
ROOT = Path(__file__).resolve().parent

# ==================== 数据加载 ====================
print("=" * 70)
print("       PhysRes vs Baseline Models - Complete Benchmark")
print("=" * 70)

df = pd.read_csv(ROOT / 'train_data.csv')
print(f"\n[1] Data loaded: {len(df)} samples")

U = df['u'].values.astype(float)
P = df['P'].values.astype(float)
Tp = df['Tp'].values.astype(float)
Tl = df['Tl'].values.astype(float)

# 7:3划分
split_idx = int(len(df) * TRAIN_RATIO)
print(f"[2] Chronological split: train {split_idx}, test {len(df)-split_idx}")
print(f"[3] Warmup: first {WARMUP} test samples excluded from evaluation")

# 变量范围
P_range = P.max() - P.min()
Tp_range = Tp.max() - Tp.min()
Tl_range = Tl.max() - Tl.min()

print(f"\n[4] Variable ranges:")
print(f"    P:  {P.min():.1f} - {P.max():.1f} hPa (range={P_range:.1f})")
print(f"    Tp: {Tp.min():.1f} - {Tp.max():.1f} °C (range={Tp_range:.1f})")
print(f"    Tl: {Tl.min():.1f} - {Tl.max():.1f} °C (range={Tl_range:.1f})")


# ==================== 模型定义 ====================

class PhysResModel:
    """Physical Reservoir Computing Model"""
    def __init__(self, N=200, lamda=1, alpha=1.0):
        self.N = N
        self.lamda = lamda
        self.alpha = alpha
        self.Win = None
        self.Wout = None
        self.bias = None
        self.x_state = None
        self.params = {}
        self.n_hyperparams = 3  # N, lamda, alpha
        
    def _normalize(self, x, xmin, xmax):
        r = xmax - xmin
        return 2 * (x - xmin) / r - 1 if r > 1e-8 else 0.0
    
    def _sigma(self, zeta):
        zeta = np.clip(zeta, -500, 500)
        sigmoid = 1 / (1 + np.exp(-self.alpha * zeta))
        if self.lamda == 0:
            return sigmoid
        result = np.zeros_like(sigmoid)
        if self.lamda < len(sigmoid):
            result[self.lamda:] = sigmoid[:-self.lamda]
        return result
    
    def fit(self, X, y, ridge_alpha=1e-5):
        L = len(X)
        input_dim = X.shape[1]
        
        # 保存归一化参数
        self.params['X_min'] = X.min(axis=0)
        self.params['X_max'] = X.max(axis=0)
        
        # 归一化
        X_norm = np.zeros_like(X)
        for i in range(input_dim):
            X_norm[:, i] = self._normalize(X[:, i], self.params['X_min'][i], self.params['X_max'][i])
        
        # 初始化输入权重
        np.random.seed(RANDOM_SEED)
        self.Win = np.random.uniform(-0.5, 0.5, (input_dim, self.N))
        
        # 计算储备池状态
        X_states = np.zeros((L, self.N))
        for t in range(1, L):
            operand = np.dot(X_norm[t], self.Win) + X_states[t-1]
            X_states[t] = self._sigma(operand)
        
        # Ridge回归（舍去预热点）
        reg = Ridge(alpha=ridge_alpha, fit_intercept=True)
        reg.fit(X_states[WARMUP:], y[WARMUP:])
        
        self.Wout = reg.coef_
        self.bias = reg.intercept_
        self.x_state = X_states[-1].copy()
        
        return X_states
    
    def predict_sequence(self, X, init_state=None):
        L = len(X)
        input_dim = X.shape[1]
        
        X_norm = np.zeros_like(X)
        for i in range(input_dim):
            X_norm[:, i] = self._normalize(X[:, i], self.params['X_min'][i], self.params['X_max'][i])
        
        X_states = np.zeros((L, self.N))
        if init_state is not None:
            X_states[0] = init_state
        
        for t in range(1, L):
            operand = np.dot(X_norm[t], self.Win) + X_states[t-1]
            X_states[t] = self._sigma(operand)
        
        y_pred = np.dot(X_states, self.Wout) + self.bias
        return y_pred, X_states[-1]
    
    def predict_one(self, x, state):
        x_norm = np.zeros(len(x))
        for i in range(len(x)):
            x_norm[i] = self._normalize(x[i], self.params['X_min'][i], self.params['X_max'][i])
        
        operand = np.dot(x_norm, self.Win) + state
        new_state = self._sigma(operand)
        y_pred = np.dot(new_state, self.Wout) + self.bias
        return y_pred, new_state
    
    def get_params_count(self):
        return self.Win.size + self.Wout.size + 1


class LinearARModel:
    """Linear Auto-Regressive Model"""
    def __init__(self, lag=5):
        self.lag = lag
        self.model = Ridge(alpha=1e-5)
        self.scaler = StandardScaler()
        self.n_hyperparams = 1  # lag
        
    def fit(self, X, y):
        L = len(X)
        features = []
        targets = []
        
        for t in range(self.lag, L):
            feat = list(X[t])
            for i in range(1, self.lag + 1):
                feat.append(y[t-i])
            features.append(feat)
            targets.append(y[t])
        
        features = np.array(features)
        targets = np.array(targets)
        
        features_norm = self.scaler.fit_transform(features)
        self.model.fit(features_norm, targets)
        
    def predict_sequence(self, X, y_init):
        L = len(X)
        y_pred = np.zeros(L)
        y_pred[:self.lag] = y_init[-self.lag:]
        
        for t in range(self.lag, L):
            feat = list(X[t])
            for i in range(1, self.lag + 1):
                feat.append(y_pred[t-i])
            feat = np.array(feat).reshape(1, -1)
            feat_norm = self.scaler.transform(feat)
            y_pred[t] = self.model.predict(feat_norm)[0]
            
            # 防止发散
            if np.abs(y_pred[t]) > 1e6:
                y_pred[t:] = y_pred[t-1]
                break
        
        return y_pred
    
    def predict_one(self, x, y_history):
        feat = list(x)
        for i in range(self.lag):
            feat.append(y_history[-(i+1)])
        feat = np.array(feat).reshape(1, -1)
        feat_norm = self.scaler.transform(feat)
        return self.model.predict(feat_norm)[0]
    
    def get_params_count(self):
        return self.model.coef_.size + 1


class MLPModel:
    """Multi-Layer Perceptron"""
    def __init__(self, hidden_sizes=(64, 32), max_iter=500):
        self.hidden_sizes = hidden_sizes
        self.max_iter = max_iter
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.weights = []
        self.biases = []
        self.n_hyperparams = 3  # hidden_sizes (2) + max_iter
        
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _init_weights(self, layer_sizes):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def _forward(self, X):
        a = X
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self._relu(z)
        return np.dot(a, self.weights[-1]) + self.biases[-1]
    
    def fit(self, X, y):
        X_norm = self.scaler_X.fit_transform(X)
        y_norm = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        input_dim = X.shape[1]
        layer_sizes = [input_dim] + list(self.hidden_sizes) + [1]
        
        np.random.seed(RANDOM_SEED)
        self._init_weights(layer_sizes)
        
        # 梯度下降训练
        lr = 0.001
        batch_size = 32
        n_samples = len(X_norm)
        
        for epoch in range(self.max_iter):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch = X_norm[batch_idx]
                y_batch = y_norm[batch_idx]
                
                # Forward pass with intermediate activations
                activations = [X_batch]
                a = X_batch
                for i in range(len(self.weights) - 1):
                    z = np.dot(a, self.weights[i]) + self.biases[i]
                    a = self._relu(z)
                    activations.append(a)
                output = np.dot(a, self.weights[-1]) + self.biases[-1]
                
                # Backward pass
                error = output.ravel() - y_batch
                d_output = error.reshape(-1, 1) / len(y_batch)
                
                # Output layer gradients
                dW = np.dot(activations[-1].T, d_output)
                db = np.sum(d_output, axis=0)
                self.weights[-1] -= lr * dW
                self.biases[-1] -= lr * db
                
                # Hidden layers (simplified)
                delta = d_output
                for i in range(len(self.weights) - 2, -1, -1):
                    delta = np.dot(delta, self.weights[i+1].T)
                    delta = delta * (activations[i+1] > 0)  # ReLU derivative
                    dW = np.dot(activations[i].T, delta)
                    db = np.sum(delta, axis=0)
                    self.weights[i] -= lr * dW
                    self.biases[i] -= lr * db
    
    def predict_sequence(self, X):
        X_norm = self.scaler_X.transform(X)
        y_norm = self._forward(X_norm)
        y_pred = self.scaler_y.inverse_transform(y_norm.reshape(-1, 1)).ravel()
        return y_pred
    
    def predict_one(self, x):
        x_norm = self.scaler_X.transform(x.reshape(1, -1))
        y_norm = self._forward(x_norm)
        return self.scaler_y.inverse_transform(y_norm.reshape(-1, 1))[0, 0]
    
    def get_params_count(self):
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return total


class ESNModel:
    """Echo State Network"""
    def __init__(self, N=200, spectral_radius=0.9, input_scaling=0.5, leak_rate=0.3):
        self.N = N
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.Win = None
        self.W = None
        self.Wout = None
        self.bias = None
        self.scaler = StandardScaler()
        self.n_hyperparams = 4  # N, spectral_radius, input_scaling, leak_rate
        
    def fit(self, X, y, ridge_alpha=1e-5):
        L = len(X)
        input_dim = X.shape[1]
        
        X_norm = self.scaler.fit_transform(X)
        
        np.random.seed(RANDOM_SEED)
        self.Win = np.random.uniform(-self.input_scaling, self.input_scaling, (input_dim, self.N))
        
        # 稀疏内部权重矩阵
        W = np.random.randn(self.N, self.N)
        W[np.random.rand(self.N, self.N) > 0.1] = 0  # 90%稀疏
        eigenvalues = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigenvalues))
        if max_eig > 0:
            W = W * (self.spectral_radius / max_eig)
        self.W = W
        
        # 计算储备池状态
        X_states = np.zeros((L, self.N))
        for t in range(1, L):
            pre_activation = np.dot(X_norm[t], self.Win) + np.dot(X_states[t-1], self.W)
            X_states[t] = (1 - self.leak_rate) * X_states[t-1] + self.leak_rate * np.tanh(pre_activation)
        
        # Ridge回归
        reg = Ridge(alpha=ridge_alpha, fit_intercept=True)
        reg.fit(X_states[WARMUP:], y[WARMUP:])
        
        self.Wout = reg.coef_
        self.bias = reg.intercept_
        
        return X_states
    
    def predict_sequence(self, X, init_state=None):
        L = len(X)
        X_norm = self.scaler.transform(X)
        
        X_states = np.zeros((L, self.N))
        if init_state is not None:
            X_states[0] = init_state
        
        for t in range(1, L):
            pre_activation = np.dot(X_norm[t], self.Win) + np.dot(X_states[t-1], self.W)
            X_states[t] = (1 - self.leak_rate) * X_states[t-1] + self.leak_rate * np.tanh(pre_activation)
        
        y_pred = np.dot(X_states, self.Wout) + self.bias
        return y_pred, X_states[-1]
    
    def predict_one(self, x, state):
        x_norm = self.scaler.transform(x.reshape(1, -1)).ravel()
        pre_activation = np.dot(x_norm, self.Win) + np.dot(state, self.W)
        new_state = (1 - self.leak_rate) * state + self.leak_rate * np.tanh(pre_activation)
        y_pred = np.dot(new_state, self.Wout) + self.bias
        return y_pred, new_state
    
    def get_params_count(self):
        return self.Win.size + self.W.size + self.Wout.size + 1


class LSTMModel:
    """Simple LSTM Implementation"""
    def __init__(self, hidden_size=50, seq_len=10):
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.Wf = None  # forget gate
        self.Wi = None  # input gate
        self.Wc = None  # cell gate
        self.Wo = None  # output gate
        self.Wy = None  # output layer
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.n_hyperparams = 2  # hidden_size, seq_len
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _init_weights(self, input_dim):
        np.random.seed(RANDOM_SEED)
        scale = 0.1
        combined_dim = input_dim + self.hidden_size
        
        self.Wf = np.random.randn(combined_dim, self.hidden_size) * scale
        self.bf = np.zeros(self.hidden_size)
        self.Wi = np.random.randn(combined_dim, self.hidden_size) * scale
        self.bi = np.zeros(self.hidden_size)
        self.Wc = np.random.randn(combined_dim, self.hidden_size) * scale
        self.bc = np.zeros(self.hidden_size)
        self.Wo = np.random.randn(combined_dim, self.hidden_size) * scale
        self.bo = np.zeros(self.hidden_size)
        self.Wy = np.random.randn(self.hidden_size, 1) * scale
        self.by = np.zeros(1)
    
    def _lstm_step(self, x, h_prev, c_prev):
        combined = np.concatenate([x, h_prev])
        
        f = self._sigmoid(np.dot(combined, self.Wf) + self.bf)
        i = self._sigmoid(np.dot(combined, self.Wi) + self.bi)
        c_tilde = np.tanh(np.dot(combined, self.Wc) + self.bc)
        c = f * c_prev + i * c_tilde
        o = self._sigmoid(np.dot(combined, self.Wo) + self.bo)
        h = o * np.tanh(c)
        
        return h, c
    
    def fit(self, X, y, epochs=100, lr=0.01):
        L = len(X)
        input_dim = X.shape[1]
        
        X_norm = self.scaler_X.fit_transform(X)
        y_norm = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        self._init_weights(input_dim)
        
        # 简化训练：使用输出层的Ridge回归
        # 先收集所有隐藏状态
        H_all = []
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        
        for t in range(L):
            h, c = self._lstm_step(X_norm[t], h, c)
            H_all.append(h.copy())
        
        H_all = np.array(H_all)
        
        # Ridge回归训练输出层
        reg = Ridge(alpha=1e-5, fit_intercept=True)
        reg.fit(H_all[WARMUP:], y_norm[WARMUP:])
        self.Wy = reg.coef_.reshape(-1, 1)
        self.by = np.array([reg.intercept_])
        
        self.last_h = h.copy()
        self.last_c = c.copy()
    
    def predict_sequence(self, X, init_h=None, init_c=None):
        L = len(X)
        X_norm = self.scaler_X.transform(X)
        
        h = init_h if init_h is not None else np.zeros(self.hidden_size)
        c = init_c if init_c is not None else np.zeros(self.hidden_size)
        
        y_pred_norm = []
        for t in range(L):
            h, c = self._lstm_step(X_norm[t], h, c)
            y_t = np.dot(h, self.Wy) + self.by
            y_pred_norm.append(y_t[0])
        
        y_pred_norm = np.array(y_pred_norm)
        y_pred = self.scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()
        
        return y_pred, h, c
    
    def predict_one(self, x, h, c):
        x_norm = self.scaler_X.transform(x.reshape(1, -1)).ravel()
        h_new, c_new = self._lstm_step(x_norm, h, c)
        y_norm = np.dot(h_new, self.Wy) + self.by
        y_pred = self.scaler_y.inverse_transform(y_norm.reshape(-1, 1))[0, 0]
        return y_pred, h_new, c_new
    
    def get_params_count(self):
        total = 0
        for w in [self.Wf, self.Wi, self.Wc, self.Wo, self.Wy]:
            if w is not None:
                total += w.size
        for b in [self.bf, self.bi, self.bc, self.bo, self.by]:
            if b is not None:
                total += b.size
        return total


# ==================== 评估函数 ====================
def compute_metrics(y_true, y_pred, var_range):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / var_range
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'NRMSE': nrmse, 'R2': r2, 'MAE': mae}


# ==================== 训练和评估 ====================
print("\n" + "=" * 70)
print("                    Training and evaluation")
print("=" * 70)

# 准备数据
# P模型: 输入 [u, P, Tp, Tl]
# T模型: 输入 [u, Tp, Tl]

# 训练集
U_train, P_train, Tp_train, Tl_train = U[:split_idx], P[:split_idx], Tp[:split_idx], Tl[:split_idx]
# 测试集
U_test, P_test, Tp_test, Tl_test = U[split_idx:], P[split_idx:], Tp[split_idx:], Tl[split_idx:]

# 构建特征
X_P_train = np.column_stack([U_train[:-1], P_train[:-1], Tp_train[:-1], Tl_train[:-1]])
y_P_train = P_train[1:]

X_T_train = np.column_stack([U_train[:-1], Tp_train[:-1], Tl_train[:-1]])
y_Tp_train = Tp_train[1:]
y_Tl_train = Tl_train[1:]

X_P_test = np.column_stack([U_test[:-1], P_test[:-1], Tp_test[:-1], Tl_test[:-1]])
y_P_test = P_test[1:]

X_T_test = np.column_stack([U_test[:-1], Tp_test[:-1], Tl_test[:-1]])
y_Tp_test = Tp_test[1:]
y_Tl_test = Tl_test[1:]

# 存储结果
results = {'P': {}, 'Tp': {}, 'Tl': {}}
predictions = {'P': {}, 'Tp': {}, 'Tl': {}}
train_times = {}
infer_times = {}
param_counts = {}
hyperparam_counts = {}

# 变量配置
var_configs = [
    ('P', X_P_train, y_P_train, X_P_test, y_P_test, P_range),
    ('Tp', X_T_train, y_Tp_train, X_T_test, y_Tp_test, Tp_range),
    ('Tl', X_T_train, y_Tl_train, X_T_test, y_Tl_test, Tl_range),
]

model_names = ['PhysRes', 'Linear_AR', 'MLP', 'ESN', 'LSTM']

for var_name, X_train, y_train, X_test, y_test, var_range in var_configs:
    print(f"\n{'='*50}")
    print(f"Training {var_name} prediction models")
    print(f"{'='*50}")
    
    for model_name in model_names:
        print(f"\n  [{model_name}]", end=" ")
        
        try:
            # 创建模型
            if model_name == 'PhysRes':
                model = PhysResModel(N=200, lamda=1, alpha=1.0)
            elif model_name == 'Linear_AR':
                model = LinearARModel(lag=5)
            elif model_name == 'MLP':
                model = MLPModel(hidden_sizes=(64, 32), max_iter=300)
            elif model_name == 'ESN':
                model = ESNModel(N=200, spectral_radius=0.9, input_scaling=0.5, leak_rate=0.3)
            elif model_name == 'LSTM':
                model = LSTMModel(hidden_size=50, seq_len=10)
            
            # 训练
            t_start = time.perf_counter()
            
            if model_name == 'PhysRes':
                train_states = model.fit(X_train, y_train)
                last_state = train_states[-1]
                y_pred, _ = model.predict_sequence(X_test, last_state)
                
            elif model_name == 'Linear_AR':
                model.fit(X_train, y_train)
                y_pred = model.predict_sequence(X_test, y_train)
                
            elif model_name == 'MLP':
                model.fit(X_train, y_train)
                y_pred = model.predict_sequence(X_test)
                
            elif model_name == 'ESN':
                train_states = model.fit(X_train, y_train)
                last_state = train_states[-1]
                y_pred, _ = model.predict_sequence(X_test, last_state)
                
            elif model_name == 'LSTM':
                model.fit(X_train, y_train)
                y_pred, _, _ = model.predict_sequence(X_test, model.last_h, model.last_c)
            
            train_time = time.perf_counter() - t_start
            
            # 单步推理时间
            t_start = time.perf_counter()
            n_infer = 5000
            for _ in range(n_infer):
                if model_name == 'PhysRes':
                    _, _ = model.predict_one(X_test[0], np.zeros(model.N))
                elif model_name == 'Linear_AR':
                    _ = model.predict_one(X_test[0], y_train[-model.lag:])
                elif model_name == 'MLP':
                    _ = model.predict_one(X_test[0])
                elif model_name == 'ESN':
                    _, _ = model.predict_one(X_test[0], np.zeros(model.N))
                elif model_name == 'LSTM':
                    _, _, _ = model.predict_one(X_test[0], np.zeros(model.hidden_size), np.zeros(model.hidden_size))
            infer_time = (time.perf_counter() - t_start) / n_infer * 1000  # ms
            
            # 计算指标（舍去前WARMUP个点）
            y_true_eval = y_test[WARMUP:]
            y_pred_eval = y_pred[WARMUP:]
            
            # 检查是否发散
            if np.any(np.isnan(y_pred_eval)) or np.any(np.abs(y_pred_eval) > 1e6):
                print("Diverged!")
                metrics = {'RMSE': np.inf, 'NRMSE': np.inf, 'R2': -np.inf, 'MAE': np.inf}
            else:
                metrics = compute_metrics(y_true_eval, y_pred_eval, var_range)
            
            results[var_name][model_name] = metrics
            predictions[var_name][model_name] = y_pred
            
            if var_name == 'P':  # 只记录一次
                train_times[model_name] = train_time
                infer_times[model_name] = infer_time
                param_counts[model_name] = model.get_params_count()
                hyperparam_counts[model_name] = model.n_hyperparams
            
            print(f"R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, Train={train_time:.3f}s")
            
        except Exception as e:
            print(f"Error: {e}")
            results[var_name][model_name] = {'RMSE': np.inf, 'NRMSE': np.inf, 'R2': -np.inf, 'MAE': np.inf}


# ==================== 保存真实值用于绘图 ====================
ground_truth = {
    'P': y_P_test,
    'Tp': y_Tp_test,
    'Tl': y_Tl_test,
}

# 保存到文件
np.savez(ROOT / 'benchmark_data.npz',
         y_P_test=y_P_test, y_Tp_test=y_Tp_test, y_Tl_test=y_Tl_test,
         predictions=predictions, results=results,
         train_times=train_times, infer_times=infer_times,
         param_counts=param_counts, hyperparam_counts=hyperparam_counts)

print("\n\n" + "=" * 70)
print("                    Result summary")
print("=" * 70)

# 打印预测精度表格
print("\n[1] Prediction accuracy")
for var_name in ['P', 'Tp', 'Tl']:
    unit = 'hPa' if var_name == 'P' else '°C'
    print(f"\n  {var_name} ({unit}):")
    print(f"  {'Model':<12} {'RMSE':>10} {'NRMSE':>10} {'R²':>10} {'MAE':>10}")
    print("  " + "-" * 54)
    
    for model_name in model_names:
        r = results[var_name][model_name]
        if r['R2'] > -100:  # 排除发散的模型
            print(f"  {model_name:<12} {r['RMSE']:>10.4f} {r['NRMSE']:>10.4f} {r['R2']:>10.4f} {r['MAE']:>10.4f}")
        else:
            print(f"  {model_name:<12} {'Diverged':>10} {'---':>10} {'---':>10} {'---':>10}")

# 打印计算效率和模型复杂度
print("\n[2] Computational efficiency")
print(f"  {'Model':<12} {'Train(s)':>12} {'Infer(ms)':>12}")
print("  " + "-" * 38)
for model_name in model_names:
    print(f"  {model_name:<12} {train_times[model_name]:>12.4f} {infer_times[model_name]:>12.4f}")

print("\n[3] Model complexity")
print(f"  {'Model':<12} {'Params':>12} {'Hyperparams':>12}")
print("  " + "-" * 38)
for model_name in model_names:
    print(f"  {model_name:<12} {param_counts[model_name]:>12} {hyperparam_counts[model_name]:>12}")

# 创建汇总表格
summary_data = []
for var_name in ['P', 'Tp', 'Tl']:
    for model_name in model_names:
        r = results[var_name][model_name]
        summary_data.append({
            'Variable': var_name,
            'Model': model_name,
            'RMSE': r['RMSE'] if r['RMSE'] < 1e6 else np.nan,
            'NRMSE': r['NRMSE'] if r['NRMSE'] < 1e6 else np.nan,
            'R2': r['R2'] if r['R2'] > -100 else np.nan,
            'MAE': r['MAE'] if r['MAE'] < 1e6 else np.nan,
            'Train_time_s': train_times[model_name],
            'Infer_time_ms': infer_times[model_name],
            'Params': param_counts[model_name],
            'Hyperparams': hyperparam_counts[model_name]
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(ROOT / 'benchmark_summary.csv', index=False)
print("\nSaved: benchmark_summary.csv")
