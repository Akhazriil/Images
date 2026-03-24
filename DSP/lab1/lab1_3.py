import numpy as np
import matplotlib.pyplot as plt
import time

class RBPF_Dipole_Optimized:
    def __init__(self, n_particles, sensors, mu0=4*np.pi*1e-7,
                 lambda_pos=1.0, delta_mom=0.25, 
                 R_noise=None, Q_pos=None, Q_mom=None):
        self.N = n_particles
        self.sensors = np.array(sensors, dtype=np.float64)
        self.L = len(sensors)
        self.mu0 = mu0
        self.lambda_pos = lambda_pos
        self.delta_mom = delta_mom
        
        self.R = R_noise if R_noise is not None else 0.2 * np.eye(self.L, dtype=np.float64)
        self.Q_pos = Q_pos if Q_pos is not None else (lambda_pos**2) * np.eye(2, dtype=np.float64)
        self.Q_mom = Q_mom if Q_mom is not None else (delta_mom**2) * np.eye(2, dtype=np.float64)
        
        self.particles = None
        self.q_mean = None
        self.q_cov = None
        self.weights = None
        
    def _compute_G_vectorized(self, particles):
        """Even more vectorized version"""
        N = particles.shape[0]
        diff = self.sensors[np.newaxis, :, :] - particles[:, np.newaxis, :]
        
        # Add small epsilon for numerical stability
        d2 = np.sum(diff**2, axis=2)
        d3 = np.power(d2 + 1e-10, 1.5)
        
        # Vectorized cross product
        G = np.zeros((N, self.L, 2))
        G[:, :, 0] = -diff[:, :, 1]
        G[:, :, 1] = diff[:, :, 0]
        
        return (self.mu0 / (4*np.pi)) * G / d3[:, :, np.newaxis]
    
    def initialize(self, p_mean, p_cov, q_mean, q_cov):
        """Инициализация фильтра"""
        self.particles = np.random.multivariate_normal(
            np.array(p_mean, dtype=np.float64), 
            np.array(p_cov, dtype=np.float64), 
            self.N
        )
        
        # Создаём массивы формы (N, 2) и (N, 2, 2)
        q_mean_arr = np.array(q_mean, dtype=np.float64).reshape(2)
        q_cov_arr = np.array(q_cov, dtype=np.float64).reshape(2, 2)
        
        self.q_mean = np.empty((self.N, 2), dtype=np.float64)
        self.q_cov = np.empty((self.N, 2, 2), dtype=np.float64)
        
        for n in range(self.N):
            self.q_mean[n] = q_mean_arr.copy()
            self.q_cov[n] = q_cov_arr.copy()
        
        self.weights = np.ones(self.N, dtype=np.float64) / self.N
        
    def predict(self):
        """Шаг предсказания"""
        self.particles += np.random.multivariate_normal(
            np.zeros(2, dtype=np.float64), 
            self.Q_pos, 
            self.N
        )
        # Add mean prediction with random walk
        self.q_mean += np.random.multivariate_normal(
            np.zeros(2, dtype=np.float64),
            self.Q_mom,
            self.N
        )
        self.q_cov += self.Q_mom
        
    def update(self, y):
        """Векторизованный шаг обновления"""
        y = np.array(y, dtype=np.float64).flatten()
        log_weights = np.zeros(self.N, dtype=np.float64)
        
        G_all = self._compute_G_vectorized(self.particles)
        
        for n in range(self.N):
            G = G_all[n]
            
            y_pred = G @ self.q_mean[n]
            nu = y - y_pred
            
            S = G @ self.q_cov[n] @ G.T + self.R
            S_inv = np.linalg.inv(S)
            K = self.q_cov[n] @ G.T @ S_inv
            
            self.q_mean[n] += K @ nu
            self.q_cov[n] = (np.eye(2, dtype=np.float64) - K @ G) @ self.q_cov[n]
            
            sign, logdet = np.linalg.slogdet(S)
            log_weights[n] = -0.5 * (nu.T @ S_inv @ nu + logdet)
        
        log_weights += np.log(self.weights + 1e-300)
        log_weights -= np.max(log_weights)
        self.weights = np.exp(log_weights)
        self.weights /= np.sum(self.weights)
        
    def resample(self, threshold=None):
        """Систематический ресемплинг с адаптивным порогом"""
        if threshold is None:
            threshold = self.N / 2
        
        ESS = 1.0 / np.sum(self.weights**2)
        if ESS < threshold:
            # Systematic resampling as before
            indices = self._systematic_resample()
            
            self.particles = self.particles[indices].copy()
            self.q_mean = self.q_mean[indices].copy()
            self.q_cov = self.q_cov[indices].copy()
            self.weights = np.ones(self.N) / self.N
            
        return ESS

    def _systematic_resample(self):
        """Separate resampling logic"""
        indices = np.zeros(self.N, dtype=int)
        positions = (np.arange(self.N) + np.random.rand()) / self.N
        cumsum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.N:
            if positions[i] < cumsum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices

    def estimate(self):
        """Возвращает взвешенную оценку состояния"""
        p_est = np.average(self.particles, weights=self.weights, axis=0)
        q_est = np.average(self.q_mean, weights=self.weights, axis=0)

        # Простое преобразование без ravel()
        p_est = np.asarray(p_est, dtype=np.float64)
        q_est = np.asarray(q_est, dtype=np.float64)

        return p_est, q_est


# === ГЕНЕРАЦИЯ ДАННЫХ ===
def generate_synthetic_data(T, sensors, true_params, 
                           lambda_pos=1.0, delta_mom=0.25, 
                           R_noise_std=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Пусть имеется L сенсоров(10)
    L = len(sensors)
    # Задание магнитной постоянной
    mu0 = 4*np.pi*1e-7
    
    p_true = np.zeros((T, 2), dtype=np.float64)
    q_true = np.zeros((T, 2), dtype=np.float64)
    y_obs = np.zeros((T, L), dtype=np.float64)
    
    p_true[0] = true_params['p_init']
    q_true[0] = true_params['q_init']
    
    def compute_b(p, q):
        # единичный вектор перпендикулярный плоскости P
        e = np.array([0, 0, 1], dtype=np.float64)
        # Вектор наблюдений
        b = np.zeros(L, dtype=np.float64)

        for j, rj in enumerate(sensors):
            rj_vec = np.array([rj[0], rj[1], rj[2]], dtype=np.float64)
            p_vec = np.array([p[0], p[1], 0], dtype=np.float64)
            q_vec = np.array([q[0], q[1], 0], dtype=np.float64)
            diff = rj_vec - p_vec
            d3 = np.linalg.norm(diff)**3
            b[j] = (mu0/(4*np.pi)) * np.dot(e, np.cross(q_vec, diff )) / d3
        return b
    
    for k in range(1, T):
        p_true[k] = p_true[k-1] + np.random.normal(0, lambda_pos, 2)
        q_true[k] = q_true[k-1] + np.random.normal(0, delta_mom, 2)
        
        b_clean = compute_b(p_true[k], q_true[k])
        y_obs[k] = b_clean + np.random.normal(0, R_noise_std, L)
    
    y_obs[0] = compute_b(p_true[0], q_true[0]) + np.random.normal(0, R_noise_std, L)
    
    return p_true, q_true, y_obs


# === ЗАПУСК ===
T = 50
L = 100
sensors = np.mgrid[0:10:10j, 0:10:10j, 3:4].reshape(3, -1).T

true_params = {
    'p_init': np.array([5.0, 5.0], dtype=np.float64),
    'q_init': np.array([1.0, 0.5], dtype=np.float64),
}

p_true, q_true, y_data = generate_synthetic_data(
    T, sensors, true_params, 
    lambda_pos=0.5, delta_mom=0.1, R_noise_std=0.2, seed=42)

start_time = time.time()

rbpf = RBPF_Dipole_Optimized(
    n_particles=1000,
    sensors=sensors,
    lambda_pos=1.0,
    delta_mom=0.25,
    R_noise=0.2**2 * np.eye(L, dtype=np.float64)
)

rbpf.initialize(
    p_mean=np.array([5.0, 5.0], dtype=np.float64), 
    p_cov=np.diag(np.array([25.0, 25.0], dtype=np.float64)),
    q_mean=np.array([0.0, 0.0], dtype=np.float64), 
    q_cov=np.diag(np.array([4.0, 4.0], dtype=np.float64))
)

# ИСПРАВЛЕНИЕ: Используем списки вместо предвыделенных массивов
p_est_list = []
q_est_list = []
ess_log = np.zeros(T, dtype=np.float64)

for k in range(T):
    if k > 0:
        rbpf.predict()
    rbpf.update(y_data[k])
    ess_log[k] = rbpf.resample()
    p_k, q_k = rbpf.estimate()
    
    # Добавляем в список (как в рабочем коде)
    p_est_list.append(p_k)
    q_est_list.append(q_k)

# Конвертируем в массивы в конце
p_est = np.array(p_est_list, dtype=np.float64)
q_est = np.array(q_est_list, dtype=np.float64)

elapsed_time = time.time() - start_time
print(f"Время выполнения: {elapsed_time:.2f} секунд")
print(f"Форма p_est: {p_est.shape}")
print(f"Форма q_est: {q_est.shape}")
print(f"q_est[:, 1] shape: {q_est[:, 1].shape}")  # Должно быть (50,)

# === ВИЗУАЛИЗАЦИЯ ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.plot(p_true[:, 0], p_true[:, 1], 'g-', label='True', linewidth=2)
ax.plot(p_est[:, 0], p_est[:, 1], 'r--', label='RBPF estimate', linewidth=1.5)
ax.scatter(sensors[:, 0], sensors[:, 1], c='gray', s=10, alpha=0.3, label='Sensors')
ax.set_xlabel('p₁'); ax.set_ylabel('p₂')
ax.set_title('Dipole Position Tracking'); ax.legend(); ax.grid(True)

ax = axes[0, 1]
# Исправление: объединяем все линии в одном вызове plot
ax.plot(q_true[:, 0], 'g-', label='True q₁', linewidth=2)
ax.plot(q_true[:, 1], 'g--', label='True q₂', linewidth=2)
ax.plot(q_est[:, 0], 'r-', label='Est q₁', linewidth=1.5)
ax.plot(q_est[:, 1], 'r--', label='Est q₂', linewidth=1.5)
ax.set_xlabel('Time step')
ax.set_ylabel('Moment')
ax.set_title('Dipole Moment Estimation')
ax.legend()
ax.grid(True)

ax = axes[1, 0]
error_p = np.linalg.norm(p_true - p_est, axis=1)
ax.plot(error_p, color='purple')
ax.set_xlabel('Time step')
ax.set_ylabel('Position error')
ax.set_title('Tracking Error')
ax.grid(True)

ax = axes[1, 1]
ax.plot(ess_log, color='brown')
ax.axhline(y=1000/2, color='red', linestyle=':', label='Threshold')
ax.set_xlabel('Time step')
ax.set_ylabel('Effective Sample Size')
ax.set_title('Particle Filter Diagnostics')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()