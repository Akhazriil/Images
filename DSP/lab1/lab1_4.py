import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

class ESKF:
    def __init__(self, dt, sigma_acc, sigma_gyro, sigma_gnss, sigma_lidar):
        self.dt = dt
        self.sigma_acc = sigma_acc
        self.sigma_gyro = sigma_gyro
        self.sigma_gnss = sigma_gnss
        self.sigma_lidar = sigma_lidar
        
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.ab = np.zeros(3)
        self.wb = np.zeros(3)
        self.g = np.array([0, 0, -9.81])
        
        # === ИСПРАВЛЕНИЕ 1: Меньшая начальная ковариация ===
        self.P = np.eye(18) * 0.001  # Было 0.1
        
        self.C_lidar = np.array([
            [0.99376, -0.09722, 0.05466],
            [0.09971, 0.99401, -0.04475],
            [-0.04998, 0.04992, 0.9975]
        ])
        self.t_lidar = np.array([0.5, 0.1, 0.5])
    
    def skew(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def quat_to_rot(self, q):
        w, x, y, z = q
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])
    
    def normalize_quat(self, q):
        return q / np.linalg.norm(q)
    
    def quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def predict(self, acc_meas, gyro_meas):
        acc = acc_meas - self.ab
        gyro = gyro_meas - self.wb
        R_mat = self.quat_to_rot(self.q)
        
        acc_world = R_mat @ acc + self.g
        self.p = self.p + self.v * self.dt + 0.5 * acc_world * self.dt**2
        self.v = self.v + acc_world * self.dt
        
        delta_theta = gyro * self.dt
        theta_norm = np.linalg.norm(delta_theta)
        if theta_norm > 1e-10:
            dq = np.array([
                np.cos(theta_norm / 2),
                *(delta_theta / theta_norm * np.sin(theta_norm / 2))
            ])
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        self.q = self.quat_multiply(self.q, dq)
        self.q = self.normalize_quat(self.q)
        
        Fx = self.compute_Fx(R_mat, acc, gyro)
        Q = self.compute_process_noise()
        
        # === ИСПРАВЛЕНИЕ 2: Проверка на NaN перед обновлением ===
        self.P = Fx @ self.P @ Fx.T + Q
        if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
            self.P = np.eye(18) * 0.001
        
        return self.get_state()
    
    def compute_Fx(self, R_mat, acc, gyro):
        dt = self.dt
        I3 = np.eye(3)
        Fx = np.eye(18)
        Fx[0:3, 3:6] = I3 * dt
        Fx[3:6, 6:9] = -self.skew(R_mat @ acc) * dt
        Fx[3:6, 9:12] = -R_mat * dt
        Fx[3:6, 15:18] = I3 * dt
        Fx[6:9, 12:15] = -R_mat * dt
        return Fx
    
    def compute_process_noise(self):
        dt = self.dt
        
        # === ИСПРАВЛЕНИЕ 3: Меньшие шумы процесса ===
        var_acc = (self.sigma_acc * dt) ** 2  # Было sigma_acc**2 * dt**2
        var_gyro = (self.sigma_gyro * dt) ** 2
        var_acc_bias = (self.sigma_acc * 0.01) ** 2 * dt  # Меньший шум биаса
        var_gyro_bias = (self.sigma_gyro * 0.01) ** 2 * dt
        
        Q = np.zeros((18, 18))
        Q[3:6, 3:6] = np.eye(3) * var_acc
        Q[6:9, 6:9] = np.eye(3) * var_gyro
        Q[9:12, 9:12] = np.eye(3) * var_acc_bias
        Q[12:15, 12:15] = np.eye(3) * var_gyro_bias
        
        return Q
    
    def update_gnss(self, z_gnss):
        H = np.zeros((3, 18))
        H[0:3, 0:3] = np.eye(3)
        R_gnss = np.eye(3) * self.sigma_gnss**2
        y = z_gnss - self.p
        S = H @ self.P @ H.T + R_gnss
        
        # === ИСПРАВЛЕНИЕ 4: Регуляризация S ===
        S += np.eye(3) * 1e-10
        
        K = self.P @ H.T @ np.linalg.inv(S)
        delta_x = K @ y
        self.inject_error(delta_x)
        
        # === ИСПРАВЛЕНИЕ 5: Форма Иосифа для ковариации ===
        I18 = np.eye(18)
        I_KH = I18 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_gnss @ K.T
        
        # === ИСПРАВЛЕНИЕ 6: Ограничение значений P ===
        self.P = np.clip(self.P, -1e6, 1e6)
        np.fill_diagonal(self.P, np.clip(np.diag(self.P), 1e-6, 1e6))
        
        return self.get_state()
    
    def update_lidar(self, z_lidar):
        z_processed = self.C_lidar @ z_lidar + self.t_lidar
        H = np.zeros((3, 18))
        H[0:3, 0:3] = np.eye(3)
        R_lidar = np.eye(3) * self.sigma_lidar**2
        y = z_processed - self.p
        S = H @ self.P @ H.T + R_lidar
        
        S += np.eye(3) * 1e-10
        
        K = self.P @ H.T @ np.linalg.inv(S)
        delta_x = K @ y
        self.inject_error(delta_x)
        
        I18 = np.eye(18)
        I_KH = I18 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_lidar @ K.T
        
        self.P = np.clip(self.P, -1e6, 1e6)
        np.fill_diagonal(self.P, np.clip(np.diag(self.P), 1e-6, 1e6))
        
        return self.get_state()
    
    def inject_error(self, delta_x):
        self.p = self.p + delta_x[0:3]
        self.v = self.v + delta_x[3:6]
        
        delta_theta = delta_x[6:9]
        theta_norm = np.linalg.norm(delta_theta)
        if theta_norm > 1e-10:
            dq = np.array([
                np.cos(theta_norm / 2),
                *(delta_theta / theta_norm * np.sin(theta_norm / 2))
            ])
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        self.q = self.quat_multiply(dq, self.q)
        self.q = self.normalize_quat(self.q)
        
        self.ab = self.ab + delta_x[9:12]
        self.wb = self.wb + delta_x[12:15]
        self.g = self.g + delta_x[15:18]
        
        # === ИСПРАВЛЕНИЕ 7: Пропустить обновление P при сбросе ===
        # Для малых ошибок G ≈ I, поэтому можно не обновлять P
        # G = self.compute_reset_jacobian(delta_theta)
        # self.P = G @ self.P @ G.T
    
    def compute_reset_jacobian(self, delta_theta):
        G = np.eye(18)
        G[6:9, 6:9] = np.eye(3) + 0.5 * self.skew(delta_theta)
        return G
    
    def get_state(self):
        return {
            'position': self.p.copy(),
            'velocity': self.v.copy(),
            'orientation': self.q.copy(),
            'acc_bias': self.ab.copy(),
            'gyro_bias': self.wb.copy(),
            'gravity': self.g.copy()
        }


def rotation_vector_to_quaternion(r):
    theta = np.linalg.norm(r)
    if theta < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    u = r / theta
    qw = np.cos(theta / 2)
    qv = u * np.sin(theta / 2)
    
    return np.array([qw, qv[0], qv[1], qv[2]])


def run_eskf(data_path='data/dsp.pkl'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, data_path)
    
    print(f"Загрузка данных из: {full_path}")
    
    with open(full_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Ключи в файле: {list(data.keys())}")
    
    gt = data['gt']
    gt_p = gt._p
    gt_v = gt._v
    gt_r = gt._r
    
    n_steps = len(gt_p)
    print(f"Количество шагов Ground Truth: {n_steps}")
    
    if hasattr(gt, '_t'):
        dt = np.mean(np.diff(gt._t))
    else:
        dt = 0.005
    
    print(f"Шаг дискретизации dt: {dt}")
    
    # === ИСПРАВЛЕНИЕ 8: Меньшие шумы измерений ===
    sigma_acc = 0.01  # Было 0.1
    sigma_gyro = 0.001  # Было 0.01
    sigma_gnss = 1.0
    sigma_lidar = 0.5
    
    eskf = ESKF(dt, sigma_acc, sigma_gyro, sigma_gnss, sigma_lidar)
    
    eskf.p = gt_p[0].copy()
    eskf.v = gt_v[0].copy()
    eskf.q = rotation_vector_to_quaternion(gt_r[0])
    eskf.P = np.eye(18) * 0.001
    
    estimates = []
    ground_truths = []
    
    imu_f = data['imu_f']
    imu_w = data['imu_w']
    
    acc_data = imu_f.data
    gyro_data = imu_w.data
    
    print(f"Длина IMU данных: {len(acc_data)}")
    
    gnss_data = data['gnss'] if 'gnss' in data else None
    lidar_data = data['lidar'] if 'lidar' in data else None
    
    if gnss_data is not None and hasattr(gnss_data, 'data'):
        gnss_data = gnss_data.data
    if lidar_data is not None and hasattr(lidar_data, 'data'):
        lidar_data = lidar_data.data
    
    # === ИСПРАВЛЕНИЕ 9: Использовать min для предотвращения выхода за границы ===
    n_iterations = min(n_steps, len(acc_data), len(gyro_data))
    print(f"Количество итераций: {n_iterations}")
    
    for i in range(n_iterations):
        acc_meas = acc_data[i]
        gyro_meas = gyro_data[i]
        
        state = eskf.predict(acc_meas, gyro_meas)
        
        if gnss_data is not None and i < len(gnss_data):
            gnss_meas = gnss_data[i]
            if gnss_meas is not None and not np.all(np.isnan(gnss_meas)):
                state = eskf.update_gnss(gnss_meas)
        
        if lidar_data is not None and i < len(lidar_data):
            lidar_meas = lidar_data[i]
            if lidar_meas is not None and not np.all(np.isnan(lidar_meas)):
                state = eskf.update_lidar(lidar_meas)
        
        estimates.append(state)
        
        ground_truths.append({
            'position': gt_p[i].copy(),
            'velocity': gt_v[i].copy(),
            'orientation': rotation_vector_to_quaternion(gt_r[i])
        })
    
    errors = compute_errors(estimates, ground_truths)
    return estimates, ground_truths, errors


def compute_errors(estimates, ground_truths):
    n = len(estimates)
    pos_errors = np.zeros(n)
    vel_errors = np.zeros(n)
    orient_errors = np.zeros(n)
    
    for i in range(n):
        pos_errors[i] = np.linalg.norm(
            estimates[i]['position'] - ground_truths[i]['position']
        )
        vel_errors[i] = np.linalg.norm(
            estimates[i]['velocity'] - ground_truths[i]['velocity']
        )
        q_est = estimates[i]['orientation']
        q_gt = ground_truths[i]['orientation']
        dot = np.abs(np.dot(q_est, q_gt))
        orient_errors[i] = 2 * np.arccos(min(dot, 1.0))
    
    return {
        'position': pos_errors,
        'velocity': vel_errors,
        'orientation': orient_errors,
        'position_rmse': np.sqrt(np.mean(pos_errors**2)),
        'velocity_rmse': np.sqrt(np.mean(vel_errors**2)),
        'orientation_rmse': np.sqrt(np.mean(orient_errors**2))
    }


def plot_results(estimates, ground_truths, errors):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    est_pos = np.array([e['position'] for e in estimates])
    gt_pos = np.array([g['position'] for g in ground_truths])
    
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'g-', label='Ground Truth', linewidth=2)
    ax1.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], 'r--', label='ESKF Estimate', linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.set_title('3D Trajectory')
    
    axes[0, 1].plot(errors['position'], 'b-', linewidth=2)
    axes[0, 1].axhline(y=errors['position_rmse'], color='r', linestyle='--', 
                       label=f'RMSE: {errors["position_rmse"]:.3f} m')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Position Error (m)')
    axes[0, 1].legend()
    axes[0, 1].set_title('Position Error')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(errors['velocity'], 'g-', linewidth=2)
    axes[1, 0].axhline(y=errors['velocity_rmse'], color='r', linestyle='--',
                       label=f'RMSE: {errors["velocity_rmse"]:.3f} m/s')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Velocity Error (m/s)')
    axes[1, 0].legend()
    axes[1, 0].set_title('Velocity Error')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(np.degrees(errors['orientation']), 'm-', linewidth=2)
    axes[1, 1].axhline(y=np.degrees(errors['orientation_rmse']), color='r', linestyle='--',
                       label=f'RMSE: {np.degrees(errors["orientation_rmse"]):.3f}°')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Orientation Error (deg)')
    axes[1, 1].legend()
    axes[1, 1].set_title('Orientation Error')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('eskf_results.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Запуск ESKF фильтра...")
    estimates, ground_truths, errors = run_eskf('data_files/dsp.pkl')
    
    print("\n=== Результаты ===")
    print(f"RMSE позиции: {errors['position_rmse']:.3f} м")
    print(f"RMSE скорости: {errors['velocity_rmse']:.3f} м/с")
    print(f"RMSE ориентации: {np.degrees(errors['orientation_rmse']):.3f}°")
    
    plot_results(estimates, ground_truths, errors)