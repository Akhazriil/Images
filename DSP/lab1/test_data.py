import pickle

with open('data_files/dsp.pkl', 'rb') as f:
    data = pickle.load(f)

print("=== IMU Force (accelerometer) ===")
imu_f = data['imu_f']
print(f"Тип: {type(imu_f)}")
print(f"Атрибуты: {[attr for attr in dir(imu_f) if not attr.startswith('_')]}")

if hasattr(imu_f, '__dict__'):
    print(f"__dict__: {imu_f.__dict__}")

# Попробуем найти данные
for attr in ['data', 'values', 'measurements', 'measurements_data']:
    if hasattr(imu_f, attr):
        val = getattr(imu_f, attr)
        print(f"imu_f.{attr}: тип={type(val)}, форма={getattr(val, 'shape', 'N/A')}")

print("\n=== IMU Omega (gyroscope) ===")
imu_w = data['imu_w']
print(f"Тип: {type(imu_w)}")

for attr in ['data', 'values', 'measurements', 'measurements_data']:
    if hasattr(imu_w, attr):
        val = getattr(imu_w, attr)
        print(f"imu_w.{attr}: тип={type(val)}, форма={getattr(val, 'shape', 'N/A')}")