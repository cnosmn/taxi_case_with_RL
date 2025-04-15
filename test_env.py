import numpy as np
from custom_taxi_env import CustomTaxiEnv

# Q-table'ı yükle
q_table = np.load("q_table.npy")

# Environment'ı başlat
env = CustomTaxiEnv()

# Test süreci: Q-table ile en iyi aksiyonları seçerek taksiyi hareket ettir
state = env.reset()
done = False
total_reward = 0

# Test için 100 adım (veya istediğiniz kadar)
for step in range(100):
    action = np.argmax(q_table[state])  # Q-table'a göre en iyi aksiyon
    state, reward, done, _ = env.step(action)  # aksiyonu uygula
    env.render()  # Durumu görselleştir
    total_reward += reward
    if done:
        print(f"Test tamamlandı! Toplam ödül: {total_reward}")
        break
