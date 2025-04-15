import numpy as np
import random
import matplotlib.pyplot as plt
from custom_taxi_env import CustomTaxiEnv

# Q-learning parametreleri - optimizasyon için güncellendi
alpha = 0.2          # Öğrenme oranı biraz arttırıldı
gamma = 0.99         # Gelecekteki ödüllerin daha etkili olması için arttırıldı
epsilon_start = 4.0  # Başlangıçta tamamen keşif
epsilon_end = 0.01   # En sonda çok az keşif
epsilon_decay = 0.9995  # Her bölümde azalma oranı
max_steps = 1000     # Bir bölümün maksimum adım sayısı
episodes = 500000     # Toplam eğitim bölümü sayısı

# Environment'ı başlatıyoruz
env = CustomTaxiEnv()

# Q-table'ı sıfırla (her state-action çifti için)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Eğitim döngüsü
episode_rewards = []
episode_lengths = []
success_rate = []
successes = 0

epsilon = epsilon_start  # Başlangıç epsilon değeri

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    # Her bölümü maksimum adım sayısı ile sınırla
    while not done and step < max_steps:
        # Epsilon değerini azalt
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Rastgele aksiyon
        else:
            action = np.argmax(q_table[state])  # En iyi aksiyon
        
        # Yeni durumu al ve ödülü hesapla
        next_state, reward, done, _ = env.step(action)
        
        # Q-table'ı güncelle (Q-learning güncelleme kuralı)
        best_next_action = np.argmax(q_table[next_state])
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * q_table[next_state, best_next_action])
        
        state = next_state
        total_reward += reward
        step += 1
        
        # Eğer görev başarıyla tamamlandıysa
        if done and reward > 0:
            successes += 1
    
    # Epsilon azaltma
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    episode_rewards.append(total_reward)
    episode_lengths.append(step)
    
    # Son 100 bölümdeki başarı oranını hesapla
    if episode < 100:
        success_rate.append(successes / (episode + 1))
    else:
        success_rate.append(successes / 100)
        if episode % 100 == 0:
            successes = 0  # Her 100 bölümde sıfırla
    
    if (episode + 1) % 100 == 0:
        print(f"Bölüm {episode + 1}/{episodes} - Toplam Ödül: {total_reward} - Epsilon: {epsilon:.4f} - Adım: {step}")
        print(f"Son 100 Bölümdeki Başarı Oranı: {success_rate[-1]:.2f}")

# Eğitim bittiğinde Q-table'ı kaydedelim
np.save("q_table.npy", q_table)

# Eğitim sürecini görselleştirelim
plt.figure(figsize=(15, 10))

# 1. Toplam Ödül Grafiği - Hareketli ortalama ile
plt.subplot(3, 1, 1)
plt.plot(episode_rewards, alpha=0.3, color='blue')
# 100'lük hareketli ortalama
rolling_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
plt.plot(rolling_avg, color='red', linewidth=2)
plt.xlabel('Bölüm')
plt.ylabel('Toplam Ödül')
plt.title('Q-learning Eğitim Ödülleri (Kırmızı: 100-Bölüm Hareketli Ortalama)')

# 2. Bölüm Uzunluğu Grafiği
plt.subplot(3, 1, 2)
plt.plot(episode_lengths, alpha=0.3, color='green')
# 100'lük hareketli ortalama
rolling_length = np.convolve(episode_lengths, np.ones(100)/100, mode='valid')
plt.plot(rolling_length, color='red', linewidth=2)
plt.xlabel('Bölüm')
plt.ylabel('Adım Sayısı')
plt.title('Bölüm Uzunlukları (Kırmızı: 100-Bölüm Hareketli Ortalama)')

# 3. Başarı Oranı Grafiği
plt.subplot(3, 1, 3)
plt.plot(success_rate, color='purple')
plt.xlabel('Bölüm')
plt.ylabel('Başarı Oranı')
plt.title('Son 100 Bölümdeki Başarı Oranı')

plt.tight_layout()
plt.savefig('q_learning_results.png')  # Grafiği kaydet
plt.show()

# Test et
def test_agent(env, q_table, num_episodes=10):
    """Eğitilmiş ajanı test et"""
    total_rewards = []
    total_steps = []
    success_count = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print("\nYeni Test Bölümü Başladı:")
        env.render()  # Başlangıç durumunu göster
        
        while not done and steps < 100:  # Maksimum 100 adım
            action = np.argmax(q_table[state])  # En iyi aksiyonu seç
            next_state, reward, done, _ = env.step(action)
            
            # Aksiyon isimlerini göster
            action_names = ["Güney", "Kuzey", "Doğu", "Batı", "Al", "Bırak"]
            print(f"Adım {steps+1}: {action_names[action]} aksiyonu seçildi, Ödül: {reward}")
            
            env.render()  # Her adımı göster
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done and reward > 0:
                success_count += 1
                print("Görev Başarıyla Tamamlandı!")
        
        if not done:
            print("Maksimum adım sayısına ulaşıldı!")
        
        total_rewards.append(total_reward)
        total_steps.append(steps)
    
    print(f"\nTest Sonuçları ({num_episodes} bölüm):")
    print(f"Ortalama Ödül: {np.mean(total_rewards):.2f}")
    print(f"Ortalama Adım: {np.mean(total_steps):.2f}")
    print(f"Başarı Oranı: {success_count/num_episodes:.2f}")

print("\nEğitim tamamlandı. Ajanı test ediyorum...")
test_agent(env, q_table, num_episodes=5)