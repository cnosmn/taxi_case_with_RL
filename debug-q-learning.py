import numpy as np
import time
import os
import random
from custom_taxi_env import CustomTaxiEnv

def debug_q_table(q_table_path):
    """
    Q-tablosunu analiz ederek sorunları tespit eder
    """
    try:
        q_table = np.load(q_table_path)
        print(f"Q-tablosu başarıyla yüklendi: {q_table.shape} boyutu")
        
        # Q-tablosunun analizi
        non_zero = np.count_nonzero(q_table)
        total_elements = q_table.size
        percentage = (non_zero / total_elements) * 100
        
        print(f"Toplam eleman sayısı: {total_elements}")
        print(f"Sıfır olmayan eleman sayısı: {non_zero}")
        print(f"Dolu eleman yüzdesi: {percentage:.2f}%")
        
        if percentage < 1:
            print("UYARI: Q-tablosu çok az dolu. Eğitim muhtemelen etkili değil.")
        
        # Min-max değerleri
        min_val = q_table.min()
        max_val = q_table.max()
        print(f"Minimum Q-değeri: {min_val}")
        print(f"Maksimum Q-değeri: {max_val}")
        
        if max_val <= 0:
            print("UYARI: Tüm Q-değerleri negatif veya sıfır. Ajan hiçbir olumlu deneyim kazanmamış.")
        
        # Ortalama değerler
        avg_val = q_table.mean()
        print(f"Ortalama Q-değeri: {avg_val}")
        
        # Rastgele 5 durum için en iyi aksiyonları göster
        print("\nRastgele durumlar için en iyi aksiyonlar:")
        action_names = ["Güney", "Kuzey", "Doğu", "Batı", "Al", "Bırak"]
        
        for _ in range(5):
            state = random.randint(0, q_table.shape[0]-1)
            best_action = np.argmax(q_table[state])
            q_values = q_table[state]
            print(f"Durum {state}: En iyi aksiyon = {action_names[best_action]} ({best_action})")
            print(f"  Q-değerleri: {q_values}")
        
        return q_table
    except Exception as e:
        print(f"Q-tablosu analizi sırasında hata: {e}")
        return None

def train_simple_agent(episodes=1000, save_path="simple_q_table.npy"):
    """
    Basit bir ajan eğitir - sorun tespiti için kullanılabilir
    """
    env = CustomTaxiEnv()
    
    # Q-learning parametreleri
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    
    # Q-tablosunu oluştur
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    print(f"Basit ajan eğitiliyor: {episodes} bölüm...")
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-greedy politika
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            # Aksiyonu uygula
            next_state, reward, done, _ = env.step(action)
            
            # Q-tablosunu güncelle
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )
            
            state = next_state
            total_reward += reward
        
        if (episode + 1) % 100 == 0:
            print(f"Bölüm {episode+1}/{episodes}, Toplam Ödül: {total_reward}")
    
    print(f"Eğitim tamamlandı. Q-tablosu kaydediliyor: {save_path}")
    np.save(save_path, q_table)
    return q_table

def test_with_random_actions(num_episodes=3):
    """
    Rastgele aksiyonlarla test yaparak ortamın çalıştığını doğrular
    """
    env = CustomTaxiEnv()
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        print(f"\n===== Rastgele Test Bölümü {episode+1} =====")
        print("Başlangıç durumu:")
        env.render()
        input("Devam etmek için ENTER'a basın...")
        
        while not done and steps < 50:  # Maksimum 50 adım
            action = env.action_space.sample()  # Rastgele aksiyon
            
            action_names = ["Güney", "Kuzey", "Doğu", "Batı", "Al", "Bırak"]
            print(f"\nAdım {steps+1}: {action_names[action]} ({action}) aksiyonu seçildi")
            
            next_state, reward, done, _ = env.step(action)
            print(f"Ödül: {reward}")
            
            env.render()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                if reward > 0:
                    print("\n🎉 GÖREV BAŞARIYLA TAMAMLANDI! 🎉")
                else:
                    print("\n❌ Görev başarısız oldu!")
            
            if not done:
                input("Devam etmek için ENTER'a basın...")
        
        print(f"\nBölüm {episode+1} sonuçları:")
        print(f"- Toplam ödül: {total_reward}")
        print(f"- Toplam adım: {steps}")
        input("Bir sonraki bölüm için ENTER'a basın...")

def print_debug_menu():
    """Debug menüsünü yazdırır"""
    print("\n===== Q-LEARNING DEBUG MENÜSÜ =====")
    print("1. Q-tablosunu analiz et")
    print("2. Ortamı rastgele aksiyonlarla test et")
    print("3. Basit bir ajan eğit (1000 bölüm)")
    print("4. Basit eğitilmiş ajanı test et")
    print("5. Çıkış")
    return input("Seçiminiz (1-5): ")

if __name__ == "__main__":
    q_table_path = "q_table.npy"
    simple_q_table_path = "simple_q_table.npy"
    
    while True:
        choice = print_debug_menu()
        
        if choice == '1':
            q_table = debug_q_table(q_table_path)
        
        elif choice == '2':
            test_with_random_actions()
        
        elif choice == '3':
            train_simple_agent(episodes=1000, save_path=simple_q_table_path)
        
        elif choice == '4':
            try:
                from interactive_test_agent import test_agent_interactive
                test_agent_interactive(simple_q_table_path, num_episodes=2)
            except ImportError:
                print("interactive_test_agent.py dosyası bulunamadı!")
                print("Lütfen önceki kod parçasını interactive_test_agent.py olarak kaydedin.")
        
        elif choice == '5':
            print("Program sonlandırılıyor...")
            break
        
        else:
            print("Geçersiz seçim. Lütfen 1-5 arasında bir sayı girin.")