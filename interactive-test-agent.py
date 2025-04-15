import numpy as np
import time
import os
from custom_taxi_env import CustomTaxiEnv

def clear_screen():
    """Terminal ekranını temizler"""
    os.system('cls' if os.name == 'nt' else 'clear')

def wait_for_enter():
    """Kullanıcı Enter tuşuna basana kadar bekler"""
    input("\nDevam etmek için ENTER tuşuna basın...")

def test_agent_interactive(q_table_path, num_episodes=5, step_by_step=True):
    """
    Eğitilmiş Q-learning modelini adım adım test et
    
    Parametreler:
    q_table_path (str): Kaydedilmiş Q-tablosunun dosya yolu
    num_episodes (int): Test edilecek bölüm sayısı
    step_by_step (bool): Her adımda kullanıcının Enter tuşuna basmasını bekle
    """
    # Q-tablosunu yükle
    print(f"Q-tablosu yükleniyor: {q_table_path}")
    q_table = np.load(q_table_path)
    print(f"Q-tablosu yüklendi: {q_table.shape} boyutunda")
    
    # Ortamı başlat
    env = CustomTaxiEnv()
    
    # Test metrikleri
    total_rewards = []
    total_steps = []
    success_count = 0
    
    action_names = ["Güney", "Kuzey", "Doğu", "Batı", "Al", "Bırak"]
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        clear_screen()
        print(f"\n===== Test Bölümü {episode+1}/{num_episodes} =====")
        print("Başlangıç Durumu:")
        env.render()
        
        if step_by_step:
            wait_for_enter()
        
        while not done and steps < 200:  # Maksimum 200 adım
            clear_screen()
            print(f"\n===== Test Bölümü {episode+1}/{num_episodes} =====")
            print(f"Adım: {steps+1}")
            
            # En iyi aksiyonu seç
            action = np.argmax(q_table[state])
            print(f"Seçilen aksiyon: {action_names[action]} ({action})")
            
            # Aksiyonu uygula
            next_state, reward, done, _ = env.step(action)
            print(f"Alınan ödül: {reward}")
            print(f"Toplam ödül: {total_reward + reward}")
            print(f"Görev tamamlandı mı: {'Evet' if done else 'Hayır'}")
            
            env.render()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Görevi başarıyla tamamladı mı?
            if done and reward > 0:
                success_count += 1
                print("\n🎉 GÖREV BAŞARIYLA TAMAMLANDI! 🎉")
            elif done:
                print("\n❌ Görev başarısız oldu!")
            
            if steps >= 200:
                print("❌ Maksimum adım sayısı aşıldı!")
                break
                
            if not done and step_by_step:
                wait_for_enter()
        
        total_rewards.append(total_reward)
        total_steps.append(steps)
        
        print(f"\nBölüm {episode+1} sonuçları:")
        print(f"- Toplam ödül: {total_reward}")
        print(f"- Toplam adım: {steps}")
        
        if episode < num_episodes - 1:  # Son bölüm değilse
            wait_for_enter()
    
    # Genel test sonuçları
    clear_screen()
    print("\n===== TEST SONUÇLARI =====")
    print(f"Ortalama ödül: {np.mean(total_rewards):.2f}")
    print(f"Ortalama adım sayısı: {np.mean(total_steps):.2f}")
    print(f"Başarı oranı: {success_count/num_episodes:.2f} ({success_count}/{num_episodes})")
    
    for ep in range(num_episodes):
        print(f"Bölüm {ep+1}: Ödül = {total_rewards[ep]}, Adım = {total_steps[ep]}")
    
    return {
        "mean_reward": np.mean(total_rewards),
        "mean_steps": np.mean(total_steps),
        "success_rate": success_count/num_episodes
    }

if __name__ == "__main__":
    # Kaydedilmiş Q-tablosunun yolu
    q_table_path = "q_table.npy"
    
    try:
        # Eğitilmiş modeli adım adım test et
        print("Q-Learning modelini test ediyorum...")
        results = test_agent_interactive(q_table_path, num_episodes=3, step_by_step=True)
    except FileNotFoundError:
        print(f"Hata: {q_table_path} dosyası bulunamadı!")
        print("Lütfen önce modeli eğittiğinizden ve Q-tablosunun kaydedildiğinden emin olun.")
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {e}")