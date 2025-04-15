import gym
from gym import spaces
import numpy as np
import random

# Bu sınıf, OpenAI Gym framework'ünde özelleştirilmiş bir taksi ortamı oluşturur
# Bu ortamda, bir taksi bir yolcuyu belirli bir konumdan alıp başka bir konuma bırakmalıdır
class CustomTaxiEnv(gym.Env):
    def __init__(self):
        # Ana Gym.Env sınıfının constructor'ını çağır
        super(CustomTaxiEnv, self).__init__()
        # Izgara boyutunu 10x10 olarak belirle
        self.grid_size = 10

        # Mümkün aksiyonları tanımla:
        # 0 = güney (aşağı), 1 = kuzey (yukarı), 2 = doğu (sağ), 3 = batı (sol), 4 = yolcu alma, 5 = yolcu bırakma
        self.action_space = spaces.Discrete(6)

        # Gözlem uzayını tanımla: 
        # (taksi_satır, taksi_sütun, yolcu_satır, yolcu_sütun, hedef_satır, hedef_sütun, yolcu_taksidemi)
        obs_size = (self.grid_size, self.grid_size, self.grid_size, self.grid_size, self.grid_size, self.grid_size, 2)
        # Tüm olası gözlemlerin toplam sayısını hesapla ve Discrete uzay olarak tanımla
        self.observation_space = spaces.Discrete(np.prod(obs_size))

        # Taksilerin girmesinin yasak olduğu hücreler (engeller)
        # 10x10 grid için blocked cells (kırmızı hücreler)
        self.blocked_cells = [
            # En üst satır (0. satır)
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
            
            # 3. satırdaki engeller (kırmızı blok)
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 9),
            
            # 5. satırda sağdaki engel
            (5, 9),
            
            # 6. satırdaki engel
            (6, 5),
            
            # 7. satırdaki sağ engel
            (7, 9),
            
            # 9. satır (en alt satır)
            (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)
        ]

        # Adım sayacı - maksimum adım kontrolü için
        self.steps_taken = 0
        # Maksimum adım sayısı (episode uzunluğu) - büyük ızgara için yüksek değer kullanılıyor
        self.max_steps = 2000

        # Ortamı başlangıç durumuna getir
        self.reset()

    # Engelsiz rastgele bir hücre döndüren yardımcı fonksiyon
    def get_random_cell(self):
        while True:
            # Rastgele bir satır ve sütun seç
            row = random.randint(0, self.grid_size - 1)
            col = random.randint(0, self.grid_size - 1)
            # Eğer seçilen hücre engelli değilse, bu hücreyi döndür
            if (row, col) not in self.blocked_cells:
                return row, col

    # Ortamı başlangıç durumuna getiren metod
    def reset(self):
        # Taksinin, yolcunun ve hedefin konumlarını rastgele belirle
        self.taxi_row, self.taxi_col = self.get_random_cell()
        self.passenger_row, self.passenger_col = self.get_random_cell()
        self.destination_row, self.destination_col = self.get_random_cell()
        # Başlangıçta yolcu taksinin içinde değil
        self.passenger_in_taxi = False
        # Adım sayacını sıfırla
        self.steps_taken = 0
        # Başlangıç durumunu kodlanmış şekilde döndür
        return self.encode()

    # Durum bilgisini tek bir tamsayıya kodlayan metod
    def encode(self):
        # Tüm durum bileşenlerini (taksi konumu, yolcu konumu, hedef, yolcunun taksidemi) 
        # tek bir tamsayıya dönüştür
        i = self.taxi_row
        i *= self.grid_size
        i += self.taxi_col
        i *= self.grid_size
        i += self.passenger_row
        i *= self.grid_size
        i += self.passenger_col
        i *= self.grid_size
        i += self.destination_row
        i *= self.grid_size
        i += self.destination_col
        i *= 2
        i += int(self.passenger_in_taxi)
        return i

    # Kodlanmış durumu bileşenlerine ayıran metod
    def decode(self, i):
        # Tek bir tamsayıyı durum bileşenlerine (taksi konumu, yolcu konumu, hedef, vs.) dönüştür
        out = []
        out.append(i % 2)  # passenger_in_taxi
        i //= 2
        out.append(i % self.grid_size)  # dest_col
        i //= self.grid_size
        out.append(i % self.grid_size)  # dest_row
        i //= self.grid_size
        out.append(i % self.grid_size)  # passenger_col
        i //= self.grid_size
        out.append(i % self.grid_size)  # passenger_row
        i //= self.grid_size
        out.append(i % self.grid_size)  # taxi_col
        i //= self.grid_size
        out.append(i % self.grid_size)  # taxi_row
        # Çıktıyı ters çevir, çünkü en son kodlanan bilgi (taxi_row) en önce gelmelidir
        return reversed(out)

    # Belirli bir aksiyonu uygulayan metod
    def step(self, action):
        # Adım sayacını artır
        self.steps_taken += 1
        
        # Her adımda varsayılan olarak küçük bir ceza (negatif ödül) ver
        # Bu, ajanı hedefine mümkün olduğunca az adımda ulaşmaya teşvik eder
        reward = -0.1
        done = False

        # Maksimum adım sayısına ulaşıldıysa episodu bitir
        if self.steps_taken >= self.max_steps:
            done = True
            return self.encode(), reward, done, {"max_steps_exceeded": True}

        # Aksiyona göre taksinin bir sonraki konumunu hesapla
        next_row, next_col = self.taxi_row, self.taxi_col

        # 0: Güney (aşağı) - satır numarası artar
        if action == 0 and self.taxi_row < self.grid_size - 1:
            next_row += 1
        # 1: Kuzey (yukarı) - satır numarası azalır
        elif action == 1 and self.taxi_row > 0:
            next_row -= 1
        # 2: Doğu (sağ) - sütun numarası artar
        elif action == 2 and self.taxi_col < self.grid_size - 1:
            next_col += 1
        # 3: Batı (sol) - sütun numarası azalır
        elif action == 3 and self.taxi_col > 0:
            next_col -= 1

        # Engelli bir hücreye gitme girişimleri için taksinin konumunu değiştirme
        # Duvar cezası yok, sadece hareket etmeme şeklinde bir "ceza"
        if (next_row, next_col) not in self.blocked_cells:
            self.taxi_row, self.taxi_col = next_row, next_col

        # Hedefe yaklaşmayı teşvik eden ödüllendirme mekanizması
        if not self.passenger_in_taxi:
            # Eğer yolcu taksinin içinde değilse, yolcuya yaklaşmayı ödüllendir
            old_distance = abs(self.taxi_row - next_row) + abs(self.taxi_col - next_col)
            new_distance = abs(self.taxi_row - self.passenger_row) + abs(self.taxi_col - self.passenger_col)
            if new_distance < old_distance:
                reward += 0.05  # Yolcuya yaklaşma ödülü
        else:
            # Eğer yolcu taksinin içindeyse, hedefe yaklaşmayı ödüllendir
            old_distance = abs(self.taxi_row - next_row) + abs(self.taxi_col - next_col)
            new_distance = abs(self.taxi_row - self.destination_row) + abs(self.taxi_col - self.destination_col)
            if new_distance < old_distance:
                reward += 0.05  # Hedefe yaklaşma ödülü

        # Yolcu alma aksiyonu
        if action == 4:  # pickup
            # Eğer taksi yolcunun konumundaysa ve yolcu henüz taksinin içinde değilse
            if not self.passenger_in_taxi and (self.taxi_row, self.taxi_col) == (self.passenger_row, self.passenger_col):
                self.passenger_in_taxi = True
                reward = 10  # Yolcuyu başarıyla alınca büyük ödül
            else:
                reward = -10  # Yanlış yerde yolcu alma girişimi için ceza

        # Yolcu bırakma aksiyonu
        elif action == 5:  # dropoff
            # Eğer yolcu taksinin içindeyse ve taksi hedef konumundaysa
            if self.passenger_in_taxi and (self.taxi_row, self.taxi_col) == (self.destination_row, self.destination_col):
                self.passenger_in_taxi = False
                reward = 20  # Yolcuyu doğru hedefte bırakınca daha büyük ödül
                done = True  # Görev tamamlandı, episodu bitir
            else:
                reward = -10  # Yanlış yerde yolcu bırakma girişimi için ceza

        # Yeni durumu kodlanmış şekilde döndür, ödül bilgisini, episodun bitip bitmediğini ve ek bilgileri içeren sözlüğü döndür
        return self.encode(), reward, done, {}  

    # Ortamın mevcut durumunu görsel olarak gösteren metod
    def render(self):
        # Boş bir ızgara oluştur
        grid = [[" ." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Engelli hücreleri işaretle
        for row, col in self.blocked_cells:
            grid[row][col] = "🟥"

        # Eğer yolcu taksinin içinde değilse, yolcunun konumunu işaretle
        if not self.passenger_in_taxi:
            grid[self.passenger_row][self.passenger_col] = "👤"
        # Hedef konumunu işaretle
        grid[self.destination_row][self.destination_col] = "🏁"
        
        # Taksiyi, yolcunun durumuna göre farklı şekilde göster
        if self.passenger_in_taxi:
            grid[self.taxi_row][self.taxi_col] = "🚖"  # Yolculu taksi
        else:
            grid[self.taxi_row][self.taxi_col] = "🚕"  # Boş taksi

        # Izgarayı ekrana yazdır
        print("\n".join(["".join(row) for row in grid]))
        # Taksi konumu ve yolcu durumu bilgisini yazdır
        print(f"Konum: ({self.taxi_row}, {self.taxi_col}), Yolcu: {self.passenger_in_taxi}")
        # Yolcu konumu bilgisini yazdır
        print(f"Yolcu konumu: ({self.passenger_row}, {self.passenger_col})")
        # Hedef konumu bilgisini yazdır
        print(f"Hedef: ({self.destination_row}, {self.destination_col})")
        # Şu ana kadar atılan adım sayısını yazdır
        print(f"Adım sayısı: {self.steps_taken}\n")