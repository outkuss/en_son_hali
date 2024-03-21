


# Bu kod, os modülünü kullanarak verilen dizindeki tüm müzik dosyalarının 
# özelliklerini hesaplar. Sonucu, NumPy dizisi olarak döndürür.

# Not: Bu kod, extract_features fonksiyonunun önceden tanımlanmış olduğunu 
# varsayar. Bu fonksiyon, müziğin ses verilerini okuyup özelliklerini 
# hesaplayan kodun üst kısmında tanımlanmıştır.

# Not 2: Bu örnek, os modülünün standart kütüphanelerden alındığına dikkat 
# edin. Kullanıcılar, farklı kütüphaneler ve yöntemleri kullanarak 
# dizindeki dosyaları gezebilir.



import os
import numpy as np

def extract_features_from_directory(dizin):
    # Boş bir liste başlatın ve tüm özellik vektörlerini içerecek
    özellikler = []

    # Dizindeki her dosyaya gez
    for dosya in os.listdir(dizin):
        # Eğer dosya bir mp3 dosyasıysa
        if dosya.endswith(".mp3"):
            # Bu dosyanın özelliklerini hesapla
            dosya_özellikleri = extract_features(os.path.join(dizin, dosya))
            özellikler.append(dosya_özellikleri)

    # Özellik listesini NumPy dizisi olarak dönüştür
    özellikler = np.array(özellikler)

    # Özellik dizisini döndür
    return özellikler