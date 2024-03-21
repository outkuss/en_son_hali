

# Aşağıdaki kod, classify_music fonksiyonunu düzenlenmiş halde gösterir. 
# Bu fonksiyon, müziğin özelliklerini hesaplar ve k-NN sınıflandırıcısını 
# kullanarak müziğin türünü belirler. Sonucu, müziğin en yakın olduğu türü 
# yazdırır.
# Bu kod, numpy ve sklearn kütüphanelerini kullanarak müziğin özelliklerini 
# hesaplar ve k-NN sınıflandırıcısını kullanarak müziğin türünü belirler. 
# Sonucu, müziğin en yakın olduğu türü yazdırır.

# Not: Bu kod, extract_features fonksiyonunun önceden tanımlanmış olduğunu 
# varsayar. Bu fonksiyon, müziğin ses verilerini okuyup özelliklerini 
# hesaplayan kodun üst kısmında tanımlanmıştır.

# Not 2: Bu örnek, numpy ve sklearn kütüphanelerinin standart 
# kütüphanelerden alındığına dikkat edin. Kullanıcılar, kendi 
# sınıflandırıcılarını kullanabilir veya numpy ve sklearn kütüphanelerinin 
# parametrelerini değiştirebilirler.


import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# K-NN sınıflandırıcısını başlat
knn = KNeighborsClassifier(n_neighbors=1)

def classify_music(dosya_ismi):
    # Dosyanın özelliklerini hesapla
    dosya_özellikleri = extract_features(dosya_ismi)

    # Eğitim verilerini yükle
    egitim_verileri = np.load("egitim_verileri.npy")

    # Eğitim etiketlerini yükle
    egitim_etiketleri = np.load("egitim_etiketleri.npy")

    # Eğitim verilerine özellikleri ekle
    knn.fit(egitim_verileri, egitim_etiketleri)

    # Özelliklerle en yakın etiketi bul
    etiket = knn.predict([dosya_özellikleri])[0]

    # Etiketi yazdır
    print("Bu müzik en yakın olduğu tür {}'dur.".format(egitim_etiketleri[etiket]))