import os
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier

def extract_features(dosya_ismi):
    # Dosyayı yükle
    ses, oran = librosa.load(dosya_ismi)

    # Spektral Merkez
    spektral_merkez = np.expand_dims(librosa.feature.spectral_centroid(y=ses, sr=oran)[0], axis=0)

    # Spektral Bant Genişliği
    spektral_bant_genisligi = np.expand_dims(librosa.feature.spectral_bandwidth(y=ses, sr=oran)[0], axis=0)

    # Spektral Rolloff
    spektral_rolloff = np.expand_dims(librosa.feature.spectral_rolloff(y=ses, sr=oran)[0], axis=0)

    # Kök Ortalama Kareleri
    kok_ortalama_kareleri = np.expand_dims(np.mean(librosa.feature.rms(y=ses)), axis=0)

    # Sıfır Geçiş Hızı
    sifir_gecis_hizi = np.expand_dims(np.mean(librosa.feature.zero_crossing_rate(y=ses)), axis=0)

    # Kroma Özellikleri
    kroma = np.expand_dims(np.mean(librosa.feature.chroma_stft(y=ses, sr=oran)), axis=0)

    # Mel-Frekans Cepstral Katsayıları (MFCCs)
    mfcc = librosa.feature.mfcc(y=ses, sr=oran)

    # Tüm özellikleri tek bir dizide döndür
    return np.concatenate([spektral_merkez, spektral_bant_genisligi, spektral_rolloff, kok_ortalama_kareleri, sifir_gecis_hizi, kroma, mfcc], axis=1)


def extract_features_from_directory(dizin):
    özellikler = []
    for dosya in os.listdir(dizin):
        if dosya.endswith(".wav"):
            dosya_özellikleri = extract_features(os.path.join(dizin, dosya))
            özellikler.append(dosya_özellikleri)
    return np.array(özellikler)

# Dosya yolu
dizin = "C:/Users/Utku Sina/OneDrive/Masaüstü/MGC/archive/Data/genres_original/rock" #kendi dosya yolunu gir

# Dosya listesini oluştur
dosya_listesi = [dosya for dosya in os.listdir(dizin) if dosya.endswith(".wav")]

# Boş bir liste başlatın ve tüm özellik vektörlerini içerecek
özellikler = []

# Dosya listesindeki her dosya için özellikleri çıkar
for dosya_ismi in dosya_listesi:
    dosya_yolu = os.path.join(dizin, dosya_ismi)
    dosya_özellikleri = extract_features(dosya_yolu)
    özellikler.append(dosya_özellikleri)

# Özellik listesini NumPy dizisi olarak dönüştür
egitim_verileri = np.array(özellikler)

# Eğitim verilerini ve etiketlerini kullanarak K-NN sınıflandırıcısını eğitin
knn = KNeighborsClassifier(n_neighbors=1)
egitim_etiketleri = np.array([0, 1, 2])  # Örnek etiketler
knn.fit(egitim_verileri, egitim_etiketleri)

# Diğer dosyaları analiz etmek için yine bir dosya listesi kullanarak özellikleri çıkarın
diger_dosya_listesi = [dosya for dosya in os.listdir(dizin) if dosya.endswith(".wav")]
for dosya_ismi in diger_dosya_listesi:
    dosya_yolu = os.path.join(dizin, dosya_ismi)
    dosya_özellikleri = extract_features(dosya_yolu)
    etiket = knn.predict([dosya_özellikleri])[0]
    print(f"{dosya_ismi} dosyası en yakın olduğu tür: {etiket}")
