# Aşağıdaki kod, extract_features fonksiyonunu türkçe komutlarıyla gösterir.
# Bu fonksiyon, müziğin ses verilerini okuyup, ses düzeyi, sessizlik, 
# frekans gibi temel özellikleri ve diğer özellikleri hesaplar.

# Bu kod, librosa kütüphanesini kullanarak müziğin ses verilerini okur. 
# Sonra, müziğin ses düzeyi, sessizlik, frekans gibi temel özelliklerini
# hesaplar ve diğer özellikleri hesaplar.
# Sonucu, NumPy dizisi olarak döndürür.
# Not: Bu kod, librosa kütüphanesini kullanarak müziğin ses verilerini 
# okuyup özelliklerini hesaplar. Kullanıcılar, farklı kütüphaneler ve 
# yöntemleri kullanarak müziğin ses verilerini okuyabilir ve özelliklerini 
# hesaplayabilir.
# Not 2: Bu örnek, librosa kütüphanesinin standart kütüphanelerden 
# alındığına dikkat edin. Kullanıcılar, kendi sınıflandırıcılarını 
# kullanabilir veya librosa kütüphanesinin parametrelerini değiştirebilirler.

import librosa
import numpy as np

def extract_features(dosya_ismi):
    # Dosyayı yükle
    ses, oran = librosa.load(dosya_ismi)

    # Spektral Merkez
    spektral_merkez = librosa.feature.spectral_centroid(ses, sr=oran)
    spektral_merkez = np.mean(spektral_merkez)

    # Spektral Bant Genişliği
    spektral_bant_genisligi = librosa.feature.spectral_bandwidth(ses, sr=oran)
    spektral_bant_genisligi = np.mean(spektral_bant_genisligi)

    # Spektral Rolloff
    spektral_rolloff = librosa.feature.spectral_rolloff(ses, sr=oran)
    spektral_rolloff = np.mean(spektral_rolloff)

    # Kök Ortalama Kareleri
    kok_ortalama_kareleri = np.mean(librosa.feature.rms(ses))

    # Sıfır Geçiş Hızı
    sifir_gecis_hizi = np.mean(librosa.feature.zero_crossing_rate(ses))

    # Kroma Özellikleri
    kroma = np.mean(librosa.feature.chroma_stft(ses, sr=oran))

    # Mel-Frekans Cepstral Katsayıları (MFCCs)
    mfcc = np.mean(librosa.feature.mfcc(ses, sr=oran), eksen=1)

    # Tüm özellikleri tek bir dizide döndür
    return np.array([spektral_merkez, spektral_bant_genisligi, spektral_rolloff, kok_ortalama_kareleri, sifir_gecis_hizi, kroma, mfcc])