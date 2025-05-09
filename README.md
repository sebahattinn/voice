# Voice Recognition Project

Bu proje, ses verilerini kullanarak konuşmacı tanıma ve ses analizi gerçekleştiren bir makine öğrenmesi uygulamasıdır. Kullanıcı arayüzü üzerinden ses kaydı alınabilir, kayıtlar makine öğrenmesi modeliyle analiz edilip tanımlanabilir.

## Özellikler
- Gerçek zamanlı ses kaydı
- Konuşmacı tanıma
- Ses dalga formu ve özellik çıkarımı
- Kullanıcı arayüzü ile kolay kullanım

## Gereksinimler

Bu projeyi çalıştırmadan önce aşağıdaki bağımlılıkların yüklü olduğundan emin olun:

```bash
pip install -r requirements.txt
```

Ek olarak, `ffmpeg` kütüphanesinin de sistemde kurulu olması gerekmektedir:

- Ubuntu:
  ```bash
  sudo apt-get install ffmpeg
  ```
- Windows:
  [FFmpeg Download](https://ffmpeg.org/download.html) üzerinden indirip sistem PATH'ine eklemeniz gerekmektedir.

## Kurulum

1. Projeyi klonlayın:
    ```bash
    git clone https://github.com/sebahattinn/voice.git
    cd voice
    ```

2. Gerekli bağımlılıkları yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

3. Kullanıcı arayüzünü başlatın:
    ```bash
    python UserInterface.py
    ```

## Kullanım

- Uygulama arayüzü açıldığında, 'Record' butonuna basarak ses kaydı yapabilirsiniz.
- Kaydedilen ses, makine öğrenmesi modeli tarafından analiz edilip sonuç ekranda gösterilecektir.
- Sonuçlar, konuşmacının kim olduğu ve ses dalga formu ile birlikte görüntülenir.

## Model Eğitimi

Projede kullanılan model, MFCC (Mel Frequency Cepstral Coefficients) özniteliklerini kullanarak konuşmacıları tanır. Eğitim sürecinde, ses dosyalarından çıkarılan MFCC öznitelikleri kullanılarak bir sınıflandırıcı (MLP Classifier) eğitilmiştir.

Eğer modeli yeniden eğitmek isterseniz, `AudioML4.ipynb` dosyasını çalıştırabilirsiniz.

## Katkıda Bulunma
Katkıda bulunmak isterseniz, lütfen bir pull request açın. Her türlü geri bildirime açığım!

## Lisans
Bu proje MIT Lisansı ile lisanslanmıştır.
