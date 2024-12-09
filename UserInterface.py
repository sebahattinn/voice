import tkinter as tk
from tkinter import messagebox
from tkinter import font
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import librosa
import threading

class AudioRecorder:

    def __init__(self, root):
        self.root = root
        self.root.title("Speaker Recognition")
        self.is_recording = False
        self.frames = []
        self.stop_identification = False  # Konuşmacı tanıma işlemini durdurmak için bayrak

        button_frame = tk.Frame(root)
        button_frame.pack(padx=10, pady=10)

        button_style = {
            "font": ("Arial", 12, "normal"),
            "bg": "#4CAF50",
            "fg": "#FFFFFF",
            "activebackground": "#45a049",
            "activeforeground": "#FFFFFF",
            "relief": "raised",
            "bd": 3,
            "width": 15
        }

        self.start_button = tk.Button(button_frame, text="Kayıt Başlat", command=self.start_recording, **button_style)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.stop_button = tk.Button(button_frame, text="Kayıt Durdur", command=self.stop_recording, **button_style)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.process_button = tk.Button(button_frame, text="Kayıdı İşle", command=self.process_recording, **button_style)
        self.process_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.exit_button = tk.Button(button_frame, text="Çıkış", command=self.close_program, **button_style)  # Çıkış butonu
        self.exit_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.custom_font = font.Font(family="Times New Roman", size=15, weight="normal")

        self.signal_plot = plt.figure(figsize=(6, 3))
        self.ax = self.signal_plot.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.signal_plot, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.X)
        
        self.histogram_plot = plt.figure(figsize=(6, 2))
        self.hist_ax = self.histogram_plot.add_subplot(111)
        self.hist_canvas = FigureCanvasTkAgg(self.histogram_plot, master=root)
        self.hist_canvas.get_tk_widget().pack(fill=tk.X)

        self.info_text = tk.Text(root, height=5, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X)

        self.info_text.tag_configure("header", font=("Arial", 12, "bold"), foreground="#4CAF50")
        self.info_text.tag_configure("content", font=("Arial", 12, "normal"), foreground="#000000")

        # Eğitilmiş modeli yükleme
        model_kayit_yolu = r'C:\Users\sebah\voice\Data\Model.pkl'
        self.model = joblib.load(model_kayit_yolu)

        self.sinif_isimleri = ['Sebahattin Tegi', 'Hüseyin Tasgeldi', 'Eren Öztürk', 'Arda Yemci']

        # Mikrofondan ses almak için gerekli parametreler
        self.saniye_basina_ornek = 44100  # Örnekleme hızı (örneğin, 44100 Hz)
        self.saniye = 5  # 5 saniyelik ses al
        self.kanal_sayisi = 1  # Tek kanallı ses

        # Yabancı kişi tanımlama
        self.known_speakers = set(self.sinif_isimleri)  # Bilinen konuşmacıların kümesi

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.stop_identification = False  # Konuşmacı tanıma işlemi durdurma bayrağını sıfırla
            self.frames = []
            self.stream = sd.InputStream(callback=self.callback, channels=1, samplerate=44100)
            self.stream.start()
            self.root.after(100, self.update_ui)
            # speaker_identification işlemini ayrı bir iş parçacığında çalıştırma
            print("Kayıt Başladı")

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.stop_identification = True
            self.stream.stop()
            self.plot_signal()
            self.plot_histogram()  # Kayıt durduğunda histogramı çiz
            self.save_recording()

    def update_info_text(self, tahmin, transcript, kelime_sayisi):
        # Bilgi metnini güncelle ve özel fontu uygula
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        self.info_text.insert(tk.END, "Konuşmacı: ", "header")
        self.info_text.insert(tk.END, f"{tahmin}\n", "content")
        
        self.info_text.insert(tk.END, "Transcript: ", "header")
        self.info_text.insert(tk.END, f"{transcript}\n", "content")
        
        self.info_text.insert(tk.END, "Kelime Sayısı: ", "header")
        self.info_text.insert(tk.END, f"{kelime_sayisi}\n", "content")
        
        self.info_text.config(state=tk.DISABLED)  # Metni düzenlenemez hale getir

    def callback(self, indata, frames, time, status):
        self.frames.append(indata.copy())
        
    def update_ui(self):
        if self.is_recording:
            self.plot_signal()
            self.plot_histogram()  # Histogramı güncelle
            self.root.after(100, self.update_ui)

    def save_recording(self):
        fs = 44100  # Örnekleme frekansı
        audio_data = np.concatenate(self.frames, axis=0)
        wav.write("kayit.wav", fs, audio_data)
        
        # Convert to PCM format
        sound = AudioSegment.from_wav("kayit.wav")
        sound.export("kayit1_pcm.wav", format="wav")

    def process_recording(self):
        # Ses dosyasını işle ve tahmin et
        file = "kayit1_pcm.wav"
        transcript, kelime_sayisi = self.getWords(file)
        tahmin = self.speaker_identification(file)
        self.update_info_text(tahmin, transcript, kelime_sayisi)

    def getWords(self, file):
        def transcribe_audio(audio_file_path):
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file_path) as source:
                audio = recognizer.record(source)
            try:
                transcript = recognizer.recognize_google(audio, language="tr-TR")
                return transcript
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError as e:
                return f"Could not request results; {e}"

        kelimeler = []

        # Transcribe the WAV file
        transcript = transcribe_audio(file)
            
        kelimeler.extend(transcript.split())
            
        return transcript, len(kelimeler)
        
    def speaker_identification(self, file):
        # Ses dosyasını yükleme ve MFCC özelliklerini çıkarma
        y, sr = librosa.load(file, sr=self.saniye_basina_ornek)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)  # Ortalama MFCC vektörü
        
        # Model üzerinden tahmin yapma
        tahmin_indeksi = self.model.predict(mfcc.reshape(1, -1))[0]
        tahmin_isim = self.sinif_isimleri[tahmin_indeksi]

        # Yabancı kişi kontrolü
        if tahmin_isim not in self.known_speakers:
            return "Yabancı"  # Eğer kişi bilinen konuşmacılar arasında değilse "Yabancı" döndür
        
        return tahmin_isim
    
    def plot_histogram(self):
        audio_data = np.concatenate(self.frames, axis=0)
        self.hist_ax.clear()
        self.hist_ax.hist(audio_data, bins=100, color='b', alpha=0.7)
        self.hist_ax.set_title('Ses Verisi Dağılımı')
        self.hist_ax.set_xlabel('Amplitüd')
        self.hist_ax.set_ylabel('Frekans')
        self.hist_canvas.draw()

    def plot_signal(self):
        audio_data = np.concatenate(self.frames, axis=0)
        self.ax.clear()
        self.ax.plot(np.linspace(0, len(audio_data) / 44100, num=len(audio_data)), audio_data)
        self.ax.set_title('Ses Sinyali')
        self.ax.set_xlabel('Zaman (s)')
        self.ax.set_ylabel('Amplitüd')
        self.canvas.draw()

    def close_program(self):
        # Kayıt yapılıyorsa durdur
        if self.is_recording:
            self.stop_recording()

        # Çıkış için tkinter penceresini kapatma
        self.root.quit()  # Tkinter uygulamasını düzgün bir şekilde sonlandırır.
        self.root.destroy()  # Tkinter penceresini kapatır.


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x700")
    app = AudioRecorder(root)
    root.mainloop()
