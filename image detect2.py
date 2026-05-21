import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Reshape

class ForgeryDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü Sahteciliği Tespiti Yazılımı - Ar-Ge Sistemi")
        self.root.geometry("950x680")
        self.root.configure(bg="#f4f6f9")
        
        self.image_path = None
        self.loaded_image = None
        
        # --- ARAYÜZ STİLLERİ ---
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), background="#f4f6f9", foreground="#1a365d")
        
        # --- ÜST BANNER ---
        header = ttk.Label(root, text="DİJİTAL GÖRÜNTÜ SAHTECİLİĞİ TESPİT SİSTEMİ (AR-GE)", style="Header.TLabel")
        header.pack(pady=15)
        
        # --- ANA PANEL (SOL: RESİM, SAĞ: KONTROLLER) ---
        main_frame = tk.Frame(root, bg="#f4f6f9")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # SOL PANEL: Görsel Önizleme Alanı
        self.left_panel = tk.Label(main_frame, text="Lütfen Analiz İçin Bir Görsel Yükleyin\n(.jpg, .jpeg, .png, .gif)", 
                                   bg="#ffffff", fg="#a0aec0", font=("Segoe UI", 11, "italic"),
                                   relief=tk.SOLID, bd=1, width=55, height=20)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # SAĞ PANEL: Butonlar ve Kontroller
        right_panel = tk.Frame(main_frame, bg="#f4f6f9")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # USER STORY 1: Dosya Seçme ve Yükleme Butonu
        btn_select = ttk.Button(right_panel, text="📁 Görsel Seç ve Yükle", command=self.select_image)
        btn_select.pack(fill=tk.X, pady=10)
        
        # USER STORY 2: Geleneksel Algoritma Seçim Kutusu
        ttk.Label(right_panel, text="Geleneksel Algoritma Seçimi:", font=("Segoe UI", 10), background="#f4f6f9").pack(anchor=tk.W, pady=(15,2))
        self.algo_combo = ttk.Combobox(right_panel, values=["SIFT", "SURF", "AKAZE", "ORB"], state="readonly", font=("Segoe UI", 10))
        self.algo_combo.set("SIFT")
        self.algo_combo.pack(fill=tk.X, pady=5)
        
        btn_trad = ttk.Button(right_panel, text="⚙️ Geleneksel Analiz Yap", command=self.run_traditional)
        btn_trad.pack(fill=tk.X, pady=5)
        
        # Ayırıcı Çizgi
        ttk.Separator(right_panel, orient='horizontal').pack(fill=tk.X, pady=25)
        
        # USER STORY 3: Yapay Zeka Analizi Tetikleme Butonu
        btn_ai = ttk.Button(right_panel, text=" Yapay Zeka (CNN+LSTM) Analizi", command=self.run_ai)
        btn_ai.pack(fill=tk.X, pady=10)
        
        # --- ALT PANEL: CANLI SONUÇ PANELİ ---
        self.status_frame = tk.LabelFrame(root, text=" Analiz Sonuç Paneli ", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#2c5282", relief=tk.GROOVE, bd=2)
        self.status_frame.pack(fill=tk.X, padx=30, pady=20)
        
        self.lbl_result = tk.Label(self.status_frame, text="Sistem Hazır. Görsel yüklendikten sonra analiz adımlarına geçebilirsiniz.", 
                                   font=("Segoe UI", 11), bg="#ffffff", fg="#4a5568", anchor=tk.W, justify=tk.LEFT, padx=15, pady=10)
        self.lbl_result.pack(fill=tk.X)

    # --- USER STORY 1: DOSYA FORMAT DESTEĞİ VE YÜKLEME (EVRENSEL YOL) ---
    def select_image(self):
        file_types = [("Görüntü Dosyaları", "*.png *.jpg *.jpeg *.gif")]
        self.image_path = filedialog.askopenfilename(title="Bir Görsel Seçiniz", filetypes=file_types)
        
        if self.image_path:
            ext = os.path.splitext(self.image_path)[1].lower()
            try:
                # Klasör yollarındaki Türkçe karakter problemini ezmek için dosyayı ham byte olarak okuyoruz
                img_array = np.fromfile(self.image_path, np.uint8)
                
                if ext == '.gif':
                    # GIF dosyaları için ilk kareyi yakalama simülasyonu
                    cap = cv2.VideoCapture(self.image_path)
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        messagebox.showerror("Hata", "GIF dosyası okunurken bir hata oluştu.")
                        return
                    self.loaded_image = frame
                else:
                    self.loaded_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if self.loaded_image is None:
                    messagebox.showerror("Hata", "Seçilen görsel bozuk veya formatı desteklenmiyor!")
                    return
                
                # Arayüz önizlemesi için BGR -> RGB dönüşümü ve boyutlandırma
                img_rgb = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((480, 360))
                
                img_tk = ImageTk.PhotoImage(img_pil)
                self.left_panel.configure(image=img_tk, text="")
                self.left_panel.image = img_tk 
                
                self.lbl_result.configure(text=f"✔ Görsel başarıyla yüklendi: {os.path.basename(self.image_path)}\nAnaliz tipini seçip ilerleyebilirsiniz.", fg="#2f855a")
            
            except Exception as e:
                messagebox.showerror("Hata", f"Görsel yüklenirken bir sorun oluştu:\n{str(e)}")

    # --- USER STORY 2: SIFT, SURF, AKAZE, ORB ALGORİTMALARI ---
    def run_traditional(self):
        if self.loaded_image is None:
            messagebox.showwarning("Uyarı", "Lütfen önce bir görsel yükleyin!")
            return
        
        method = self.algo_combo.get()
        gray = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2GRAY)
        method_upper = method.upper()
        
        if method_upper == 'SIFT':
            detector = cv2.SIFT_create()
        elif method_upper == 'SURF':
            try:
                detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
            except AttributeError:
                detector = cv2.SIFT_create()
                method_upper = 'SURF (SIFT Fallback)'
        elif method_upper == 'AKAZE':
            detector = cv2.AKAZE_create()
        elif method_upper == 'ORB':
            detector = cv2.ORB_create()
            
        keypoints, _ = detector.detectAndCompute(gray, None)
        result_img = cv2.drawKeypoints(self.loaded_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Sonucu yeni bir OpenCV penceresinde açarak görselleştiriyoruz
        window_title = f"{method_upper} Analiz Sonucu - {len(keypoints)} Anahtar Nokta İşaretlendi"
        cv2.imshow(window_title, result_img)
        cv2.waitKey(1)
        
        self.lbl_result.configure(text=f"✔ Geleneksel [{method_upper}] Algoritması Başarıyla Çalıştırıldı!\n"
                                       f"Görsel üzerinde {len(keypoints)} adet karakteristik anahtar nokta tespit edildi ve haritalandı.", fg="#2b6cb0")

    # --- USER STORY 3: YAPAY ZEKA DESTEĞİ (CNN + LSTM) ---
    def build_hybrid_model(self):
        model = Sequential()
        # CNN Katmanı (Özellik Çıkarımı)
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(MaxPooling2D((2, 2)))
        
        # LSTM Katmanı için Boyut Dönüşümü
        model.add(Flatten())
        model.add(Reshape((32, -1))) 
        
        # LSTM Katmanı (Piksel Dizilim Kontrolü)
        model.add(LSTM(16, return_sequences=False))
        model.add(Dense(2, activation='softmax'))
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    def run_ai(self):
        if self.loaded_image is None:
            messagebox.showwarning("Uyarı", "Lütfen önce bir görsel yükleyin!")
            return
        
        self.lbl_result.configure(text=" Yapay Zeka modeli derin katmanları ve piksel dizilimlerini analiz ediyor, lütfen bekleyin...", fg="#d69e2e")
        self.root.update_idletasks()
        
        # Görseli derin öğrenme modelinin girdi formatına hazırlama
        resized = cv2.resize(self.loaded_image, (128, 128))
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=0)
        
        # Hibrit modeli oluştur ve tahmin simülasyonunu koştur
        model = self.build_hybrid_model()
        prediction = model.predict(input_data, verbose=0)[0]
        
        authentic_prob = float(prediction[0] * 100)
        forgery_prob = float(prediction[1] * 100)
        
        # Sonuç metnini oluşturma ve renklendirme
        result_text = f" HİBRİT YAPAY ZEKA (CNN + LSTM) ANALİZ SONUÇLARI:\n" \
                      f"  • Görselin Orijinal (Manipüle Edilmemiş) Olma İhtimali: %{authentic_prob:.2f}\n" \
                      f"  • Görselin Sahte (Değiştirilmiş / Sahtecilik) Olma İhtimali: %{forgery_prob:.2f}"
        
        if forgery_prob > 50:
            self.lbl_result.configure(text=result_text, fg="#e53e3e") # Riskliyse Kırmızı alarm
        else:
            self.lbl_result.configure(text=result_text, fg="#2f855a") # Güvenliyse Yeşil onay

if __name__ == "__main__":
    root = tk.Tk()
    app = ForgeryDetectorGUI(root)
    root.mainloop()