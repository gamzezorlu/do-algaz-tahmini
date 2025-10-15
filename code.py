import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Doğalgaz Tüketim Tahmin Uygulaması",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Doğalgaz Tüketim Tahmin Uygulaması")
st.markdown("---")

# Sidebar - Veri Yükleme
st.sidebar.header("📁 Veri Yükleme")
uploaded_file = st.sidebar.file_uploader(
    "Excel dosyasını yükleyin",
    type=['xlsx', 'xls'],
    help="Tarih, Doğalgaz Tüketim, Ortalama Sıcaklık, Nem, Rüzgar, Yağış sütunları olmalıdır"
)

# Örnek veri formatı göster
with st.sidebar.expander("📋 Beklenen Veri Formatı"):
    st.write("""
    **Gerekli Sütunlar:**
    - Tarih (Jan2010, Feb2010, ...)
    - Dogalgaz_Tuketim (Milyon m³)
    - Ortalama_Sicaklik (°C)
    - Nem (%)
    - Ruzgar (m/s)
    - Yagis (mm)
    """)

def load_and_process_data(df):
    """Veriyi yükle ve işle"""
    # Tarih sütununu datetime'a çevir
    if 'Tarih' in df.columns:
        df['Tarih'] = pd.to_datetime(df['Tarih'], format='%b%Y', errors='coerce')
    
    # Eksik tarihleri temizle
    df = df.dropna(subset=['Tarih']).sort_values('Tarih').reset_index(drop=True)
    
    # Zaman serisi özellikleri ekle
    df['Yil'] = df['Tarih'].dt.year
    df['Ay'] = df['Tarih'].dt.month
    df['Mevsim'] = df['Ay'].map({
        12: 'Kış', 1: 'Kış', 2: 'Kış',
        3: 'İlkbahar', 4: 'İlkbahar', 5: 'İlkbahar',
        6: 'Yaz', 7: 'Yaz', 8: 'Yaz',
        9: 'Sonbahar', 10: 'Sonbahar', 11: 'Sonbahar'
    })
    
    # Sıcaklık bazlı özellikler
    df['Sicaklik_Kare'] = df['Ortalama_Sicaklik'] ** 2
    df['Sogutma_Derece_Gun'] = np.where(df['Ortalama_Sicaklik'] > 18, 
                                         df['Ortalama_Sicaklik'] - 18, 0)
    df['Isitma_Derece_Gun'] = np.where(df['Ortalama_Sicaklik'] < 18, 
                                       18 - df['Ortalama_Sicaklik'], 0)
    
    return df
    
    ------------------------------
# 🔄 2. Log Dönüşümü ve Gecikmeli Değişkenler
# --------------------------------------------
df['Dogalgaz_Tuketim_Log'] = np.log1p(df['Dogalgaz_Tuketim'])

# Gecikmeli (Lag) değişkenler ekleniyor
df['Lag1'] = df['Dogalgaz_Tuketim_Log'].shift(1)
df['Lag2'] = df['Dogalgaz_Tuketim_Log'].shift(2)
df['Lag3'] = df['Dogalgaz_Tuketim_Log'].shift(3)

df = df.dropna()  # Boş satırları atıyoruz (ilk 3 ay)

# --------------------------------------------
# 🧩 3. Özellikler
# --------------------------------------------
df['Ay'] = df['Tarih'].dt.month
df['Yil'] = df['Tarih'].dt.year

# Meteorolojik veriler sabitse en azından sabit bırakılabilir
df['Sıcaklık'] = df.get('Sıcaklık', 1)
df['Nem'] = df.get('Nem', 1)
df['Rüzgar'] = df.get('Rüzgar', 1)
df['Yağış'] = df.get('Yağış', 1)

feature_cols = ['Ay', 'Yil', 'Sıcaklık', 'Nem', 'Rüzgar', 'Yağış', 'Lag1', 'Lag2', 'Lag3']
X = df[feature_cols]
y = df['Dogalgaz_Tuketim_Log']

# --------------------------------------------
# ⚙️ 4. Özellik Ölçekleme
# --------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------
# 🧠 5. Model Eğitimi
# --------------------------------------------
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=6,
    min_samples_split=5,
    random_state=42
)
model.fit(X_scaled, y)

# --------------------------------------------
# 🔮 6. Gelecek Tahminleri (örnek: 2025)
# --------------------------------------------
future_dates = pd.date_range('2025-01-01', '2025-12-01', freq='MS')

# Son 3 log tüketimi al
last_values = list(df['Dogalgaz_Tuketim_Log'].iloc[-3:])

future_preds = []
for date in future_dates:
    ay = date.month
    yil = date.year
    sicaklik = 1
    nem = 1
    ruzgar = 1
    yagis = 1
    lag1, lag2, lag3 = last_values[-1], last_values[-2], last_values[-3]
    X_future = np.array([[ay, yil, sicaklik, nem, ruzgar, yagis, lag1, lag2, lag3]])
    X_future_scaled = scaler.transform(X_future)
    pred_log = model.predict(X_future_scaled)[0]
    pred_real = np.expm1(pred_log)  # log'dan geri dön
    future_preds.append(pred_real)
    last_values.append(pred_log)

future_df = pd.DataFrame({
    'Tarih': future_dates,
    'Tahmin_Tuketim': future_preds
})

# --------------------------------------------
# 📈 7. Görselleştirme
# --------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(df['Tarih'], df['Dogalgaz_Tuketim'], label='Gerçek')
plt.plot(future_df['Tarih'], future_df['Tahmin_Tuketim'], label='Tahmin (2025)', linestyle='--', color='orange')
plt.title('Doğalgaz Tüketim Tahmini (Geliştirilmiş Model)')
plt.xlabel('Tarih')
plt.ylabel('Tüketim (m³)')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------
# 📊 8. Performans Değerlendirme
# --------------------------------------------
y_pred_log = model.predict(X_scaled)
y_pred = np.expm1(y_pred_log)
y_real = np.expm1(y)

mae = mean_absolute_error(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
r2 = r2_score(y_real, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")
