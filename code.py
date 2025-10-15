import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    help="Tarih, Dogalgaz_Tuketim, Ortalama_Sicaklik, Nem, Ruzgar, Yagis sütunları olmalıdır"
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
    
    # Log dönüşümü
    df['Dogalgaz_Tuketim_Log'] = np.log1p(df['Dogalgaz_Tuketim'])
    
    # Gecikmeli değişkenler
    df['Lag1'] = df['Dogalgaz_Tuketim_Log'].shift(1)
    df['Lag2'] = df['Dogalgaz_Tuketim_Log'].shift(2)
    df['Lag3'] = df['Dogalgaz_Tuketim_Log'].shift(3)
    
    # Sıcaklık bazlı özellikler
    if 'Ortalama_Sicaklik' in df.columns:
        df['Sicaklik_Kare'] = df['Ortalama_Sicaklik'] ** 2
        df['Sogutma_Derece_Gun'] = np.where(df['Ortalama_Sicaklik'] > 18, 
                                             df['Ortalama_Sicaklik'] - 18, 0)
        df['Isitma_Derece_Gun'] = np.where(df['Ortalama_Sicaklik'] < 18, 
                                           18 - df['Ortalama_Sicaklik'], 0)
    
    # Eksik değerleri temizle
    df = df.dropna()
    
    return df

def train_model(df):
    """Model eğitimi"""
    # Özellik sütunları
    feature_cols = ['Ay', 'Yil', 'Ortalama_Sicaklik', 'Nem', 'Ruzgar', 
                   'Yagis', 'Lag1', 'Lag2', 'Lag3']
    
    X = df[feature_cols]
    y = df['Dogalgaz_Tuketim_Log']
    
    # Ölçekleme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model eğitimi
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Tahminler
    y_pred_log = model.predict(X_scaled)
    y_pred = np.expm1(y_pred_log)
    y_real = np.expm1(y)
    
    return model, scaler, X, y, y_pred, y_real

def predict_future(model, scaler, df, n_months=12):
    """Gelecek tahminleri"""
    last_date = df['Tarih'].max()
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), 
                                 periods=n_months, freq='MS')
    
    # Son 3 log değer
    last_values = list(df['Dogalgaz_Tuketim_Log'].iloc[-3:])
    
    # Ortalama meteorolojik değerler
    avg_temp = df['Ortalama_Sicaklik'].mean()
    avg_nem = df['Nem'].mean()
    avg_ruzgar = df['Ruzgar'].mean()
    avg_yagis = df['Yagis'].mean()
    
    future_preds = []
    for date in future_dates:
        ay = date.month
        yil = date.year
        
        # Mevsimsel sıcaklık tahmini
        seasonal_temp = df[df['Ay'] == ay]['Ortalama_Sicaklik'].mean()
        temp = seasonal_temp if not pd.isna(seasonal_temp) else avg_temp
        
        lag1, lag2, lag3 = last_values[-1], last_values[-2], last_values[-3]
        
        X_future = np.array([[ay, yil, temp, avg_nem, avg_ruzgar, avg_yagis, 
                             lag1, lag2, lag3]])
        X_future_scaled = scaler.transform(X_future)
        
        pred_log = model.predict(X_future_scaled)[0]
        pred_real = np.expm1(pred_log)
        
        future_preds.append(pred_real)
        last_values.append(pred_log)
    
    future_df = pd.DataFrame({
        'Tarih': future_dates,
        'Tahmin_Tuketim': future_preds
    })
    
    return future_df

# Ana uygulama akışı
if uploaded_file is not None:
    try:
        # Veriyi yükle
        df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Veri başarıyla yüklendi! ({len(df)} satır)")
        
        # Veriyi işle
        df_processed = load_and_process_data(df)
        
        # Model eğit
        with st.spinner("🧠 Model eğitiliyor..."):
            model, scaler, X, y, y_pred, y_real = train_model(df_processed)
        
        # Performans metrikleri
        mae = mean_absolute_error(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        r2 = r2_score(y_real, y_pred)
        
        # Metrikler
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE (Ortalama Mutlak Hata)", f"{mae:.2f} M m³")
        with col2:
            st.metric("RMSE (Kök Ortalama Kare Hata)", f"{rmse:.2f} M m³")
        with col3:
            st.metric("R² Skoru", f"{r2:.3f}")
        
        st.markdown("---")
        
        # Tahmin ayarları
        col1, col2 = st.columns([1, 3])
        with col1:
            n_months = st.slider("Tahmin Süresi (Ay)", 1, 24, 12)
        
        # Gelecek tahminleri
        future_df = predict_future(model, scaler, df_processed, n_months)
        
        # Ana grafik
        fig = go.Figure()
        
        # Gerçek değerler
        fig.add_trace(go.Scatter(
            x=df_processed['Tarih'],
            y=df_processed['Dogalgaz_Tuketim'],
            mode='lines+markers',
            name='Gerçek Tüketim',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        # Tahminler (geçmiş)
        fig.add_trace(go.Scatter(
            x=df_processed['Tarih'],
            y=y_pred,
            mode='lines',
            name='Model Tahmini (Geçmiş)',
            line=dict(color='#2ca02c', width=2, dash='dot'),
            opacity=0.7
        ))
        
        # Gelecek tahminleri
        fig.add_trace(go.Scatter(
            x=future_df['Tarih'],
            y=future_df['Tahmin_Tuketim'],
            mode='lines+markers',
            name='Gelecek Tahmini',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='Doğalgaz Tüketim Tahmini',
            xaxis_title='Tarih',
            yaxis_title='Tüketim (Milyon m³)',
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tahmin tablosu
        st.subheader("📊 Gelecek Tahmin Detayları")
        future_display = future_df.copy()
        future_display['Tarih'] = future_display['Tarih'].dt.strftime('%B %Y')
        future_display['Tahmin_Tuketim'] = future_display['Tahmin_Tuketim'].round(2)
        future_display.columns = ['Tarih', 'Tahmini Tüketim (M m³)']
        st.dataframe(future_display, use_container_width=True)
        
        # Özellik önemliliği
        st.subheader("🎯 Özellik Önemliliği")
        feature_importance = pd.DataFrame({
            'Özellik': ['Ay', 'Yıl', 'Sıcaklık', 'Nem', 'Rüzgar', 'Yağış', 
                       'Lag1', 'Lag2', 'Lag3'],
            'Önem': model.feature_importances_
        }).sort_values('Önem', ascending=False)
        
        fig_importance = px.bar(
            feature_importance,
            x='Önem',
            y='Özellik',
            orientation='h',
            title='Model Özellik Önemliliği'
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Hata oluştu: {str(e)}")
        st.info("Lütfen veri formatının doğru olduğundan emin olun.")

else:
    st.info("👈 Lütfen sol menüden bir Excel dosyası yükleyin.")
    
    # Örnek veri göster
    st.subheader("📝 Örnek Veri Formatı")
    example_data = pd.DataFrame({
        'Tarih': ['Jan2020', 'Feb2020', 'Mar2020', 'Apr2020', 'May2020'],
        'Dogalgaz_Tuketim': [450.5, 420.3, 380.2, 320.1, 280.5],
        'Ortalama_Sicaklik': [2.5, 4.0, 8.5, 14.2, 18.5],
        'Nem': [75, 70, 65, 60, 55],
        'Ruzgar': [3.2, 3.5, 3.8, 3.0, 2.8],
        'Yagis': [45, 38, 32, 28, 25]
    })
    st.dataframe(example_data, use_container_width=True)
