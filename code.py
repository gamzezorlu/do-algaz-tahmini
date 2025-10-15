import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
    help="Tarih, Dogalgaz_Tuketim, Ortalama_Sicaklik sütunları olmalıdır"
)

# Model seçimi
st.sidebar.header("⚙️ Model Ayarları")
model_type = st.sidebar.selectbox(
    "Model Seçin",
    ["Random Forest", "Gradient Boosting", "Ensemble (İkisi Birden)"]
)

# Örnek veri formatı göster
with st.sidebar.expander("📋 Beklenen Veri Formatı"):
    st.write("""
    **Gerekli Sütunlar:**
    - Tarih (Jan2016, Feb2016, ...)
    - Dogalgaz_Tuketim (493.813.816 formatında)
    - Ortalama_Sicaklik (2,9 formatında - virgül ile)
    - Nem, Ruzgar, Yagis (opsiyonel)
    """)

def clean_number(value):
    """Sayı formatını temizle"""
    if isinstance(value, str):
        value = value.replace('.', '')
        value = value.replace(',', '.')
        try:
            return float(value)
        except:
            return np.nan
    return float(value)

def load_and_process_data(df):
    """Veriyi yükle ve işle"""
    df.columns = df.columns.str.strip()
    
    # Tarih sütununu datetime'a çevir
    df['Tarih'] = pd.to_datetime(df['Tarih'], format='%b%Y', errors='coerce')
    df = df.dropna(subset=['Tarih']).sort_values('Tarih').reset_index(drop=True)
    
    # Tüketim değerini temizle ve milyona çevir
    df['Dogalgaz_Tuketim'] = df['Dogalgaz_Tuketim'].apply(clean_number) / 1_000_000
    
    # Sıcaklık değerini temizle
    if 'Ortalama_Sicaklik' in df.columns:
        df['Ortalama_Sicaklik'] = df['Ortalama_Sicaklik'].apply(clean_number)
    else:
        df['Ortalama_Sicaklik'] = 15
    
    # Diğer meteorolojik değerleri temizle
    for col in ['Nem', 'Ruzgar', 'Yagis']:
        if col in df.columns:
            df[col] = df[col].apply(clean_number)
        else:
            df[col] = 1
    
    # Zaman serisi özellikleri
    df['Yil'] = df['Tarih'].dt.year
    df['Ay'] = df['Tarih'].dt.month
    df['Ceyrek'] = df['Tarih'].dt.quarter
    
    # Mevsimsel sinüs/kosinüs özellikleri
    df['Ay_Sin'] = np.sin(2 * np.pi * df['Ay'] / 12)
    df['Ay_Cos'] = np.cos(2 * np.pi * df['Ay'] / 12)
    
    # Trend değişkeni
    df['Trend'] = np.arange(len(df))
    
    # Mevsim dummy değişkenleri
    df['Kis'] = df['Ay'].isin([12, 1, 2]).astype(int)
    df['Ilkbahar'] = df['Ay'].isin([3, 4, 5]).astype(int)
    df['Yaz'] = df['Ay'].isin([6, 7, 8]).astype(int)
    df['Sonbahar'] = df['Ay'].isin([9, 10, 11]).astype(int)
    
    # Yıllık normalizasyon
    df['Yil_Normalized'] = (df['Yil'] - df['Yil'].min()) / (df['Yil'].max() - df['Yil'].min() + 1)
    
    # Sıcaklık özellikleri
    df['Sicaklik_Kare'] = df['Ortalama_Sicaklik'] ** 2
    df['Sicaklik_Kup'] = df['Ortalama_Sicaklik'] ** 3
    df['Isitma_Derece_Gun'] = np.maximum(18 - df['Ortalama_Sicaklik'], 0)
    df['Sogutma_Derece_Gun'] = np.maximum(df['Ortalama_Sicaklik'] - 24, 0)
    
    # Gecikmeli değişkenler (Lag) - FORWARD FILL ile eksikleri doldur
    for i in range(1, 4):
        df[f'Lag{i}'] = df['Dogalgaz_Tuketim'].shift(i)
    
    # Yıllık gecikmeli değişken
    df['Lag12'] = df['Dogalgaz_Tuketim'].shift(12)
    
    # Hareketli ortalamalar
    df['MA3'] = df['Dogalgaz_Tuketim'].rolling(window=3, min_periods=1).mean()
    df['MA6'] = df['Dogalgaz_Tuketim'].rolling(window=6, min_periods=1).mean()
    df['MA12'] = df['Dogalgaz_Tuketim'].rolling(window=12, min_periods=1).mean()
    
    # Hareketli standart sapma
    df['STD3'] = df['Dogalgaz_Tuketim'].rolling(window=3, min_periods=1).std().fillna(0)
    
    # Yıllık büyüme oranı
    df['YoY_Growth'] = df['Dogalgaz_Tuketim'].pct_change(12).fillna(0)
    
    # Sıcaklık ve tüketim etkileşimi
    df['Temp_Tuketim_Interaction'] = df['Ortalama_Sicaklik'] * df['Lag1'].fillna(df['Dogalgaz_Tuketim'])
    
    # Eksik değerleri ileri doldur (backward fill yerine forward fill)
    lag_cols = ['Lag1', 'Lag2', 'Lag3', 'Lag12']
    for col in lag_cols:
        df[col] = df[col].fillna(method='bfill').fillna(df['Dogalgaz_Tuketim'].mean())
    
    # Kalan NaN'ları temizle
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def train_model(df, model_type):
    """Model eğitimi - SON 3 YIL AĞIRLIKLI"""
    feature_cols = [
        'Ay', 'Yil', 'Ceyrek', 'Trend', 'Yil_Normalized',
        'Ay_Sin', 'Ay_Cos',
        'Kis', 'Ilkbahar', 'Yaz', 'Sonbahar',
        'Ortalama_Sicaklik', 'Sicaklik_Kare', 'Sicaklik_Kup',
        'Isitma_Derece_Gun', 'Sogutma_Derece_Gun',
        'Lag1', 'Lag2', 'Lag3', 'Lag12',
        'MA3', 'MA6', 'MA12', 'STD3',
        'YoY_Growth', 'Temp_Tuketim_Interaction'
    ]
    
    X = df[feature_cols]
    y = df['Dogalgaz_Tuketim']
    
    # AĞIRLIKLANDIRMA: Son 3 yıla daha fazla önem ver
    max_year = df['Yil'].max()
    weights = np.ones(len(df))
    for i, year in enumerate(df['Yil']):
        if year >= max_year - 2:  # Son 3 yıl
            weights[i] = 3.0
        elif year >= max_year - 4:  # Son 5 yıl
            weights[i] = 2.0
        else:
            weights[i] = 1.0
    
    # Ölçekleme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model seçimi ve eğitimi
    if model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=42
        )
    else:  # Ensemble
        model1 = RandomForestRegressor(
            n_estimators=400, max_depth=20, min_samples_split=2, random_state=42, n_jobs=-1
        )
        model2 = GradientBoostingRegressor(
            n_estimators=400, max_depth=8, learning_rate=0.05, random_state=42
        )
        model1.fit(X_scaled, y, sample_weight=weights)
        model2.fit(X_scaled, y, sample_weight=weights)
        model = (model1, model2)
    
    if model_type != "Ensemble (İkisi Birden)":
        model.fit(X_scaled, y, sample_weight=weights)
        y_pred = model.predict(X_scaled)
    else:
        y_pred = (model[0].predict(X_scaled) + model[1].predict(X_scaled)) / 2
    
    y_real = y
    
    return model, scaler, X, y, y_pred, y_real, feature_cols

def predict_future(model, scaler, df, feature_cols, n_months, model_type):
    """Gelecek tahminleri - GEÇMİŞ AYLIK ORTALAMA BAZLI"""
    last_date = df['Tarih'].max()
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), 
                                 periods=n_months, freq='MS')
    
    # Son 24 ayın gerçek değerlerini kullan
    last_24_values = list(df['Dogalgaz_Tuketim'].iloc[-24:])
    last_trend = df['Trend'].iloc[-1]
    base_year = df['Yil'].min()
    year_range = df['Yil'].max() - base_year
    
    # Aylık ortalama sıcaklıklar - SON 3 YILIN ORTALAMASINI AL
    recent_years = df['Yil'].max() - 2
    recent_df = df[df['Yil'] >= recent_years]
    monthly_avg_temp = recent_df.groupby('Ay')['Ortalama_Sicaklik'].mean().to_dict()
    
    # HER AY İÇİN SON 3 YILIN ORTALAMA TÜKETİMİNİ HESAPLA
    monthly_avg_consumption = recent_df.groupby('Ay')['Dogalgaz_Tuketim'].mean().to_dict()
    
    future_preds = []
    prediction_values = list(df['Dogalgaz_Tuketim'].iloc[-24:])
    
    for i, date in enumerate(future_dates):
        ay = date.month
        yil = date.year
        ceyrek = (ay - 1) // 3 + 1
        
        # Zaman özellikleri
        trend = last_trend + i + 1
        yil_normalized = (yil - base_year) / max(year_range, 1)
        ay_sin = np.sin(2 * np.pi * ay / 12)
        ay_cos = np.cos(2 * np.pi * ay / 12)
        
        # Mevsim
        kis = 1 if ay in [12, 1, 2] else 0
        ilkbahar = 1 if ay in [3, 4, 5] else 0
        yaz = 1 if ay in [6, 7, 8] else 0
        sonbahar = 1 if ay in [9, 10, 11] else 0
        
        # Sıcaklık tahmini
        temp = monthly_avg_temp.get(ay, df['Ortalama_Sicaklik'].mean())
        temp_kare = temp ** 2
        temp_kup = temp ** 3
        isitma = max(18 - temp, 0)
        sogutma = max(temp - 24, 0)
        
        # Lag değerleri
        lag1 = prediction_values[-1]
        lag2 = prediction_values[-2] if len(prediction_values) >= 2 else lag1
        lag3 = prediction_values[-3] if len(prediction_values) >= 3 else lag1
        lag12 = prediction_values[-12] if len(prediction_values) >= 12 else lag1
        
        # MA değerleri
        ma3 = np.mean(prediction_values[-3:])
        ma6 = np.mean(prediction_values[-6:])
        ma12 = np.mean(prediction_values[-12:])
        std3 = np.std(prediction_values[-3:]) if len(prediction_values) >= 3 else 0
        
        # YoY Growth
        if len(prediction_values) >= 12:
            yoy_growth = (prediction_values[-1] - prediction_values[-12]) / (abs(prediction_values[-12]) + 1e-10)
        else:
            yoy_growth = 0
        
        # Etkileşim
        temp_interaction = temp * lag1
        
        # Özellikleri birleştir
        features = [
            ay, yil, ceyrek, trend, yil_normalized,
            ay_sin, ay_cos,
            kis, ilkbahar, yaz, sonbahar,
            temp, temp_kare, temp_kup, isitma, sogutma,
            lag1, lag2, lag3, lag12,
            ma3, ma6, ma12, std3,
            yoy_growth, temp_interaction
        ]
        
        X_future = np.array([features])
        X_future_scaled = scaler.transform(X_future)
        
        # Tahmin
        if model_type == "Ensemble (İkisi Birden)":
            pred_model = (model[0].predict(X_future_scaled)[0] + 
                         model[1].predict(X_future_scaled)[0]) / 2
        else:
            pred_model = model.predict(X_future_scaled)[0]
        
        # AYLIK ORTALAMA İLE BLEND (60% model, 40% historical average)
        historical_avg = monthly_avg_consumption.get(ay, pred_model)
        pred = 0.70 * pred_model + 0.30 * historical_avg
        
        # Negatif tahminleri engelle
        pred = max(pred, 0)
        
        future_preds.append(pred)
        prediction_values.append(pred)
    
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
        
        # İlk 5 satırı göster
        with st.expander("🔍 Ham Veri Önizleme"):
            st.dataframe(df.head(10))
        
        # Veriyi işle
        with st.spinner("⚙️ Veri işleniyor..."):
            df_processed = load_and_process_data(df)
        
        st.info(f"✓ İşlenmiş veri: {len(df_processed)} satır")
        
        # Son 5 satırı göster - debugging için
        with st.expander("🔍 Son 5 Satır (Debugging)"):
            st.dataframe(df_processed[['Tarih', 'Dogalgaz_Tuketim', 'Ortalama_Sicaklik', 'Lag1', 'Lag12']].tail())
        
        # Model eğit
        with st.spinner(f"🧠 {model_type} modeli eğitiliyor..."):
            model, scaler, X, y, y_pred, y_real, feature_cols = train_model(df_processed, model_type)
        
        # Performans metrikleri
        mae = mean_absolute_error(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        r2 = r2_score(y_real, y_pred)
        mape = np.mean(np.abs((y_real - y_pred) / (y_real + 1e-10))) * 100
        
        # Metrikler
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{mae:.2f} M m³")
        with col2:
            st.metric("RMSE", f"{rmse:.2f} M m³")
        with col3:
            st.metric("R² Skoru", f"{r2:.4f}")
        with col4:
            st.metric("MAPE", f"{mape:.2f}%")
        
        st.markdown("---")
        
        # Tahmin ayarları
        col1, col2 = st.columns([1, 3])
        with col1:
            n_months = st.slider("Tahmin Süresi (Ay)", 1, 24, 12)
        
        # Gelecek tahminleri
        with st.spinner("🔮 Gelecek tahminleri hesaplanıyor..."):
            future_df = predict_future(model, scaler, df_processed, feature_cols, n_months, model_type)
        
        # Ana grafik
        fig = go.Figure()
        
        # Gerçek değerler
        fig.add_trace(go.Scatter(
            x=df_processed['Tarih'],
            y=df_processed['Dogalgaz_Tuketim'],
            mode='lines+markers',
            name='Gerçek Tüketim',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=5)
        ))
        
        # Model tahminleri (geçmiş)
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
            marker=dict(size=7, symbol='diamond')
        ))
        
        fig.update_layout(
            title=f'Doğalgaz Tüketim Tahmini ({model_type})',
            xaxis_title='Tarih',
            yaxis_title='Tüketim (Milyon m³)',
            hovermode='x unified',
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hata analizi grafiği
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=y_real,
                y=y_pred,
                mode='markers',
                marker=dict(size=8, color='blue', opacity=0.6),
                name='Tahminler'
            ))
            fig_scatter.add_trace(go.Scatter(
                x=[y_real.min(), y_real.max()],
                y=[y_real.min(), y_real.max()],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Mükemmel Tahmin'
            ))
            fig_scatter.update_layout(
                title='Tahmin vs Gerçek Değerler',
                xaxis_title='Gerçek Tüketim',
                yaxis_title='Tahmin Edilen Tüketim',
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            errors = y_pred - y_real
            fig_error = go.Figure()
            fig_error.add_trace(go.Histogram(
                x=errors,
                nbinsx=30,
                marker_color='indianred',
                name='Hata Dağılımı'
            ))
            fig_error.update_layout(
                title='Tahmin Hatası Dağılımı',
                xaxis_title='Hata (M m³)',
                yaxis_title='Frekans',
                height=400
            )
            st.plotly_chart(fig_error, use_container_width=True)
        
        # Tahmin tablosu
        st.subheader("📊 Gelecek Tahmin Detayları")
        future_display = future_df.copy()
        future_display['Tarih'] = future_display['Tarih'].dt.strftime('%B %Y')
        future_display['Tahmin_Tuketim'] = future_display['Tahmin_Tuketim'].round(2)
        
        future_display['Mevsim'] = future_df['Tarih'].dt.month.map({
            12: 'Kış', 1: 'Kış', 2: 'Kış',
            3: 'İlkbahar', 4: 'İlkbahar', 5: 'İlkbahar',
            6: 'Yaz', 7: 'Yaz', 8: 'Yaz',
            9: 'Sonbahar', 10: 'Sonbahar', 11: 'Sonbahar'
        })
        
        future_display.columns = ['Tarih', 'Tahmini Tüketim (M m³)', 'Mevsim']
        st.dataframe(future_display, use_container_width=True, height=400)
        
        # Özellik önemliliği
        st.subheader("🎯 Özellik Önemliliği")
        
        if model_type == "Ensemble (İkisi Birden)":
            feature_importance = pd.DataFrame({
                'Özellik': feature_cols,
                'Önem': (model[0].feature_importances_ + model[1].feature_importances_) / 2
            }).sort_values('Önem', ascending=False).head(15)
        else:
            feature_importance = pd.DataFrame({
                'Özellik': feature_cols,
                'Önem': model.feature_importances_
            }).sort_values('Önem', ascending=False).head(15)
        
        fig_importance = px.bar(
            feature_importance,
            x='Önem',
            y='Özellik',
            orientation='h',
            title='Top 15 Önemli Özellikler',
            color='Önem',
            color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(height=500)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # İstatistikler
        with st.expander("📈 Detaylı İstatistikler"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Tüketim İstatistikleri:**")
                st.write(f"- Ortalama: {df_processed['Dogalgaz_Tuketim'].mean():.2f} M m³")
                st.write(f"- Medyan: {df_processed['Dogalgaz_Tuketim'].median():.2f} M m³")
                st.write(f"- Min: {df_processed['Dogalgaz_Tuketim'].min():.2f} M m³")
                st.write(f"- Max: {df_processed['Dogalgaz_Tuketim'].max():.2f} M m³")
                st.write(f"- Std: {df_processed['Dogalgaz_Tuketim'].std():.2f} M m³")
            
            with col2:
                st.write("**Model Performansı:**")
                st.write(f"- Ortalama Hata: {mae:.2f} M m³")
                st.write(f"- Hata Yüzdesi (MAPE): {mape:.2f}%")
                st.write(f"- Açıklanan Varyans (R²): {r2*100:.2f}%")
                st.write(f"- Eğitim Verisi: {len(df_processed)} ay")
        
    except Exception as e:
        st.error(f"❌ Hata oluştu: {str(e)}")
        st.exception(e)
        st.info("💡 Lütfen veri formatının doğru olduğundan emin olun.")

else:
    st.info("👈 Lütfen sol menüden bir Excel dosyası yükleyin.")
    
    st.subheader("📝 Örnek Veri Formatı")
    example_data = pd.DataFrame({
        'Tarih': ['Jan2016', 'Feb2016', 'Mar2016', 'Apr2016', 'May2016'],
        'Dogalgaz_Tuketim': ['493.813.816', '311.920.398', '283.742.149', '112.999.486', '91.270.226'],
        'Ortalama_Sicaklik': ['2,9', '4,1', '7,7', '12,3', '17,1'],
        'Nem': [1, 1, 1, 1, 1],
        'Ruzgar': [1, 1, 1, 1, 1],
        'Yagis': [1, 1, 1, 1, 1]
    })
    st.dataframe(example_data, use_container_width=True)
