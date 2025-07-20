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

def train_models(df):
    """Modelleri eğit ve en iyisini seç"""
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=['Mevsim'], prefix='Mevsim')
    
    # Özellikler
    feature_cols = ['Ortalama_Sicaklik', 'Sicaklik_Kare', 'Nem', 'Ruzgar', 'Yagis', 
                    'Isitma_Derece_Gun', 'Sogutma_Derece_Gun', 'Ay']
    feature_cols.extend([col for col in df_encoded.columns if col.startswith('Mevsim_')])
    
    X = df_encoded[feature_cols]
    y = df_encoded['Dogalgaz_Tuketim']
    
    # Modelleri test et
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Lineer Regresyon': LinearRegression()
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Cross validation
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)//2), scoring='neg_mean_absolute_error')
        
        # Model eğit
        model.fit(X, y)
        predictions = model.predict(X)
        
        results[name] = {
            'CV_MAE': -cv_scores.mean(),
            'CV_STD': cv_scores.std(),
            'R2': r2_score(y, predictions),
            'MAE': mean_absolute_error(y, predictions),
            'RMSE': np.sqrt(mean_squared_error(y, predictions))
        }
        
        trained_models[name] = model
    
    # Polinom Regresyon
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    
    try:
        poly_cv_scores = cross_val_score(poly_model, X_poly, y, cv=min(3, len(X)//3), scoring='neg_mean_absolute_error')
        poly_model.fit(X_poly, y)
        poly_predictions = poly_model.predict(X_poly)
        
        results['Polinom Regresyon'] = {
            'CV_MAE': -poly_cv_scores.mean(),
            'CV_STD': poly_cv_scores.std(),
            'R2': r2_score(y, poly_predictions),
            'MAE': mean_absolute_error(y, poly_predictions),
            'RMSE': np.sqrt(mean_squared_error(y, poly_predictions))
        }
        
        trained_models['Polinom Regresyon'] = (poly_model, poly)
    except:
        st.warning("Polinom regresyon modeli eğitilemedi, veri boyutu küçük olabilir.")
    
    # En iyi modeli seç
    results_df = pd.DataFrame(results).T
    best_model_name = results_df['CV_MAE'].idxmin()
    
    return trained_models, results_df, best_model_name, feature_cols

def predict_future(df, best_model, best_model_name, feature_cols, months=12):
    """Gelecek aylar için tahmin yap"""
    
    # Son tarihi bul
    last_date = df['Tarih'].max()
    
    # Gelecek aylar için tarihler oluştur
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months, freq='MS')
    
    # Geçmiş verilerin istatistiklerini hesapla (meteoroloji için)
    weather_stats = {}
    for col in ['Ortalama_Sicaklik', 'Nem', 'Ruzgar', 'Yagis']:
        monthly_stats = df.groupby('Ay')[col].agg(['mean', 'std']).round(2)
        weather_stats[col] = monthly_stats
    
    future_predictions = []
    future_data = []
    
    for date in future_dates:
        month = date.month
        year = date.year
        
        # Geçmiş verilerden o ay için ortalama meteoroloji değerleri
        avg_temp = weather_stats['Ortalama_Sicaklik'].loc[month, 'mean']
        avg_humidity = weather_stats['Nem'].loc[month, 'mean']
        avg_wind = weather_stats['Ruzgar'].loc[month, 'mean']
        avg_rain = weather_stats['Yagis'].loc[month, 'mean']
        
        # Mevsim belirleme
        if month in [12, 1, 2]:
            season = 'Kış'
        elif month in [3, 4, 5]:
            season = 'İlkbahar'
        elif month in [6, 7, 8]:
            season = 'Yaz'
        else:
            season = 'Sonbahar'
        
        # Özellik vektörü oluştur
        features = {
            'Ortalama_Sicaklik': avg_temp,
            'Sicaklik_Kare': avg_temp**2,
            'Nem': avg_humidity,
            'Ruzgar': avg_wind,
            'Yagis': avg_rain,
            'Isitma_Derece_Gun': max(0, 18 - avg_temp),
            'Sogutma_Derece_Gun': max(0, avg_temp - 18),
            'Ay': month
        }
        
        # Mevsim dummy variables
        for col in [c for c in feature_cols if c.startswith('Mevsim_')]:
            features[col] = 1 if col == f'Mevsim_{season}' else 0
        
        # DataFrame'e çevir
        feature_df = pd.DataFrame([features])
        feature_df = feature_df[feature_cols]
        
        # Tahmin yap
        if best_model_name == 'Random Forest':
            prediction = best_model.predict(feature_df)[0]
        elif best_model_name == 'Polinom Regresyon':
            model, poly = best_model
            feature_poly = poly.transform(feature_df)
            prediction = model.predict(feature_poly)[0]
        else:
            prediction = best_model.predict(feature_df)[0]
        
        future_predictions.append(prediction)
        future_data.append({
            'Tarih': date,
            'Tahmin': prediction,
            'Ay': month,
            'Yil': year,
            'Mevsim': season,
            'Ortalama_Sicaklik': avg_temp,
            'Nem': avg_humidity,
            'Ruzgar': avg_wind,
            'Yagis': avg_rain
        })
    
    return pd.DataFrame(future_data)

# Ana uygulama mantığı
if uploaded_file is not None:
    try:
        # Veriyi yükle
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Veri başarıyla yüklendi! ({len(df)} satır)")
        
        # Veri önizlemesi
        st.subheader("📊 Veri Önizlemesi")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**İlk 5 satır:**")
            st.dataframe(df.head())
            
        with col2:
            st.write("**Temel İstatistikler:**")
            if 'Dogalgaz_Tuketim' in df.columns:
                st.metric("Toplam Veri Sayısı", len(df))
                st.metric("Ortalama Tüketim", f"{df['Dogalgaz_Tuketim'].mean():.0f} Milyon m³")
                st.metric("Maksimum Tüketim", f"{df['Dogalgaz_Tuketim'].max():.0f} Milyon m³")
        
        # Veriyi işle
        df_processed = load_and_process_data(df)
        
        if len(df_processed) < 6:
            st.error("❌ Model eğitimi için en az 6 aylık veri gerekli!")
        else:
            # Modelleri eğit
            with st.spinner('🤖 Modeller eğitiliyor...'):
                trained_models, results_df, best_model_name, feature_cols = train_models(df_processed)
            
            st.success(f"✅ En iyi model: **{best_model_name}**")
            
            # Model performansları
            st.subheader("📈 Model Performans Karşılaştırması")
            
            # Performans tablosu
            st.dataframe(results_df.round(3))
            
            # Performans grafiği
            fig = px.bar(
                x=results_df.index, 
                y=results_df['CV_MAE'],
                title="Cross-Validation Mean Absolute Error (Düşük = İyi)",
                labels={'x': 'Model', 'y': 'CV MAE'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Geçmiş veri görselleştirmesi
            st.subheader("📊 Geçmiş Veri Analizi")
            
            # Zaman serisi grafiği
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_processed['Tarih'],
                y=df_processed['Dogalgaz_Tuketim'],
                mode='lines+markers',
                name='Gerçek Tüketim',
                line=dict(width=3, color='blue'),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Doğalgaz Tüketimi Zaman Serisi",
                xaxis_title="Tarih",
                yaxis_title="Tüketim (Milyon m³)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Mevsimsel analiz
            col1, col2 = st.columns(2)
            
            with col1:
                # Mevsimsel box plot
                fig = px.box(df_processed, x='Mevsim', y='Dogalgaz_Tuketim',
                           title="Mevsimsel Tüketim Dağılımı")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sıcaklık vs tüketim
                fig = px.scatter(df_processed, x='Ortalama_Sicaklik', y='Dogalgaz_Tuketim',
                               color='Mevsim', title="Sıcaklık vs Tüketim İlişkisi")
                st.plotly_chart(fig, use_container_width=True)
            
            # Gelecek tahminler
            st.subheader("🔮 Gelecek 12 Ay Tahmini")
            
            # Gelecek tahminleri yap
            with st.spinner('🔮 Gelecek tahminler hesaplanıyor...'):
                best_model = trained_models[best_model_name]
                future_df = predict_future(df_processed, best_model, best_model_name, feature_cols, 12)
            
            st.success("✅ Gelecek 12 ay tahminleri hazır!")
            
            # Tahmin tablosu
            future_display = future_df.copy()
            future_display['Tarih'] = future_display['Tarih'].dt.strftime('%b %Y')
            future_display = future_display[['Tarih', 'Tahmin', 'Mevsim', 'Ortalama_Sicaklik', 'Nem']].round(2)
            future_display['Tahmin'] = future_display['Tahmin'].round(0)
            
            st.dataframe(future_display, use_container_width=True)
            
            # Tahmin grafiği (geçmiş + gelecek)
            fig = go.Figure()
            
            # Geçmiş veriler
            fig.add_trace(go.Scatter(
                x=df_processed['Tarih'],
                y=df_processed['Dogalgaz_Tuketim'],
                mode='lines+markers',
                name='Geçmiş Veriler',
                line=dict(width=3, color='blue'),
                marker=dict(size=6)
            ))
            
            # Gelecek tahminler
            fig.add_trace(go.Scatter(
                x=future_df['Tarih'],
                y=future_df['Tahmin'],
                mode='lines+markers',
                name='Tahminler',
                line=dict(width=3, color='red', dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
            # Ayırıcı çizgi
            fig.add_vline(
                x=df_processed['Tarih'].max(),
                line_dash="dot",
                line_color="gray",
                annotation_text="Tahmin Başlangıcı"
            )
            
            fig.update_layout(
                title="Doğalgaz Tüketimi: Geçmiş Veriler ve Gelecek Tahminleri",
                xaxis_title="Tarih",
                yaxis_title="Tüketim (Milyon m³)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Özet istatistikler
            st.subheader("📊 Tahmin Özeti")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Yıllık Toplam Tahmin", 
                    f"{future_df['Tahmin'].sum():.0f} Milyon m³"
                )
            
            with col2:
                st.metric(
                    "Aylık Ortalama", 
                    f"{future_df['Tahmin'].mean():.0f} Milyon m³"
                )
            
            with col3:
                st.metric(
                    "En Yüksek Ay", 
                    f"{future_df['Tahmin'].max():.0f} Milyon m³"
                )
            
            with col4:
                st.metric(
                    "En Düşük Ay", 
                    f"{future_df['Tahmin'].min():.0f} Milyon m³"
                )
            
            # Mevsimsel tahmin özeti
            seasonal_summary = future_df.groupby('Mevsim')['Tahmin'].agg(['sum', 'mean']).round(0)
            st.write("**Mevsimsel Tahmin Özeti:**")
            st.dataframe(seasonal_summary)
            
            # Excel'e aktarma butonu
            @st.cache_data
            def convert_to_excel(future_df):
                from io import BytesIO
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    future_display.to_excel(writer, sheet_name='Tahminler', index=False)
                    seasonal_summary.to_excel(writer, sheet_name='Mevsimsel_Ozet')
                return output.getvalue()
            
            excel_data = convert_to_excel(future_df)
            
            st.download_button(
                label="📥 Tahminleri Excel Olarak İndir",
                data=excel_data,
                file_name=f"dogalgaz_tahminleri_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Model güvenilirliği hakkında uyarı
            st.info(f"""
            **ℹ️ Model Güvenilirliği:**
            - En iyi model: {best_model_name}
            - Cross-validation MAE: {results_df.loc[best_model_name, 'CV_MAE']:.2f} Milyon m³
            - R² skoru: {results_df.loc[best_model_name, 'R2']:.3f}
            
            **📝 Notlar:**
            - Tahminler geçmiş verilerin istatistiksel ortalamasına dayanır
            - Ekstrem hava koşulları tahminleri etkileyebilir
            - BOTAŞ bildirimi için %5-10 güvenlik marjı eklemeniz önerilir
            """)
    
    except Exception as e:
        st.error(f"❌ Veri işleme hatası: {str(e)}")
        st.write("Lütfen Excel dosyanızın doğru formatta olduğundan emin olun.")

else:
    st.info("👆 Lütfen sol panelden Excel dosyanızı yükleyin.")
    
    # Örnek veri göster
    st.subheader("📋 Örnek Veri Formatı")
    sample_data = {
        'Tarih': ['Jan2020', 'Feb2020', 'Mar2020', 'Apr2020'],
        'Dogalgaz_Tuketim': [4500, 4200, 3800, 3200],
        'Ortalama_Sicaklik': [5.2, 7.8, 12.1, 16.5],
        'Nem': [75, 68, 62, 58],
        'Ruzgar': [1.8, 2.1, 1.9, 2.0],
        'Yagis': [2.5, 1.8, 1.2, 0.8]
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)
    
    st.markdown("""
    **Kullanım Adımları:**
    1. Excel dosyanızı yukarıdaki formatta hazırlayın
    2. Sol panelden dosyayı yükleyin
    3. Model otomatik olarak eğitilecek ve en iyi model seçilecek
    4. Gelecek 12 ay için tahmin yapılacak
    5. Sonuçları Excel olarak indirebilirsiniz
    """)

# Footer
st.markdown("---")
st.markdown("*🔥 Doğalgaz Tüketim Tahmin Uygulaması - BOTAŞ Bildirimleri İçin Geliştirildi*")
