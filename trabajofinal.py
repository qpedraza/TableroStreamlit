import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
import spacy
import re
import requests

nltk.download('stopwords', quiet=True)

#Carga del modelo de lematización en español
@st.cache_resource
def cargar_nlp():
    return spacy.load("es_core_news_sm")

nlp = cargar_nlp()

#Configuración de página
st.set_page_config(page_title="Análisis de Opiniones Tech", page_icon="💻", layout="wide")

#Fondo oscuro con CSS personalizado
st.markdown("""
    <style>
        .stApp { background-color: #1a1a2e; color: #e0e0e0; }
        [data-testid="stSidebar"] { background-color: #16213e; }
        h1, h2, h3 { color: #00d4ff !important; }
        p, label, div { color: #e0e0e0; }
        .stButton > button { background-color: #0f3460; color: white; border: 1px solid #00d4ff; border-radius: 8px; }
        .stButton > button:hover { background-color: #00d4ff; color: #1a1a2e; }
        [data-testid="metric-container"] { background-color: #16213e; border-radius: 8px; padding: 10px; border: 1px solid #0f3460; }
        textarea { background-color: #16213e !important; color: #e0e0e0 !important; border: 1px solid #00d4ff !important; }
        [data-testid="stSelectbox"] > div > div { background-color: #0f3460 !important; color: #e0e0e0 !important; border: 1px solid #00d4ff !important; border-radius: 6px; }
        [data-testid="stSelectbox"] ul { background-color: #0f3460 !important; color: #e0e0e0 !important; }
        [data-testid="stSelectbox"] ul li:hover { background-color: #00d4ff !important; color: #1a1a2e !important; }
        input[type="password"], input[type="text"] { background-color: #0f3460 !important; color: #e0e0e0 !important; border: 1px solid #00d4ff !important; border-radius: 6px !important; }
        input::placeholder { color: #8899aa !important; }
        hr { border-color: #0f3460; }
    </style>
""", unsafe_allow_html=True)

st.title("💻 Análisis de Opiniones de Productos Tecnológicos")

#Carga del dataset
@st.cache_data
def cargar_datos():
    return pd.read_csv("tecnologia.csv")

df = cargar_datos()

#Stopwords
stop_words = set(stopwords.words("spanish"))
stop_words.update([
    "si", "muy", "bien", "ser", "estar", "tener", "poco", "solo",
    "también", "cuando", "para", "pero", "como", "todo", "cada",
    "uno", "vez", "hay", "puede", "tiene", "son", "sus", "los",
    "las", "del", "con", "por", "una", "que", "no", "es", "más",
    "así", "hacer", "aunque", "desde", "hasta", "sobre", "porque",
    "este", "esta", "esto", "ese", "esa", "cual", "algo", "sin"
])

#Lematización con spaCy
def preprocesar(texto):
    
    texto = re.sub(r'[^a-záéíóúüñ\s]', '', texto.lower())
    doc = nlp(texto)
    return [
        token.lemma_                    # Forma base real de la palabra
        for token in doc
        if token.lemma_ not in stop_words
        and not token.is_punct          # Sin puntuación
        and not token.is_space          # Sin espacios
        and len(token.lemma_) > 3       # Solo palabras de más de 3 letras
    ]

#Sidebar
st.sidebar.title("🔧 Filtros")
opciones = ["Todas"] + sorted(df["estrellas"].unique().tolist())
seleccion = st.sidebar.selectbox("Filtrar por estrellas:", opciones)

hf_token = st.sidebar.text_input(
    "🔑 Token de HuggingFace:", type="password",
    help="Ingresa tu token de HuggingFace para analizar comentarios"
)

df_filtrado = df.copy() if seleccion == "Todas" else df[df["estrellas"] == int(seleccion)].copy()
st.sidebar.metric("Opiniones seleccionadas", len(df_filtrado))

#Palabras procesadas 
palabras = []
for op in df_filtrado["opinion"]:
    palabras.extend(preprocesar(str(op)))

conteo = Counter(palabras)
top10 = conteo.most_common(10)

#Estilo oscuro para gráficos matplotlib 
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.labelcolor": "#e0e0e0",
    "axes.titlecolor": "#00d4ff",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "text.color": "#e0e0e0",
    "grid.color": "#0f3460"
})

#Nube de palabras y Barras
st.header("📊 Frecuencia de Palabras")
col1, col2 = st.columns(2)

with col1:
    st.subheader("☁️ Nube de Palabras")
    if palabras:
        wc = WordCloud(
            width=700, height=380,
            background_color="#1a1a2e",
            colormap="cool",
            max_words=80
        ).generate(" ".join(palabras))
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.imshow(wc, interpolation="bilinear")
        ax1.axis("off")
        st.pyplot(fig1)
        plt.close(fig1)
    else:
        st.warning("Sin datos suficientes para la nube.")

with col2:
    st.subheader("📈 Top 10 Palabras")
    if top10:
        pals = [t[0] for t in top10]
        freqs = [t[1] for t in top10]
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        bars = ax2.barh(pals, freqs, color="#00d4ff")
        ax2.invert_yaxis()
        ax2.set_xlabel("Frecuencia")
        ax2.set_title("Palabras más frecuentes")
        for bar, freq in zip(bars, freqs):
            ax2.text(bar.get_width() + 0.1,
                     bar.get_y() + bar.get_height() / 2,
                     str(freq), va="center", fontsize=9, color="#e0e0e0")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

#Distribución de estrellas
st.header("⭐ Distribución de Calificaciones")
conteo_estrellas = df_filtrado["estrellas"].value_counts().sort_index()
colores_map = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#2ecc71", 5: "#27ae60"}
colores_barras = [colores_map.get(i, "gray") for i in conteo_estrellas.index]

fig3, ax3 = plt.subplots(figsize=(8, 4))
barras = ax3.bar(conteo_estrellas.index, conteo_estrellas.values,
                  color=colores_barras, edgecolor="#1a1a2e")
for b in barras:
    h = b.get_height()
    ax3.text(b.get_x() + b.get_width() / 2, h + 0.1,
             str(int(h)), ha="center", fontsize=10, fontweight="bold")
ax3.set_xlabel("Estrellas")
ax3.set_ylabel("Cantidad de opiniones")
ax3.set_title("Distribución de calificaciones")
ax3.set_xticks([1, 2, 3, 4, 5])
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

#Análisis con HuggingFace
st.header("Analiza tu Propio Comentario")
st.markdown("Escribe una opinión y la IA de HuggingFace detectará su sentimiento.")

comentario = st.text_area(
    "Escribe tu opinión aquí:",
    placeholder="Ej: La batería de este celular dura muy poco...",
    height=120
)

def analizar_con_huggingface(texto, token):
    API_URL = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(API_URL, headers=headers,
                              json={"inputs": texto}, timeout=30)
    if response.status_code == 503:
        raise ValueError("El modelo está cargando, espera 20 segundos e intenta de nuevo.")
    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.text}")
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        items = result[0] if isinstance(result[0], list) else result
        mejor = max(items, key=lambda x: x["score"])
        etiqueta = mejor["label"].upper()
        
        mapa = {
            "POSITIVE": ("😊 Positivo", "#2ecc71"),
            "NEGATIVE": ("😠 Negativo", "#e74c3c"),
            "NEUTRAL":  ("😐 Neutro",   "#f1c40f")
        }
        return mapa.get(etiqueta, ("😐 Neutro", "#f1c40f"))
    raise ValueError(f"Respuesta inesperada: {result}")

if st.button("🧠 Analizar Sentimiento", type="primary"):
    if not comentario.strip():
        st.warning("Por favor escribe un comentario antes de analizar.")
    elif not hf_token:
        st.warning("Por favor ingresa tu token de HuggingFace en el panel lateral.")
    else:
        with st.spinner("Analizando con HuggingFace..."):
            try:
                sentimiento, color = analizar_con_huggingface(comentario, hf_token)
                st.markdown(
                    f"<div style='background:{color};padding:14px;border-radius:8px;"
                    f"font-size:18px;font-weight:bold;text-align:center;color:#1a1a2e'>"
                    f"Sentimiento detectado: {sentimiento}</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
