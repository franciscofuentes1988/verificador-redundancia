from flask import Flask, request, render_template, send_file
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import io
import tempfile
import os

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')

def dividir_en_chunks(texto, metodo="frases", chunk_size=100):
    if metodo == "parrafos":
        return [p.strip() for p in texto.split("\n\n") if p.strip()]
    elif metodo == "palabras":
        palabras = texto.split()
        return [" ".join(palabras[i:i + chunk_size]) for i in range(0, len(palabras), chunk_size)]
    else:
        return [f.strip() for f in re.split(r'(?<=[.!?])\s+', texto.strip()) if f.strip()]

def clasificar_similitud(valor):
    if valor >= 0.9:
        return "Alta redundancia"
    elif valor >= 0.75:
        return "Redundancia media"
    elif valor >= 0.5:
        return "Baja redundancia"
    else:
        return "No hay redundancia"

def extraer_contenido(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        text = soup.body.get_text(separator=" ", strip=True)
        if any(err in text for err in ["Not Acceptable", "Mod_Security", "404", "403", "500"]):
            return None
        return text
    except:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        urls = request.form['urls'].strip().splitlines()
        all_chunks = []
        all_embeddings = []
        origen = []

        for url in urls:
            texto = extraer_contenido(url)
            if texto:
                chunks = dividir_en_chunks(texto)
                emb = model.encode(chunks)
                all_chunks.extend(chunks)
                all_embeddings.extend(emb)
                origen.extend([url] * len(chunks))

        if not all_embeddings:
            return "No se pudieron generar embeddings para las URLs ingresadas. Revisa que sean válidas."

        resultados = []
        sim_matrix = cosine_similarity(all_embeddings)
        for i in range(len(all_chunks)):
            for j in range(i + 1, len(all_chunks)):
                score = sim_matrix[i][j]
                if score >= 0.5:
                    resultados.append({
                        "URL 1": origen[i],
                        "Chunk 1": all_chunks[i],
                        "URL 2": origen[j],
                        "Chunk 2": all_chunks[j],
                        "Similitud": round(score, 3),
                        "Clasificacion": clasificar_similitud(score)
                    })

        df = pd.DataFrame(resultados)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        df.to_excel(tmp.name, index=False)
        return send_file(tmp.name, as_attachment=True, download_name='resultados_redundancia.xlsx')

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
