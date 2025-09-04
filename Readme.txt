# App Malla & Fragmentación 

Esta app en **Streamlit** permite:

- Editar *inputs* de perforación/voladura (B, S, H, Pasadura, Taco, diámetro, densidad, etc.).
- Recalcular variables derivadas (A, qL, Lc, m_h, PF_base) y **P80** usando exponente **n1/n2/n3**.
- Explorar un escenario de **escalamiento proporcional** (k·B, k·S) replicando tu hoja `Expand_BS`.
- Analizar un caso de **B y S manuales** (según tu requerimiento).
- Graficar PF vs k, P80 vs k y curva P80–PF.
- Exportar resultados a Excel.

## Cómo ejecutar

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

> Si usas Python 3.13 y encuentras incompatibilidades, te sugiero crear el entorno con **Python 3.11/3.12** como alternativa temporal.
