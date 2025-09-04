
# 🧨 Malla & Fragmentación —(Streamlit)

Aplicación interactiva en **Streamlit** para analizar parámetros de perforación/voladura y su efecto en **Powder Factor (PF)** y **fragmentación (P80)**.  
Incluye:
- Escalamiento proporcional (k en %) para Burden (B) y Espaciamiento (S).
- Caso B y S manuales.
- KPIs con nombres + unidades.
- Gráficos PF vs k, P80 vs k y curva P80–PF con OLS (statsmodels).
- Exportación a Excel (Inputs, Expand_BS y B_S_Manual).

## ▶️ Instalación y ejecución (Windows / PowerShell)
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

La app abrirá en `http://localhost:8501`.
