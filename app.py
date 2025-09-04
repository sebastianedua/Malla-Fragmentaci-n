# -*- coding: utf-8 -*-
import io
import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Verificación temprana: statsmodels es requerido por trendline="ols"
_SM_OK = True
try:
    import statsmodels.api as sm  # noqa: F401
except Exception:
    _SM_OK = False

st.set_page_config(page_title="Malla & Fragmentación", page_icon="🧨", layout="wide")

IN2M = 0.0254  # 1 pulgada = 0.0254 m

# Estado de sesión para compartir datos entre pestañas
if 'dfk' not in st.session_state:
    st.session_state['dfk'] = pd.DataFrame()
if 'manual_case' not in st.session_state:
    st.session_state['manual_case'] = None

# -----------------------------
# Utilidades
# -----------------------------
@dataclass
class Params:
    B: float  # Burden (m)
    S: float  # Espaciamiento (m)
    H: float  # Altura banco (m)
    Pasadura: float  # (m)
    Taco: float  # (m)
    d_in: float  # Diámetro (in)
    d_m: float  # Diámetro (m)
    rho: float  # Densidad explosivo (kg/m3)
    P80_base: float  # mm
    n1: float
    n2: float
    n3: float

    @property
    def L(self) -> float:
        return self.H + self.Pasadura

    @property
    def A(self) -> float:
        return math.pi * (self.d_m ** 2) / 4.0

    @property
    def qL(self) -> float:
        return self.rho * self.A

    @property
    def Lc(self) -> float:
        return max(self.L - self.Taco, 0.0)

    @property
    def m_h(self) -> float:
        return self.qL * self.Lc

    def volumen_por_pozo(self, B: float = None, S: float = None, H: float = None) -> float:
        B = self.B if B is None else B
        S = self.S if S is None else S
        H = self.H if H is None else H
        return B * S * H

    def pf(self, B: float = None, S: float = None, H: float = None) -> float:
        V = self.volumen_por_pozo(B, S, H)
        return self.m_h / V if V > 0 else float('nan')

    def p80(self, pf_ref: float, pf_new: float, expo: float) -> float:
        ratio = pf_ref / pf_new if pf_new > 0 else float('nan')
        return self.P80_base * (ratio ** expo)

    def ratios(self, B: float = None, S: float = None):
        B = self.B if B is None else B
        S = self.S if S is None else S
        return (S / B if B else float('nan'), B / self.d_m if self.d_m else float('nan'))


def load_params_from_excel(file) -> Dict[str, float]:
    try:
        df0 = pd.read_excel(file, sheet_name=0, header=0)
    except Exception:
        return {}

    cols = [c.lower() for c in df0.columns]
    var_col = None
    val_col = None
    for i, c in enumerate(cols):
        if 'variable' in c:
            var_col = df0.columns[i]
        if 'valor' in c:
            val_col = df0.columns[i]
    if var_col is None or val_col is None:
        var_col = df0.columns[0]
        val_col = df0.columns[1] if len(df0.columns) > 1 else df0.columns[0]

    pairs = {}
    for _, row in df0[[var_col, val_col]].dropna().iterrows():
        key = str(row[var_col]).strip()
        try:
            val = float(str(row[val_col]).strip())
        except Exception:
            continue
        pairs[key] = val
    return pairs


def defaults_from_pairs(pairs: Dict[str, float]) -> Params:
    B = pairs.get('Burden B (m)', 3.8)
    S = pairs.get('Espaciamiento S (m)', 4.3)
    H = pairs.get('Altura banco H (m)', 5.0)
    Pasadura = pairs.get('Pasadura (m)', 0.6)
    Taco = pairs.get('Taco (m)', 1.7)
    d_in = pairs.get('Diámetro (in)', 4.5)
    d_m = pairs.get('Diámetro (m)', d_in * IN2M if d_in else 0.1143)  # si no viene, lo inferimos
    rho = pairs.get('Densidad explosivo ρ (kg/m^3)', 770)
    P80_base = pairs.get('P80 base (mm)', 175)
    n1 = pairs.get('Exponente n1 (sensibilidad energía)', 0.6)
    n2 = pairs.get('Exponente n2', 0.8)
    n3 = pairs.get('Exponente n3', 1.0)
    return Params(B, S, H, Pasadura, Taco, d_in, d_m, rho, P80_base, n1, n2, n3)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header('📄 Archivo de origen (opcional)')
up = st.sidebar.file_uploader('Carga tu Excel de "Inputs" (mantén columnas Variable/Valor).', type=['xlsx'])
if up is not None:
    pairs = load_params_from_excel(up)
else:
    pairs = {}

params = defaults_from_pairs(pairs)

st.sidebar.header('⚙️ Parámetros de Diseño (editables)')
col1, col2 = st.sidebar.columns(2)
params.B = col1.number_input('Burden B (m)', min_value=0.1, value=float(params.B), step=0.1)
params.S = col2.number_input('Espaciamiento S (m)', min_value=0.1, value=float(params.S), step=0.1)
col3, col4 = st.sidebar.columns(2)
params.H = col3.number_input('Altura banco H (m)', min_value=0.1, value=float(params.H), step=0.1)
params.Pasadura = col4.number_input('Pasadura (m)', min_value=0.0, value=float(params.Pasadura), step=0.1)

# --- Sincronización de diámetros (in → m)
col5, col6 = st.sidebar.columns(2)
d_in_val = col5.number_input('Diámetro (in)', min_value=1.0, value=float(params.d_in), step=0.1)
params.d_in = d_in_val
params.d_m = round(params.d_in * IN2M, 4)
col6.number_input('Diámetro (m) (auto)', min_value=0.01, value=float(params.d_m), step=0.001, format='%.4f', disabled=True)

params.Taco = st.sidebar.number_input('Taco (m)', min_value=0.0, value=float(params.Taco), step=0.1)
params.rho = st.sidebar.number_input('Densidad explosivo ρ (kg/m³)', min_value=100.0, value=float(params.rho), step=10.0)
params.P80_base = st.sidebar.number_input('P80 base (mm)', min_value=1.0, value=float(params.P80_base), step=1.0)
col7, col8, col9 = st.sidebar.columns(3)
params.n1 = col7.number_input('n1', min_value=0.0, value=float(params.n1), step=0.05)
params.n2 = col8.number_input('n2', min_value=0.0, value=float(params.n2), step=0.05)
params.n3 = col9.number_input('n3', min_value=0.0, value=float(params.n3), step=0.05)

st.sidebar.markdown('---')
st.sidebar.caption('Al cambiar el diámetro en pulgadas, el valor en metros se actualiza automáticamente (1 in = 0,0254 m).')

# -----------------------------
# Encabezado
# -----------------------------
st.title('🧨 Análisis de Malla & Fragmentación ')
st.write('Interfaz interactiva para explorar parámetros de perforación/voladura y su efecto en PF y P80. \n'
         'Por Sebastián Zúñiga Leyton - Ingeniero Civil de Minas')

# -----------------------------
# Panel: Inputs derivados (KPIs con nombres + unidades)
# -----------------------------
st.markdown('''
<style>
.kpi-grid {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;}
.kpi {border: 1px solid #e6e6e6; border-radius: 8px; padding: 10px 12px; background: #fff;}
.kpi .label {font-size: 0.85rem; color: #444; margin-bottom: 4px;}
.kpi .value {font-size: 1.4rem; font-weight: 600;}
@media (max-width: 1100px){ .kpi-grid{grid-template-columns: repeat(2,1fr);} }
</style>
''', unsafe_allow_html=True)

with st.expander('🔎 **Inputs y variables derivadas**', expanded=True):
    Vh0 = params.volumen_por_pozo()
    PF0 = params.pf()
    s_over_b, b_over_d = params.ratios()

    kpi_html = f'''
    <div class="kpi-grid">
      <div class="kpi"><div class="label">L = H + Pasadura (m)</div><div class="value">{params.L:.2f}</div></div>
      <div class="kpi"><div class="label">Longitud de carga Lc (m)</div><div class="value">{params.Lc:.2f}</div></div>
      <div class="kpi"><div class="label">Área sección A (m²)</div><div class="value">{params.A:.5f}</div></div>
      <div class="kpi"><div class="label">Carga lineal q<sub>L</sub> (kg/m)</div><div class="value">{params.qL:.3f}</div></div>
      <div class="kpi"><div class="label">Carga por pozo m<sub>h</sub> (kg)</div><div class="value">{params.m_h:.3f}</div></div>
      <div class="kpi"><div class="label">Volumen por pozo V<sub>h</sub> (m³)</div><div class="value">{Vh0:.1f}</div></div>
      <div class="kpi"><div class="label">Powder Factor PF<sub>base</sub> (kg/m³)</div><div class="value">{PF0:.6f}</div></div>
      <div class="kpi"><div class="label">Relaciones S/B y B/d (adim.)</div><div class="value">{s_over_b:.3f} | {b_over_d:.3f}</div></div>
    </div>
    '''
    st.markdown(kpi_html, unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    '📈 Escalamiento proporcional (k·B, k·S)',
    '📌 B y S manuales',
    '📊 Gráficos',
    '📤 Exportar'])

# --- Tab 1: Escalamiento proporcional ---
with tab1:
    st.subheader('Escenario: B y S escalados por un % ')
    colk1, colk2 = st.columns([2,1])
    k_pct_range = colk1.slider('Rango de k (%)', min_value=-50, max_value=150, value=(0, 100), step=1,
                               help='Porcentaje respecto de la base (k=1). Ej: 25% → k=1,25.')
    delta_pct = colk2.number_input('Paso Δk (%)', min_value=0.1, max_value=25.0, value=2.5, step=0.1,
                                   help='Incremento del % para generar la tabla.')

    # Convertimos a k
    k_min, k_max = [1 + p/100.0 for p in k_pct_range]
    paso = delta_pct / 100.0

    ks = np.round(np.arange(k_min, k_max + 1e-12, paso), 4)
    ks_pct = (ks - 1.0) * 100.0

    rows = []
    PF0 = params.pf()
    for k, kp in zip(ks, ks_pct):
        Bn = params.B * k
        Sn = params.S * k
        Vn = params.volumen_por_pozo(Bn, Sn)
        PF_n = params.pf(Bn, Sn)
        ratio = PF0 / PF_n if PF_n > 0 else np.nan
        p80_1 = params.p80(PF0, PF_n, params.n1)
        p80_2 = params.p80(PF0, PF_n, params.n2)
        p80_3 = params.p80(PF0, PF_n, params.n3)
        s_over_b_n = Sn / Bn if Bn else np.nan
        b_over_d_n = Bn / params.d_m if params.d_m else np.nan
        rows.append(dict(k=k, k_pct=kp, B_nuevo=Bn, S_nuevo=Sn, V_nuevo=Vn, PF_nuevo=PF_n, Ratio=ratio,
                         P80_n1=p80_1, P80_n2=p80_2, P80_n3=p80_3, S_over_B=s_over_b_n,
                         B_over_d=b_over_d_n, m_h=params.m_h))

    dfk = pd.DataFrame(rows)
    st.session_state['dfk'] = dfk.copy()

    st.dataframe(
        dfk.rename(columns={
            'k':'k (adim.)', 'k_pct':'k (%)',
            'B_nuevo':'B nuevo (m)', 'S_nuevo':'S nuevo (m)', 'V_nuevo':'V nuevo (m³)',
            'PF_nuevo':'PF nuevo (kg/m³)', 'Ratio':'PF_base / PF_nuevo',
            'P80_n1':'P80 n1 (mm)', 'P80_n2':'P80 n2 (mm)', 'P80_n3':'P80 n3 (mm)',
            'S_over_B':'S/B', 'B_over_d':'B/d'
        }),
        use_container_width=True, height=420
    )

# --- Tab 2: B y S manuales ---
with tab2:
    st.subheader('Escenario: ingresar **B** y **S** específicos')
    c1, c2 = st.columns(2)
    Bm = c1.number_input('Burden específico B_m (m)', min_value=0.1, value=float(params.B), step=0.1, key='Bm_input')
    Sm = c2.number_input('Espaciamiento específico S_m (m)', min_value=0.1, value=float(params.S), step=0.1, key='Sm_input')

    # Cálculos manuales
    Vm = params.volumen_por_pozo(Bm, Sm)
    PFm = params.pf(Bm, Sm)
    PF0 = params.pf()
    ratio_m = PF0 / PFm if PFm > 0 else float('nan')
    p80m_1 = params.p80(PF0, PFm, params.n1)
    p80m_2 = params.p80(PF0, PFm, params.n2)
    p80m_3 = params.p80(PF0, PFm, params.n3)
    s_over_b_m = Sm / Bm if Bm else float('nan')
    b_over_d_m = Bm / params.d_m if params.d_m else float('nan')

    m1, m2, m3 = st.columns(3)
    m1.metric('V (m³)', f"{Vm:.2f}")
    m2.metric('PF (kg/m³)', f"{PFm:.6f}")
    m3.metric('PF_base / PF', f"{ratio_m:.3f}")

    n1c, n2c, n3c = st.columns(3)
    n1c.metric('P80 n1 (mm)', f"{p80m_1:.2f}")
    n2c.metric('P80 n2 (mm)', f"{p80m_2:.2f}")
    n3c.metric('P80 n3 (mm)', f"{p80m_3:.2f}")

    r1, r2 = st.columns(2)
    r1.metric('S/B', f"{s_over_b_m:.3f}")
    r2.metric('B/d', f"{b_over_d_m:.3f}")

    st.session_state['manual_case'] = {
        'B_m (m)': Bm, 'S_m (m)': Sm, 'V (m³)': Vm, 'PF (kg/m³)': PFm, 'PF_base/PF': ratio_m,
        'P80 n1 (mm)': p80m_1, 'P80 n2 (mm)': p80m_2, 'P80 n3 (mm)': p80m_3,
        'S/B': s_over_b_m, 'B/d': b_over_d_m
    }

    st.markdown('**Resumen del caso manual**')
    st.dataframe(pd.DataFrame([st.session_state['manual_case']]), use_container_width=True)

# --- Tab 3: Gráficos ---
with tab3:
    st.subheader('Visualizaciones')
    dfk_ss = st.session_state.get('dfk', pd.DataFrame())

    if not dfk_ss.empty:
        c1, c2 = st.columns(2)
        fig1 = px.line(dfk_ss, x='k_pct', y=['PF_nuevo'], markers=True,
                       labels={'value':'PF (kg/m³)', 'k_pct':'k (%)'})
        fig1.update_layout(title='PF vs k (%)', legend_title='')
        c1.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(dfk_ss, x='k_pct', y=['P80_n1','P80_n2','P80_n3'], markers=True,
                       labels={'value':'P80 (mm)', 'k_pct':'k (%)'})
        fig2.update_layout(title='P80 vs k (%)', legend_title='')
        c2.plotly_chart(fig2, use_container_width=True)

        # --- Curva P80–PF (n2) con OLS de plotly express (requiere statsmodels) ---
        if _SM_OK:
            fig3 = px.scatter(dfk_ss, x='PF_nuevo', y='P80_n2', trendline='ols',
                              labels={'PF_nuevo':'PF (kg/m³)', 'P80_n2':'P80 (mm, n2)'})
            fig3.update_layout(title='Curva P80–PF (n2)')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.error('Para ver la recta OLS en la curva P80–PF necesitas instalar **statsmodels**:\n'
                     '```bash\npip install statsmodels\n```',
                     icon="⚠️")
    else:
        st.info('Genera la tabla en la pestaña "Escalamiento proporcional" para habilitar los gráficos.')

# --- Tab 4: Exportar ---
with tab4:
    st.subheader('Descargar resultados')
    dfk_ss = st.session_state.get('dfk', pd.DataFrame())
    manual = st.session_state.get('manual_case', None)

    if not dfk_ss.empty or manual is not None:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer:
            # Inputs
            pd.DataFrame({
                'Variable': list(asdict(params).keys()),
                'Valor': list(asdict(params).values())
            }).to_excel(writer, index=False, sheet_name='Inputs')

            # Tabla Expand_BS
            if not dfk_ss.empty:
                dfk_ss.to_excel(writer, index=False, sheet_name='Expand_BS')

            # Caso manual
            if manual is not None:
                pd.DataFrame([manual]).to_excel(writer, index=False, sheet_name='B_S_Manual')

            # README
            pd.DataFrame({'Info':["PF = m_h / (B·S·H)",
                                  "P80 = P80_base · (PF_base/PF_nuevo)^n",
                                  "k (%) controla B y S: Bn = B·k, Sn = S·k",
                                  "Archivo generado por la app Streamlit"]}).to_excel(writer, index=False, sheet_name='README')

        st.download_button('⬇️ Descargar Excel (Inputs + Expand_BS + B_S_Manual)', data=out.getvalue(),
                           file_name='Resultados_Fragmentacion.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    else:
        st.info('Genera la tabla en "Escalamiento proporcional" o calcula un caso en "B y S manuales" para habilitar la descarga.')

