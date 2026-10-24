# -*- coding: utf-8 -*-
"""
Póster Econometría II - Metodología Box-Jenkins 
Serie: Residential Property Prices for Colombia

Librerias que requiere: pandas numpy matplotlib scipy statsmodels openpyxl
"""
# =================================================================
# PAQUETES
# =================================================================
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =================================================================
# 1. RUTAS RELATIVAS 
# =================================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:                 
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "datos"
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)
ruta_datos = DATA_DIR / "Datos_Colombia_FRED.xlsx"

# =================================================================
# 2. CARGA DE DATOS (se modela la serie TAL CUAL, sin log ni desplazamiento)
# =================================================================
df = pd.read_excel(ruta_datos)
df = df.iloc[:, :2]
df.columns = ["fecha", "dato"]
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.set_index("fecha").sort_index()
df.index.freq = "QS"
# se filtra explícitamente desde el año 2000
y = df.loc["2000-01-01":, "dato"].astype(float)
y.name = "Variación anual de precios de vivienda (%)"
print(f"T = {len(y)} observaciones: {y.index[0].date()} a {y.index[-1].date()}")

# La serie ya es una tasa de variación anual y toma valores negativos;
# el log con constante de desplazamiento (+3.74) no tiene interpretación
# económica y no es necesario para ARIMA. Se trabaja entonces en niveles originales.

# =================================================================
# 3. IDENTIFICACIÓN
# =================================================================
# 3.1 Gráfica de la serie
fig, ax = plt.subplots(figsize=(11, 4.5))
y.plot(ax=ax, color="#1f4e79", lw=2)
ax.axhline(y.mean(), color="gray", ls="--", label=f"Media = {y.mean():.2f}")
ax.set_title("Precios de vivienda en Colombia: variación anual (%)")
ax.set_xlabel("")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "01_serie.png", dpi=200)
plt.show()


# 3.2 Pruebas de raíz unitaria: en NIVEL y en PRIMERA DIFERENCIA
def reporta(x, nombre):
    adf_c = adfuller(x, regression="c", autolag="AIC")
    adf_ct = adfuller(x, regression="ct", autolag="AIC")
    kp = kpss(x, regression="c", nlags="auto")
    print(f"{nombre:<14} ADF(c) p={adf_c[1]:.4f} | ADF(c+t) p={adf_ct[1]:.4f} | KPSS p={kp[1]:.4f}")


print("\n=========== PRUEBAS DE RAÍZ UNITARIA ===========")
print("H0 ADF: raíz unitaria | H0 KPSS: estacionaria")
reporta(y, "Nivel")
reporta(y.diff().dropna(), "1ª diferencia")
print("-> ADF no rechaza raíz unitaria en nivel y sí en 1ª diferencia: d = 1.")
print("   (KPSS no rechaza estacionariedad en ninguno: las pruebas no coinciden;")
print("    se justifica d=1 por la alta persistencia de la serie.)")
D = 1

# 3.3 ACF y PACF 
fig, axs = plt.subplots(2, 2, figsize=(13, 8))
plot_acf(y, lags=16, ax=axs[0, 0], title="ACF - nivel")
plot_pacf(y, lags=16, ax=axs[0, 1], title="PACF - nivel", method="ywm")
plot_acf(y.diff().dropna(), lags=16, ax=axs[1, 0], title="ACF - primera diferencia")
plot_pacf(y.diff().dropna(), lags=16, ax=axs[1, 1], title="PACF - primera diferencia", method="ywm")
fig.tight_layout()
fig.savefig(OUT_DIR / "02_acf_pacf.png", dpi=200)
plt.show()
# Interpretación: en la primera diferencia la ACF tiene un pico fuerte en el rezago 4 (-0.37).
# Se espera que la serie es una variación ANUAL (4 trimestres) y eso induce
# estructura MA en el rezago 4.

# =================================================================
# 4. ESTIMACIÓN: se comparan candidatos 
# =================================================================
candidatos = {
    "ARIMA(1,1,0)  [su modelo original]": (1, D, 0),
    "ARIMA(0,1,1)": (0, D, 1),
    "ARIMA(2,1,0)": (2, D, 0),
    "ARIMA(1,1,1)": (1, D, 1),
    "ARIMA(2,1,2)": (2, D, 2),
    "ARIMA(1,1,4)": (1, D, 4),
    "ARIMA(0,1,[4])": (0, D, [0, 0, 0, 1]),
    "ARIMA(1,1,[4])": (1, D, [0, 0, 0, 1]),
    "ARIMA(1,1,[3,4])": (1, D, [0, 0, 1, 1]),
}
filas = []
for nombre, orden in candidatos.items():
    # CAMBIO: se quitan enforce_stationarity=False / enforce_invertibility=False
    r = SARIMAX(y, order=orden, trend="n").fit(disp=False)
    k = len(r.params) - 1
    lb8 = acorr_ljungbox(r.resid.iloc[D:], lags=[8], model_df=k)["lb_pvalue"].iloc[0]
    filas.append([nombre, k, r.aic, r.bic, lb8, r.pvalues.drop("sigma2").max()])
comp = pd.DataFrame(filas, columns=["Modelo", "k", "AIC", "BIC", "p LB(8)", "max p coef."]).sort_values("BIC")
print("\n=========== COMPARACIÓN DE MODELOS (ordenados por BIC) ===========")
print(comp.round(3).to_string(index=False))
comp.round(3).to_csv(OUT_DIR / "tabla_comparacion_modelos.csv", index=False)

# Modelo elegido: AR(1) + MA en el rezago 4 (mínimo BIC, coeficientes
# significativos y residuos sin autocorrelación).
ORDEN = (1, D, [0, 0, 0, 1])
ETIQ = "ARIMA(1,1,[4])"
modelo = SARIMAX(y, order=ORDEN, trend="n").fit(disp=False)
print(f"\n=================== RESULTADOS DEL MODELO {ETIQ} ===================")
print(modelo.summary())
print("Módulo de raíces AR:", np.abs(modelo.arroots).round(3), "(deben ser > 1)")
print("Módulo de raíces MA:", np.abs(modelo.maroots).round(3), "(deben ser > 1)")

# =================================================================
# 5. VALIDACIÓN DE SUPUESTOS
# =================================================================
# Se descarta el primer residuo (inicialización de la diferenciación)
# y Ljung-Box usa model_df = nº de parámetros ARMA.
residuos = modelo.resid.iloc[D:]
residuos_cuadrado = residuos**2
k = len(modelo.params) - 1

print("\n======================== PRUEBAS DE SUPUESTOS ========================")
for lag in (6, 8, 12):
    p = acorr_ljungbox(residuos, lags=[lag], model_df=k)["lb_pvalue"].iloc[0]
    print(f"Ljung-Box ({lag:>2}) autocorrelación   - p-valor: {p:.4f}")
print(f"ARCH-LM (4) heterocedasticidad   - p-valor: {het_arch(residuos, nlags=4)[1]:.4f}")
print(f"Jarque-Bera normalidad           - p-valor: {jarque_bera(residuos)[1]:.4f}")
print(f"Shapiro-Wilk normalidad          - p-valor: {stats.shapiro(residuos)[1]:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
plot_acf(residuos, ax=axes[0, 0], title="FAC de los residuales", lags=16)
plot_acf(residuos_cuadrado, ax=axes[0, 1], title="FAC residuales al cuadrado", lags=16)
axes[1, 0].hist(residuos, bins=20, density=True, color="lightgray", edgecolor="black")
xs = np.linspace(residuos.min(), residuos.max(), 200)
axes[1, 0].plot(xs, stats.norm.pdf(xs, residuos.mean(), residuos.std()), "r")
axes[1, 0].set_title("Histograma y normal ajustada")
stats.probplot(residuos, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("QQ-Plot de los residuales")
fig.tight_layout()
fig.savefig(OUT_DIR / "03_diagnostico_residuos.png", dpi=200)
plt.show()

# =================================================================
# 6. PRONÓSTICO A 10 PERIODOS (en las unidades originales: %)
# =================================================================
pred = modelo.get_forecast(steps=10)
media = pred.predicted_mean
ic = pred.conf_int(alpha=0.05)
tabla = pd.DataFrame({"Pronóstico": media, "IC95 inf.": ic.iloc[:, 0], "IC95 sup.": ic.iloc[:, 1]})
print("\n=================== PRONÓSTICO 10 PERIODOS ===================")
print(tabla.round(2).to_string())
tabla.round(3).to_csv(OUT_DIR / "tabla_pronostico.csv")

fig, ax = plt.subplots(figsize=(11, 5))
y.loc["2015":].plot(ax=ax, color="black", lw=1.8, label="Histórico")
media.plot(ax=ax, color="red", lw=2.2, label="Pronóstico")
ax.fill_between(ic.index, ic.iloc[:, 0], ic.iloc[:, 1], color="red", alpha=0.2, label="IC 95 %")
ax.set_title(f"Pronóstico {ETIQ} - Precios de vivienda en Colombia (variación anual, %)")
ax.set_xlabel("")
ax.set_ylabel("%")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "04_pronostico.png", dpi=200)
plt.show()
