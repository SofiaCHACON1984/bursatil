# FINTECH – Valuación y desempeño de activos

Aplicación en **Python + Streamlit** para calcular indicadores de desempeño y riesgo de activos financieros a partir de precios históricos de Yahoo Finance.

## Indicadores

- Rentabilidad anualizada
- Volatilidad anualizada
- Índice Sharpe
- Índice Treynor
- Correlación de Pearson
- Beta
- CAPM
- Alpha
- Valor z
- VaR %
- VaR $$
- Regresión vs. índice de referencia
- Matriz de correlación

## Inputs

- Número de activos
- Tickers
- Índice bursátil de referencia
- País de la tasa libre de riesgo
- Tasa libre de riesgo anual
- Capital a invertir
- Confianza VaR: 90%, 95% o 99%
- VaR: 1 día o 1 mes
- Periodicidad: diaria, semanal o mensual
- Plazo: 5 días, 3 meses, 6 meses, YTD, 12 meses, 1 año o 5 años

## Ejecutar localmente

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Instalar:

```bash
pip install -r requirements.txt
```

Ejecutar:

```bash
streamlit run app.py
```

## Publicar en GitHub + Streamlit Community Cloud

1. Crear un repositorio en GitHub.
2. Subir `app.py`, `requirements.txt` y `README.md`.
3. En Streamlit Community Cloud seleccionar el repositorio.
4. Archivo principal: `app.py`.
5. Ejecutar/deploy.

## Nota metodológica

Las fórmulas siguen el documento académico anexo:

- Retorno anual: `(Valor Final / Valor Inicial)^(1/n) - 1`
- Volatilidad anual: `σ × √n`
- Sharpe: `(Rp - Rf) / σp`
- Treynor: `(Ra - Rf) / βa`
- Beta: `Cov(Ri,Rm) / Var(Rm)`
- CAPM: `Rf + βi(Rm - Rf)`
- Alpha: `Ri - [Rf + βi(Rm - Rf)]`
- VaR: `μ + zα σ`

Para presentar VaR como **pérdida positiva**, la aplicación toma el negativo del retorno adverso y lo limita a cero. Para 1 mes se utiliza un horizonte de 21 sesiones de mercado.

La tasa libre de riesgo se captura como input para que el usuario pueda utilizar la tasa correspondiente al país de origen y al periodo de valuación, evitando hardcodear una tasa potencialmente desactualizada.
