# app_streamlit.py
# Requisitos:
# pip install streamlit pandas numpy matplotlib plotly openpyxl scikit-learn joblib packaging statsmodels seaborn

import os
import io
import textwrap
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import sklearn
from packaging.version import Version
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

ENEM_NOTAS_LOWER = [
    'nota_cn_ciencias_da_natureza',
    'nota_ch_ciencias_humanas',
    'nota_lc_linguagens_e_codigos',
    'nota_mt_matematica',
    'nota_redacao',
    'nota_media_5_notas'
]

# -------------------------
# Helpers (funções reutilizáveis)
# -------------------------
def make_onehot(handle_unknown='ignore'):
    try:
        if Version(sklearn.__version__) >= Version("1.2"):
            return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
        else:
            return OneHotEncoder(handle_unknown=handle_unknown, sparse=False)
    except Exception:
        try:
            return OneHotEncoder(handle_unknown=handle_unknown)
        except Exception:
            return OneHotEncoder()

@st.cache_data(show_spinner="Carregando arquivo do disco (se existir)...")
def load_enem_from_disk(path: str):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path, engine='openpyxl')
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df

def read_uploaded_file(uploaded) -> pd.DataFrame:
    try:
        if uploaded.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded, engine='openpyxl')
        else:
            df = pd.read_csv(uploaded)
        df.columns = [str(c).lower().strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Erro ao ler arquivo enviado: {e}")
        return pd.DataFrame()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def summary_stats(df, cols):
    desc = df[cols].describe().T
    desc['missing'] = df[cols].isna().sum().values
    desc = desc[['count', 'missing', 'mean', '50%', 'std', 'min', 'max']]
    desc = desc.rename(columns={'50%': 'median'})
    return desc

def build_numeric_transformer():
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42, model_dir=MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer([('num', build_numeric_transformer(), num_cols), ('cat', make_onehot(handle_unknown='ignore'), cat_cols)], remainder='drop')
    models = {'RandomForest': RandomForestRegressor(n_estimators=200, random_state=random_state), 'LinearRegression': LinearRegression(), 'SVR': SVR()}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    metrics = {}; model_paths = {}
    for name, estimator in models.items():
        pipe = Pipeline([('preproc', preprocessor), ('model', estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        metrics[name] = {'r2': float(r2), 'mse': float(mse)}
        path = os.path.join(model_dir, f'pipeline_{name}.joblib')
        joblib.dump(pipe, path)
        model_paths[name] = path
    sorted_models = sorted(metrics.items(), key=lambda x: (-x[1]['r2'], x[1]['mse']))
    best_name = sorted_models[0][0]
    best_path = model_paths[best_name]
    best_pipeline = joblib.load(best_path)
    joblib.dump(best_pipeline, os.path.join(model_dir, 'best_pipeline.joblib'))
    feature_names = X.columns.tolist()
    feature_defaults = X.median().to_dict()
    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.joblib'))
    joblib.dump(feature_defaults, os.path.join(model_dir, 'feature_defaults.joblib'))
    return {'metrics': metrics, 'best_model': best_name, 'model_paths': model_paths, 'best_path': best_path}

def load_best_pipeline(model_dir=MODEL_DIR):
    best_pipeline_path = os.path.join(model_dir, 'best_pipeline.joblib')
    if os.path.exists(best_pipeline_path):
        return joblib.load(best_pipeline_path)
    return None

def load_feature_metadata(model_dir=MODEL_DIR):
    fn_path = os.path.join(model_dir, 'feature_names.joblib')
    fd_path = os.path.join(model_dir, 'feature_defaults.joblib')
    if os.path.exists(fn_path) and os.path.exists(fd_path):
        return joblib.load(fn_path), joblib.load(fd_path)
    return None, None

def stepwise_selection(X, y, threshold_in=0.01, threshold_out=0.05):
    included = []
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for col in excluded:
            try:
                model = sm.OLS(y, sm.add_constant(X[included + [col]])).fit()
                new_pval[col] = model.pvalues[col]
            except Exception:
                new_pval[col] = np.nan
        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
        if included:
            model = sm.OLS(y, sm.add_constant(X[included])).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                removed = pvalues.idxmax()
                included.remove(removed)
                changed = True
        if not changed:
            break
    return included

def evaluate_features(df, features, target, k=5, random_state=42, class_threshold=None):
    X = df[features].values; y = df[target].values
    thr = np.median(y) if class_threshold is None else class_threshold
    y_bin = (y >= thr).astype(int)
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    rmse_list, r2_list = [], []
    roc_list, acc_list, prec_list, recall_list, f1_list, spec_list = [], [], [], [], [], []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        ybin_te = y_bin[test_idx]
        lr = LinearRegression(); lr.fit(X_tr, y_tr); y_pred = lr.predict(X_te)
        rmse_list.append(np.sqrt(mean_squared_error(y_te, y_pred))); r2_list.append(r2_score(y_te, y_pred))
        try: roc_list.append(roc_auc_score(ybin_te, y_pred))
        except Exception: roc_list.append(np.nan)
        y_pred_bin = (y_pred >= thr).astype(int)
        acc_list.append(accuracy_score(ybin_te, y_pred_bin))
        prec_list.append(precision_score(ybin_te, y_pred_bin, zero_division=0))
        recall_list.append(recall_score(ybin_te, y_pred_bin, zero_division=0))
        f1_list.append(f1_score(ybin_te, y_pred_bin, zero_division=0))
        cm = confusion_matrix(ybin_te, y_pred_bin, labels=[0,1])
        spec = (cm.ravel()[0] / (cm.ravel()[0] + cm.ravel()[1])) if cm.size==4 and (cm.ravel()[0]+cm.ravel()[1])>0 else np.nan
        spec_list.append(spec)
    results = {'RMSE_mean': np.nanmean(rmse_list),'RMSE_std': np.nanstd(rmse_list),'R2_mean': np.nanmean(r2_list),'R2_std': np.nanstd(r2_list),'ROC_AUC_mean': np.nanmean(roc_list),'ACC_mean': np.nanmean(acc_list),'PREC_mean': np.nanmean(prec_list),'RECALL_mean': np.nanmean(recall_list),'F1_mean': np.nanmean(f1_list),'SPEC_mean': np.nanmean(spec_list),'threshold_used': thr}
    return results

# -------------------------
# UI: uploader + tabs (somente 2)
# -------------------------
st.markdown("<h1 style='color:#0b4a6f'>ENEM 2024 — Regressão & Diagnósticos</h1>", unsafe_allow_html=True)
st.markdown("Faça upload do seu arquivo `.xlsx`/`.csv` no uploader abaixo — depois vá para a aba 'Regressão & Diagnósticos' e clique em executar.")

# carregar default se existir
DEFAULT_ENEM = os.path.join("data", "raw", "Enem_2024_Amostra_Perfeita.xlsx")
if 'df_enem' not in st.session_state:
    st.session_state['df_enem'] = load_enem_from_disk(DEFAULT_ENEM)

col_up1, col_up2 = st.columns([3,1])
with col_up1:
    uploaded = st.file_uploader("Upload do arquivo (xlsx/csv) — sobrescreve sessão", type=["xlsx","csv","xls"])
with col_up2:
    if st.button("Limpar dataset em sessão"):
        st.session_state['df_enem'] = pd.DataFrame()
        st.success("Dataset em sessão limpo.")

if uploaded is not None:
    df_uploaded = read_uploaded_file(uploaded)
    if not df_uploaded.empty:
        st.session_state['df_enem'] = df_uploaded
        st.success("Arquivo carregado e salvo na sessão.")

tab_intro, tab_reg = st.tabs(["Introdução (robusta)", "Regressão & Diagnósticos"])

# -------------------------
# Introdução robusta (nova)
# -------------------------
with tab_intro:
    st.header("Introdução — objetivo, fluxo e interpretação")
    st.markdown("Esta aplicação é uma interface interativa para explorar relações entre notas do ENEM (amostra fornecida) e construir/avaliar modelos de regressão linear. Leia com atenção antes de executar a análise.")
    st.markdown("### 1) Objetivo")
    st.markdown(textwrap.dedent("""
    - Fornecer uma ferramenta reprodutível para **explorar** correlações entre notas (ex.: Matemática vs Redação/Ciências).
    - Construir modelos lineares (univariado e multivariado via seleção *stepwise*) e **diagnosticar** suposições do modelo.
    - Comparar modelos com AIC/BIC e validação cruzada (K-fold).
    """))
    st.markdown("### 2) Fluxo de uso (passo-a-passo)")
    st.markdown(textwrap.dedent("""
    1. Faça upload do `.xlsx` ou `.csv`. O app normaliza nomes das colunas para `lowercase`.
    2. Na aba **Regressão & Diagnósticos**, selecione target (por padrão tentamos detectar Matemática) e features.
    3. Opcionalmente limite a amostra para testes rápidos.
    4. Ajuste `K-fold` e `p-in`/`p-out` do stepwise.
    5. Clique em **Executar análise completa**. O app exibirá gráficos, sumários, testes, VIF, influência e CV.
    """))
    st.markdown("### 3) Metodologia e avisos")
    st.markdown(textwrap.dedent("""
    - **Stepwise** é um procedimento exploratório; não substitui avaliação substantiva.
    - **Diagnósticos**: verificamos heterocedasticidade (Breusch-Pagan), independência (Durbin-Watson), normalidade (Q-Q) e multicolinearidade (VIF).
    - **Validação**: usamos K-fold CV para estimar RMSE e R². Para relações não-lineares considere Random Forest/SVR (incluídos nos pipelines).
    - **Limitações**: conclusões dependem da qualidade e representatividade da amostra.
    """))

# -------------------------
# Regressão & Diagnósticos
# -------------------------
with tab_reg:
    st.header("Regressão linear — seleção stepwise, diagnósticos e comparação")
    st.markdown("Em cada seção há explicação sobre o que o gráfico/tabela mostra e como interpretar. Os gráficos de influência (DFFITS / DFBETAS / Cook) são desenhados de forma robusta (fallback automático se necessário).")

    df = st.session_state.get('df_enem', pd.DataFrame())
    if df.empty:
        st.info("Nenhum dataset carregado. Faça upload no topo desta página.")
        st.stop()

    df.columns = [str(c).lower().strip() for c in df.columns]

    # target detection
    Y_candidates = [c for c in df.columns if 'nota_mt' in c or 'matem' in c]
    if not Y_candidates:
        st.warning("Não encontrei coluna com 'nota_mt' automaticamente. Selecione manualmente abaixo.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("Não há colunas numéricas detectadas — verifique seu arquivo.")
            st.stop()
        Y = st.selectbox("Escolha a variável alvo (target)", options=numeric_cols)
    else:
        Y = st.selectbox("Variável alvo (target) detectada", options=Y_candidates, index=0)

    suggested_X = [c for c in df.columns if c != Y and ('nota' in c or c in ENEM_NOTAS_LOWER)]
    if not suggested_X:
        all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        suggested_X = [c for c in all_numeric if c != Y]
    chosen_X = st.multiselect("Escolha variáveis explicativas (X)", options=suggested_X, default=suggested_X[:4])

    st.markdown("### Controles")
    col1, col2, col3 = st.columns(3)
    with col1:
        sample_max = st.number_input("Amostra (máx. observações a usar — 0 = todas)", min_value=0, value=0, step=100)
    with col2:
        kfold = st.number_input("K-fold (CV)", min_value=2, max_value=20, value=5, step=1)
    with col3:
        pval_in = st.number_input("p-in (stepwise)", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f")
    pval_out = st.number_input("p-out (stepwise)", min_value=0.001, max_value=0.2, value=0.05, format="%.4f")

    run_button = st.button("Executar análise completa (stepwise, diagnósticos, CV)")

    if run_button:
        if not chosen_X:
            st.error("Escolha ao menos 1 variável explicativa.")
            st.stop()

        with st.spinner("Executando análise... (pode demorar alguns segundos)"):
            use_df = df[[Y] + chosen_X].copy()
            for c in use_df.columns:
                use_df[c] = pd.to_numeric(use_df[c], errors='coerce')
            if sample_max and sample_max > 0 and use_df.shape[0] > sample_max:
                use_df = use_df.sample(int(sample_max), random_state=42).reset_index(drop=True)
            use_df = use_df.dropna().reset_index(drop=True)
            if use_df.shape[0] < 10:
                st.error("Poucas observações válidas após limpeza (<10). Aumente a amostra.")
                st.stop()

            # Heatmap + explicação
            st.subheader("1) Heatmap de Correlação")
            st.markdown("**O que é:** matriz de correlações Pearson entre todas as variáveis do subset. Valores próximos a +1/-1 indicam correlação forte.")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(use_df.corr(), annot=True, cmap='coolwarm', ax=ax, fmt=".2f", annot_kws={"size":10})
            ax.set_title("Matriz de Correlação (subset)", fontsize=14)
            st.pyplot(fig)

            # Pearson table + explicação
            st.subheader("2) Correlação (Pearson) entre target e cada X")
            st.markdown("**O que é:** coeficiente r de Pearson e p-valor. r próximo de ±1 indica associação linear forte; p pequeno (ex.: <0.05) indica significância estatística.")
            pearson_rows = []
            for col in chosen_X:
                try:
                    r, p = pearsonr(use_df[Y], use_df[col])
                except Exception:
                    r, p = np.nan, np.nan
                pearson_rows.append({'variavel': col, 'r': float(np.round(r, 4)) if not np.isnan(r) else np.nan, 'pvalor': float(np.round(p,6)) if not np.isnan(p) else np.nan})
            pearson_df = pd.DataFrame(pearson_rows).sort_values('r', ascending=False)
            st.dataframe(pearson_df, use_container_width=True)

            # Stepwise + explicação
            st.subheader("3) Seleção stepwise (forward & backward)")
            st.markdown("**O que faz:** adiciona/remove variáveis com base em p-values (p-in/p-out). É automatizado — avalie sentido prático das variáveis selecionadas.")
            selected_vars = stepwise_selection(use_df[chosen_X], use_df[Y], threshold_in=float(pval_in), threshold_out=float(pval_out))
            if len(selected_vars) == 0:
                selected_vars = chosen_X.copy()
            st.success(f"Variáveis selecionadas: {selected_vars}")

            # Models + sumários
            corrs = use_df.corr()[Y].abs().sort_values(ascending=False)
            top_var = corrs.index[1] if len(corrs) > 1 else selected_vars[0]
            X1 = sm.add_constant(use_df[[top_var]])
            modelo1 = sm.OLS(use_df[Y], X1).fit()
            X2 = sm.add_constant(use_df[selected_vars])
            modelo2 = sm.OLS(use_df[Y], X2).fit()

            with st.expander("Sumário Modelo 1 — Univariado (abrir p/ texto completo)", expanded=False):
                st.text(modelo1.summary().as_text())
            with st.expander("Sumário Modelo 2 — Multivariado (abrir p/ texto completo)", expanded=False):
                st.text(modelo2.summary().as_text())

            # Key stats + explicação
            st.subheader("4) Estatísticas chave dos modelos (AIC / BIC / R² / RMSE)")
            st.markdown("AIC/BIC penalizam complexidade; R² explica proporção da variância; RMSE mede erro médio quadrático.")
            model_stats = pd.DataFrame([
                {'modelo': 'Modelo_1_univ', 'features': top_var, 'AIC': modelo1.aic, 'BIC': modelo1.bic, 'R2': modelo1.rsquared, 'R2_adj': modelo1.rsquared_adj, 'RMSE': np.sqrt(mean_squared_error(modelo1.model.endog, modelo1.fittedvalues))},
                {'modelo': 'Modelo_2_multiv', 'features': ",".join(selected_vars), 'AIC': modelo2.aic, 'BIC': modelo2.bic, 'R2': modelo2.rsquared, 'R2_adj': modelo2.rsquared_adj, 'RMSE': np.sqrt(mean_squared_error(modelo2.model.endog, modelo2.fittedvalues))}
            ])
            st.dataframe(model_stats.style.format({'AIC':'{:.2f}','BIC':'{:.2f}','R2':'{:.4f}','R2_adj':'{:.4f}','RMSE':'{:.4f}'}), use_container_width=True)

            # Residual diagnostics + explicações
            fitted = modelo2.fittedvalues; residuals = modelo2.resid
            st.subheader("5) Diagnósticos de resíduos — Resíduos vs Ajustados")
            st.markdown("Resíduos aleatórios em torno de zero são desejáveis. Padrões (funil, curva) indicam problemas de especificação ou heterocedasticidade.")
            colg1, colg2 = st.columns(2)
            with colg1:
                fig_r, axr = plt.subplots(figsize=(6,4))
                axr.scatter(fitted, residuals, alpha=0.6)
                axr.axhline(0, color='red', linewidth=1)
                axr.set_xlabel("Valores Ajustados"); axr.set_ylabel("Resíduos"); axr.set_title("Resíduos vs Ajustados")
                st.pyplot(fig_r)
            with colg2:
                st.subheader("Distribuição dos resíduos")
                st.markdown("Histogram + KDE para avaliar aproximação à normalidade.")
                fig_h, axh = plt.subplots(figsize=(6,4))
                sns.histplot(residuals, kde=True, ax=axh); axh.set_title("Distribuição dos Resíduos")
                st.pyplot(fig_h)
            with st.expander("Q-Q plot (resíduos) — interpretação", expanded=False):
                st.markdown("Q-Q plot compara quantis dos resíduos com quantis teóricos da normal. Desvios significativos indicam não-normalidade.")
                fig_qq = sm.qqplot(residuals, line='s'); st.pyplot(fig_qq)

            # Tests explanation + values
            st.subheader("6) Testes: Breusch-Pagan (heterocedasticidade) e Durbin-Watson (autocorrelação)")
            try:
                bp = het_breuschpagan(residuals, modelo2.model.exog)
                bp_names = ['LM stat', 'p-value', 'f-value', 'f p-value']
                bp_dict = dict(zip(bp_names, bp))
                st.write("Breusch-Pagan:", {k: float(np.round(v,6)) for k,v in bp_dict.items()})
                st.markdown("Interpretação: p-value pequeno indica heterocedasticidade (variância dos resíduos não constante).")
            except Exception as e:
                st.write("Erro Breusch-Pagan:", e)
            st.write("Durbin-Watson:", float(np.round(durbin_watson(residuals),4)))
            st.markdown("Interpretação: valores ~2 indicam ausência de autocorrelação; valores muito <2 ou >2 merecem atenção.")

            # VIF + explicação
            st.subheader("7) VIF — multicolinearidade")
            st.markdown("VIF > 5 (ou 10) sugere multicolinearidade problemática; considere remover variáveis correlacionadas.")
            try:
                vif_df = pd.DataFrame({'Variavel': selected_vars, 'VIF': [variance_inflation_factor(use_df[selected_vars].values, i) for i in range(len(selected_vars))]})
                st.dataframe(vif_df.style.format({'VIF':'{:.4f}'}), use_container_width=True)
                st.download_button("Baixar VIF (CSV)", data=df_to_csv_bytes(vif_df), file_name="vif.csv", mime="text/csv")
            except Exception as e:
                st.write("Erro calculando VIF:", e)

            # Influence + explain + robust plotting (DFFITS, DFBETAS, Cook)
            st.subheader("8) Análise de Influência — DFFITS / DFBETAS / Cook's D (gráficos)")
            st.markdown("Identifica observações que têm impacto desproporcional nos coeficientes ou no ajuste. Investigue registros acima dos limiares antes de excluir.")

            influence = modelo2.get_influence()
            # DFFITS
            try:
                dffits_vals, _ = influence.dffits
                # sanitizar e converter para arrays numéricos
                x_raw = np.arange(len(dffits_vals))
                x = np.array(pd.to_numeric(pd.Series(x_raw), errors='coerce'))
                y = np.array(pd.to_numeric(pd.Series(dffits_vals), errors='coerce'))
                valid_mask = (~np.isnan(x)) & (~np.isnan(y))
                x = x[valid_mask]; y = y[valid_mask]
                n = int(modelo2.nobs); p = int(modelo2.df_model) + 1
                limiar_dffits = 2 * np.sqrt(p) / np.sqrt(n)
                st.write(f"Limiar DFFITS: {limiar_dffits:.4f}")
                if x.size == 0:
                    st.write("Nenhum valor válido para DFFITS.")
                else:
                    fig_dfits, axd = plt.subplots(figsize=(10,3))
                    try:
                        axd.stem(x, y, basefmt=" ", use_line_collection=True)
                    except Exception:
                        axd.vlines(x, ymin=0, ymax=y, linewidth=1)
                        axd.scatter(x, y, s=10)
                    axd.axhline(limiar_dffits, color='red', linestyle='--', label=f'limiar + ({limiar_dffits:.4f})')
                    axd.axhline(-limiar_dffits, color='red', linestyle='--', label=f'limiar - ({-limiar_dffits:.4f})')
                    axd.set_xlabel("Índice da observação"); axd.set_ylabel("DFFITS"); axd.set_title("DFFITS por observação")
                    axd.legend(loc='upper right')
                    st.pyplot(fig_dfits)
                    # tabela de outliers DFFITS
                    dffits_df = pd.DataFrame({'index': np.arange(len(dffits_vals)), 'DFFITS': dffits_vals})
                    out_dffits = dffits_df[np.abs(dffits_df['DFFITS']) > limiar_dffits]
                    st.write("Observações com |DFFITS| > limiar:")
                    st.dataframe(out_dffits, use_container_width=True)
            except Exception as e:
                st.write("Erro ao calcular/plotar DFFITS:", e)

            # DFBETAS: heatmap of absolute dfbetas, plus top-per-observation stem (max abs)
            try:
                dfbetas = influence.dfbetas  # shape (n_obs, n_params)
                # garantir numerico
                dfbetas_df = pd.DataFrame(dfbetas, columns=modelo2.params.index)
                abs_dfbetas = dfbetas_df.abs()
                st.write("Heatmap (valores absolutos) de DFBETAS por observação x parâmetro:")
                fig_hdb, axhdb = plt.subplots(figsize=(10, max(3, min(12, abs_dfbetas.shape[0]*0.2))))
                sns.heatmap(abs_dfbetas.T, cmap='Reds', cbar_kws={'label': 'abs(DFBETAS)'}, ax=axhdb)
                axhdb.set_xlabel("Índice da observação")
                axhdb.set_ylabel("Parâmetros (coef.)")
                axhdb.set_title("Heatmap abs(DFBETAS)")
                st.pyplot(fig_hdb)
                # stem do máximo absoluto por observação (identifica observações com algum coeficiente com alto impacto)
                max_abs_per_obs = abs_dfbetas.max(axis=1).values
                x = np.arange(len(max_abs_per_obs))
                y = np.array(pd.to_numeric(pd.Series(max_abs_per_obs), errors='coerce'))
                valid_mask = ~np.isnan(y)
                x = x[valid_mask]; y = y[valid_mask]
                limiar_dfbetas = 2 / np.sqrt(int(modelo2.nobs))
                st.write(f"Limiar DFBETAS (heurístico): {limiar_dfbetas:.4f}")
                if x.size == 0:
                    st.write("Nenhum valor válido para DFBETAS.")
                else:
                    fig_db, axdb = plt.subplots(figsize=(10,3))
                    try:
                        axdb.stem(x, y, basefmt=" ", use_line_collection=True)
                    except Exception:
                        axdb.vlines(x, ymin=0, ymax=y, linewidth=1)
                        axdb.scatter(x, y, s=10)
                    axdb.axhline(limiar_dfbetas, color='red', linestyle='--', label=f'limiar ({limiar_dfbetas:.4f})')
                    axdb.set_xlabel("Índice da observação"); axdb.set_ylabel("max(abs(DFBETAS))"); axdb.set_title("Máximo abs(DFBETAS) por observação")
                    axdb.legend(loc='upper right')
                    st.pyplot(fig_db)
                    out_dfbetas = dfbetas_df[(abs_dfbetas > limiar_dfbetas).any(axis=1)]
                    st.write("Observações com algum |DFBETAS| > limiar (tabela):")
                    st.dataframe(out_dfbetas, use_container_width=True)
            except Exception as e:
                st.write("Erro ao calcular/plotar DFBETAS:", e)

            # Cook's distance
            try:
                cooks_d = influence.cooks_distance[0]
                cooks_df = pd.DataFrame({'index': np.arange(len(cooks_d)), 'Cooks_D': cooks_d})
                # sanitizar
                x_raw = cooks_df['index']
                y_raw = cooks_df['Cooks_D']
                x = pd.to_numeric(x_raw, errors='coerce').to_numpy()
                y = pd.to_numeric(y_raw, errors='coerce').to_numpy()
                valid_mask = (~np.isnan(x)) & (~np.isnan(y))
                x = x[valid_mask]; y = y[valid_mask]
                n = int(modelo2.nobs)
                limiar_cook = 4 / n
                st.write(f"Limiar Cook: {limiar_cook:.6f}")
                if x.size == 0:
                    st.write("Nenhum valor válido para Cook's D.")
                else:
                    fig_cook, axc = plt.subplots(figsize=(10,3))
                    try:
                        axc.stem(x, y, basefmt=" ", use_line_collection=True)
                    except Exception:
                        axc.vlines(x, ymin=0, ymax=y, linewidth=1)
                        axc.scatter(x, y, s=10)
                    axc.axhline(limiar_cook, color='red', linestyle='--', label=f'limiar ({limiar_cook:.6f})')
                    axc.set_xlabel("Índice da observação"); axc.set_ylabel("Cook's D"); axc.set_title("Cook's Distance por observação")
                    axc.legend(loc='upper right')
                    st.pyplot(fig_cook)
                    out_cook = cooks_df[cooks_df['Cooks_D'] > limiar_cook]
                    st.write("Observações com Cook's D > limiar:")
                    st.dataframe(out_cook, use_container_width=True)
            except Exception as e:
                st.write("Erro ao calcular/plotar Cook's D:", e)

            # CV + AIC/BIC + explicação
            st.subheader("9) Avaliação comparativa — K-fold CV + AIC/BIC")
            st.markdown("Comparamos um modelo 'top3 por correlação' com o modelo stepwise usando AIC/BIC e métricas médias em K-fold CV (RMSE, R², AUC-ROC binarizado).")
            df6 = use_df.copy()
            corrs = df6.corr()[Y].abs().sort_values(ascending=False)
            modelo1_feats = list(corrs.index[1:4]) if len(corrs) > 1 else [top_var]
            modelo2_feats = selected_vars if selected_vars else modelo1_feats
            m1_cv = evaluate_features(df6, modelo1_feats, Y, k=int(kfold))
            m2_cv = evaluate_features(df6, modelo2_feats, Y, k=int(kfold))
            m1_sm = sm.OLS(df6[Y], sm.add_constant(df6[modelo1_feats])).fit()
            m2_sm = sm.OLS(df6[Y], sm.add_constant(df6[modelo2_feats])).fit()
            m1_aic, m1_bic = m1_sm.aic, m1_sm.bic
            m2_aic, m2_bic = m2_sm.aic, m2_sm.bic
            summary = pd.DataFrame([
                {'modelo': 'Modelo_1_top3_corr', 'features': ','.join(modelo1_feats), 'AIC': m1_aic, 'BIC': m1_bic, 'CV_RMSE_mean': m1_cv['RMSE_mean'], 'CV_R2_mean': m1_cv['R2_mean'], 'CV_ROC_AUC_mean': m1_cv['ROC_AUC_mean']},
                {'modelo': 'Modelo_2_stepwise', 'features': ','.join(modelo2_feats), 'AIC': m2_aic, 'BIC': m2_bic, 'CV_RMSE_mean': m2_cv['RMSE_mean'], 'CV_R2_mean': m2_cv['R2_mean'], 'CV_ROC_AUC_mean': m2_cv['ROC_AUC_mean']}
            ])
            st.dataframe(summary.style.format({'AIC':'{:.2f}','BIC':'{:.2f}','CV_RMSE_mean':'{:.4f}','CV_R2_mean':'{:.4f}','CV_ROC_AUC_mean':'{:.4f}'}), use_container_width=True)
            st.download_button("Baixar resumo comparativo (CSV)", data=df_to_csv_bytes(summary), file_name="summary_comparativo.csv", mime="text/csv")

            # Conclusão automática curta
            st.markdown("**Conclusão automática (resumida):**")
            if m2_aic < m1_aic:
                st.write("→ Modelo 2 (Stepwise) possui menor AIC.")
            else:
                st.write("→ Modelo 1 possui menor AIC.")
            if m2_cv['RMSE_mean'] < m1_cv['RMSE_mean']:
                st.write("→ Modelo 2 possui menor RMSE em CV.")
            else:
                st.write("→ Modelo 1 possui menor RMSE em CV.")
            if m2_cv['ROC_AUC_mean'] > m1_cv['ROC_AUC_mean']:
                st.write("→ Modelo 2 possui maior AUC-ROC média.")
            else:
                st.write("→ Modelo 1 possui maior AUC-ROC média.")
            st.success("Análise concluída — use os botões de download para exportar resultados.")

# Fim do app
