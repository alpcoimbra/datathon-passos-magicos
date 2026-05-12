import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# ======================
# CONFIGURAÇÃO DA PÁGINA
# ======================

st.set_page_config(
    page_title="Previsão de Risco de Defasagem",
    layout="centered"
)


# ======================
# CARREGAMENTO DO MODELO E DADOS
# ======================

@st.cache_resource
def load_model():
    return joblib.load("models/modelo_risco.pkl")


@st.cache_data
def load_model_data():
    return pd.read_csv("data/gold/base_modelo_risco.csv")


@st.cache_data
def load_gold_data():
    return pd.read_csv("data/gold/base_gold_analitica.csv")


model = load_model()


# ======================
# ABAS
# ======================

tab1, tab2 = st.tabs(["Predição", "Insights Analíticos"])


# ======================
# ABA 1 - PREDIÇÃO
# ======================

with tab1:
    st.title("Sistema Preditivo de Risco de Defasagem")
    st.markdown(
        "Preencha os indicadores do aluno para estimar a probabilidade de risco futuro "
        "de queda de desempenho, queda de engajamento, aumento da defasagem ou evasão."
    )

    # =====================
    # ENTRADAS DO USUÁRIO
    # =====================

    st.markdown("### Dados gerais")

    fase = st.number_input("Fase", min_value=0, max_value=8, value=3)
    idade = st.number_input("Idade", min_value=6, max_value=30, value=12)
    defasagem = st.number_input("Defasagem", min_value=-10, max_value=5, value=0)

    st.markdown("### Indicadores educacionais")

    ida = st.slider("IDA - Desempenho Acadêmico", 0.0, 10.0, 6.0)
    ieg = st.slider("IEG - Engajamento", 0.0, 10.0, 7.0)
    iaa = st.slider("IAA - Autoavaliação", 0.0, 10.0, 7.0)
    ips = st.slider("IPS - Psicossocial", 0.0, 10.0, 7.0)
    ipp = st.slider("IPP - Psicopedagógico", 0.0, 10.0, 7.0)
    ipv = st.slider("IPV - Ponto de Virada", 0.0, 10.0, 7.0)
    ian = st.slider("IAN - Adequação de Nível", 0.0, 10.0, 7.0)

    st.markdown("### Desempenho por disciplina")

    matematica = st.slider("Matemática", 0.0, 10.0, 6.0)
    portugues = st.slider("Português", 0.0, 10.0, 6.0)
    ingles = st.slider("Inglês", 0.0, 10.0, 6.0)

    features = [
        "fase",
        "idade",
        "defasagem",
        "ida",
        "ieg",
        "iaa",
        "ips",
        "ipp",
        "ipv",
        "ian",
        "matematica",
        "portugues",
        "ingles"
    ]

    input_data = pd.DataFrame({
        "fase": [fase],
        "idade": [idade],
        "defasagem": [defasagem],
        "ida": [ida],
        "ieg": [ieg],
        "iaa": [iaa],
        "ips": [ips],
        "ipp": [ipp],
        "ipv": [ipv],
        "ian": [ian],
        "matematica": [matematica],
        "portugues": [portugues],
        "ingles": [ingles]
    })

    # =====================
    # BOTÃO DE PREVISÃO
    # =====================

    if st.button("Realizar Previsão"):
        prediction = model.predict(input_data[features])[0]
        proba = model.predict_proba(input_data[features])[0][1]

        st.markdown("---")
        st.metric("Probabilidade estimada de risco futuro", f"{proba:.2%}")

        if prediction == 1:
            st.error("Aluno classificado com RISCO FUTURO.")
        else:
            st.success("Aluno classificado como SEM RISCO FUTURO.")

        st.markdown("""
        **Interpretação:**  
        O modelo estima a probabilidade de o aluno apresentar risco futuro considerando queda de desempenho,
        queda de engajamento, piora da defasagem ou ausência no ano seguinte.
        """)


# ======================
# ABA 2 - INSIGHTS ANALÍTICOS
# ======================

with tab2:
    st.title("Painel Analítico - Risco de Defasagem")

    df = load_model_data()
    df_gold = load_gold_data()

    target = "risco_defasagem_futuro"

    # =========================
    # FILTROS INTERATIVOS
    # =========================

    st.sidebar.header("Filtros do Painel")

    anos = sorted(df["ano_ref"].dropna().unique())
    ano_selecionado = st.sidebar.selectbox(
        "Ano de referência",
        options=["Todos"] + anos
    )

    fases = sorted(df["fase"].dropna().unique())
    fase_selecionada = st.sidebar.selectbox(
        "Fase",
        options=["Todas"] + fases
    )

    risco_selecionado = st.sidebar.selectbox(
        "Grupo de risco",
        options=["Todos", "Sem risco futuro", "Risco futuro"]
    )

    # Aplicação dos filtros

    df_filtrado = df.copy()

    if ano_selecionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado["ano_ref"] == ano_selecionado]

    if fase_selecionada != "Todas":
        df_filtrado = df_filtrado[df_filtrado["fase"] == fase_selecionada]

    if risco_selecionado == "Sem risco futuro":
        df_filtrado = df_filtrado[df_filtrado[target] == 0]
    elif risco_selecionado == "Risco futuro":
        df_filtrado = df_filtrado[df_filtrado[target] == 1]

    # =========================
    # BLOCO 1 - KPIs
    # =========================

    st.markdown("## Indicadores Estratégicos")

    total = len(df_filtrado)

    if total > 0:
        perc_risco = round(df_filtrado[target].mean() * 100, 1)
        ida_medio = round(df_filtrado["ida"].mean(), 2)
        ieg_medio = round(df_filtrado["ieg"].mean(), 2)
        defasagem_media = round(df_filtrado["defasagem"].mean(), 2)
    else:
        perc_risco = 0
        ida_medio = 0
        ieg_medio = 0
        defasagem_media = 0

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total de Alunos", total)
    k2.metric("% em Risco Futuro", f"{perc_risco}%")
    k3.metric("Desemp Esc Médio - IDA", ida_medio)
    k4.metric("Engajamento Médio - IEG", ieg_medio)

    st.markdown("---")

      # =========================
    # BLOCO 2 - DISTRIBUIÇÃO DO RISCO
    # =========================

    st.markdown("## Distribuição do Risco Futuro")

    col1, col2 = st.columns([2, 1])

    with col1:
        if not df_filtrado.empty:
            dist = df_filtrado[target].value_counts().reset_index()
            dist.columns = ["Risco", "Quantidade"]
            dist["Risco"] = dist["Risco"].map({
                0: "Sem risco futuro",
                1: "Risco futuro"
            })

            fig = px.bar(
                dist,
                x="Risco",
                y="Quantidade",
                text="Quantidade",
                title="Distribuição dos Grupos de Risco"
            )

            fig.update_traces(textposition="outside")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Nenhum dado disponível para os filtros selecionados.")

    with col2:
        st.markdown(f"""
        **Insight Executivo:**
        - {perc_risco}% dos registros filtrados estão classificados como risco futuro.
        - O modelo considera risco como queda futura no Desempenho Escolar (IDA), queda futura no Engajamento (IEG),
          piora da defasagem ou evasão.
        """)

    st.markdown("---")

    st.markdown("## Evolução do IDA Médio por Ano")

    ida_ano = (
        df_gold
        .groupby("ano_ref")["ida"]
        .mean()
        .reset_index()
    )

    fig_ida = px.line(
        ida_ano,
        x="ano_ref",
        y="ida",
        markers=True,
        text=ida_ano["ida"].round(2),
        title="Evolução do IDA Médio por Ano"
    )

    fig_ida.update_traces(textposition="top center")

    fig_ida.update_layout(
        xaxis_title="Ano",
        yaxis_title="IDA Médio"
    )

    st.plotly_chart(fig_ida, use_container_width=True)

    st.markdown("""
    **Leitura:**  
    O gráfico acompanha a evolução média do desempenho acadêmico ao longo dos anos,
    permitindo observar oscilações no IDA antes da etapa de modelagem preditiva.
    """)

    st.markdown("---")
    st.markdown("## Correlação dos Indicadores com o INDE")

    cols_corr = ["inde", "ida", "ieg", "ipv", "ipp", "ian", "iaa", "ips"]

    corr_inde = (
        df_gold[cols_corr]
        .corr()[["inde"]]
        .sort_values("inde", ascending=False)
        .reset_index()
    )

    corr_inde.columns = ["Indicador", "Correlação com INDE"]

    fig_corr = px.imshow(
        corr_inde[["Correlação com INDE"]].T,
        x=corr_inde["Indicador"],
        y=["Correlação"],
        text_auto=".2f",
        color_continuous_scale="Blues",
        aspect="auto",
        title="Correlação dos Indicadores com o INDE"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    **Leitura:**  
    IDA, IEG e IPV aparecem como os indicadores mais associados ao INDE,
    reforçando a importância de desempenho, engajamento e ponto de virada
    na composição da nota global do aluno.
    """)

    st.markdown("---")
    st.markdown("## INDE Médio por Combinação de IDA e IEG")

    df_heat = df_gold.dropna(subset=["ida", "ieg", "inde"]).copy()

    df_heat["faixa_ida"] = pd.qcut(
        df_heat["ida"],
        q=3,
        labels=["Baixo IDA", "Médio IDA", "Alto IDA"],
        duplicates="drop"
    )

    df_heat["faixa_ieg"] = pd.qcut(
        df_heat["ieg"],
        q=3,
        labels=["Baixo IEG", "Médio IEG", "Alto IEG"],
        duplicates="drop"
    )

    heat_inde = df_heat.pivot_table(
        index="faixa_ida",
        columns="faixa_ieg",
        values="inde",
        aggfunc="mean"
    )

    fig_heat = px.imshow(
        heat_inde,
        text_auto=".2f",
        color_continuous_scale="Blues",
        aspect="auto",
        title="INDE Médio por Combinação de IDA e IEG"
    )

    fig_heat.update_layout(
        xaxis_title="Faixa de IEG",
        yaxis_title="Faixa de IDA"
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    **Leitura:**  
    O maior INDE médio ocorre quando o aluno combina alto desempenho acadêmico
    com alto engajamento. Esse resultado reforça que IDA e IEG são dimensões centrais
    para explicar melhores níveis de desenvolvimento educacional.
    """)

    st.markdown("---")


      # =========================
    # BLOCO 3 - MÉDIA DOS INDICADORES POR GRUPO
    # =========================

    st.markdown("## Indicadores Médios por Grupo de Risco")

    indicadores = [
        "ida",
        "ieg",
        "iaa",
        "ips",
        "ipp",
        "ipv",
        "ian",
        "matematica",
        "portugues",
        "ingles"
    ]

    if not df_filtrado.empty:
        medias = (
            df_filtrado
            .groupby(target)[indicadores]
            .mean()
            .T
            .reset_index()
        )

        medias.columns.name = None
        medias = medias.rename(columns={"index": "Indicador"})

        medias_long = medias.melt(
            id_vars="Indicador",
            var_name="Grupo",
            value_name="Média"
        )

        medias_long["Grupo"] = medias_long["Grupo"].map({
            0: "Sem risco futuro",
            1: "Risco futuro"
        })

        fig2 = px.bar(
            medias_long,
            x="Indicador",
            y="Média",
            color="Grupo",
            barmode="group",
            title="Indicadores Médios por Grupo"
        )

        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Sem dados para calcular médias dos indicadores.")

    st.markdown("---")

    # =========================
    # BLOCO 4 - IDA E IEG POR GRUPO
    # =========================

    st.markdown("## Desempenho e Engajamento por Grupo")

    col3, col4 = st.columns(2)

    with col3:
        if not df_filtrado.empty:
            fig3 = px.box(
                df_filtrado,
                x=target,
                y="ida",
                points=False,
                title="Distribuição do IDA por Grupo"
            )

            fig3.update_xaxes(
                tickvals=[0, 1],
                ticktext=["Sem risco", "Risco"]
            )

            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        if not df_filtrado.empty:
            fig4 = px.box(
                df_filtrado,
                x=target,
                y="ieg",
                points=False,
                title="Distribuição do IEG por Grupo"
            )

            fig4.update_xaxes(
                tickvals=[0, 1],
                ticktext=["Sem risco", "Risco"]
            )

            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # =========================
    # BLOCO 5 - DEFASAGEM
    # =========================

    st.markdown("## Defasagem por Grupo de Risco")

    if not df_filtrado.empty:
        fig5 = px.box(
            df_filtrado,
            x=target,
            y="defasagem",
            points="outliers",
            title="Distribuição da Defasagem por Grupo"
        )

        fig5.update_xaxes(
            tickvals=[0, 1],
            ticktext=["Sem risco", "Risco"]
        )

        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    **Interpretação:**  
    O painel permite observar diferenças médias entre os alunos classificados com e sem risco futuro.
    A modelagem considera indicadores acadêmicos, psicossociais, psicopedagógicos e de engajamento.
    """)