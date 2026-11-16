
import streamlit as st
import pandas as pd
import numpy as np
import requests
import statsmodels.formula.api as smf
import plotly.express as px
from io import BytesIO

# ==========================================
# 페이지 설정
# ==========================================

st.set_page_config(
    page_title="운동 시간은 많을수록 정말 좋은가?",
    page_icon="🏃",
    layout="wide"
)

st.title("🏃 운동 시간은 많을수록 정말 좋은가?")
st.subheader("운동량과 건강 지표의 관계 분석")
st.caption("CDC NHANES 2017~2018 공개 데이터 활용")

st.markdown("""
이 프로젝트는 운동량과 건강 지표 사이에 어떤 관계가 있는지
회귀 분석을 통해 알아보는 프로젝트입니다.

운동하는 날의 수와 BMI, 혈압, 맥박수 사이의 관계를 분석합니다.
""")

st.divider()

# ==========================================
# 데이터 주소
# ==========================================

DATA_URLS = {
    "운동": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PAQY_J.xpt",
    "신체계측": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.xpt",
    "혈압": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.xpt",
    "인구정보": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt"
}

# ==========================================
# 데이터 다운로드
# ==========================================

@st.cache_data(ttl=86400)
def download_xpt(url):

    response = requests.get(
        url,
        timeout=90
    )

    response.raise_for_status()

    return pd.read_sas(
        BytesIO(response.content),
        format="xport"
    )


@st.cache_data(ttl=86400)
def load_data():

    # 원본 데이터 다운로드
    paq = download_xpt(DATA_URLS["운동"])
    bmx = download_xpt(DATA_URLS["신체계측"])
    bpx = download_xpt(DATA_URLS["혈압"])
    demo = download_xpt(DATA_URLS["인구정보"])

    # 필요한 열만 선택
    paq = paq[[
        "SEQN",
        "PAQ706"
    ]]

    bmx = bmx[[
        "SEQN",
        "BMXBMI"
    ]]

    bpx_columns = ["SEQN"]

    for col in [
        "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
        "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4",
        "BPXPLS"
    ]:
        if col in bpx.columns:
            bpx_columns.append(col)

    bpx = bpx[bpx_columns]

    demo = demo[[
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR"
    ]]

    # 개인 식별번호를 기준으로 결합
    df = paq.merge(
        bmx,
        on="SEQN",
        how="left"
    )

    df = df.merge(
        bpx,
        on="SEQN",
        how="left"
    )

    df = df.merge(
        demo,
        on="SEQN",
        how="left"
    )

    # 숫자형 변환
    numeric_columns = [
        col for col in df.columns
        if col != "SEQN"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # 운동 일수 유효 응답만 사용
    df = df[
        df["PAQ706"].between(0, 7)
    ].copy()

    # 혈압 여러 번 측정한 값의 개인별 평균
    systolic = [
        c for c in [
            "BPXSY1", "BPXSY2",
            "BPXSY3", "BPXSY4"
        ] if c in df.columns
    ]

    diastolic = [
        c for c in [
            "BPXDI1", "BPXDI2",
            "BPXDI3", "BPXDI4"
        ] if c in df.columns
    ]

    df["수축기혈압"] = df[systolic].mean(axis=1)
    df["이완기혈압"] = df[diastolic].mean(axis=1)

    # 분석용 변수 이름
    df = df.rename(columns={
        "PAQ706": "운동일수",
        "BMXBMI": "BMI",
        "BPXPLS": "맥박수",
        "RIDAGEYR": "나이",
        "RIAGENDR": "성별"
    })

    df["성별"] = df["성별"].map({
        1: "남성",
        2: "여성"
    })

    return df


# ==========================================
# 데이터 불러오기
# ==========================================

st.header("1. 데이터 수집")

try:

    with st.spinner(
        "CDC 공식 데이터를 다운로드하고 있습니다..."
    ):
        df = load_data()

    st.success(
        "CDC 원자료 다운로드 및 결합 완료!"
    )

except Exception as e:

    st.error(
        "데이터를 불러오지 못했습니다."
    )

    st.write(
        "인터넷 연결이나 CDC 서버 상태를 확인해 주세요."
    )

    st.exception(e)

    st.stop()


# ==========================================
# 데이터 소개
# ==========================================

st.header("2. 데이터 소개")

st.markdown("""
| 항목 | 내용 |
|---|---|
| 조사 기관 | 미국 질병통제예방센터(CDC) |
| 데이터 | NHANES 2017~2018 |
| 조사 대상 | 2~17세 청소년 신체활동 조사 참여자 |
| 독립변수 | 일주일 중 하루 60분 이상 활동한 날의 수 |
| 종속변수 | BMI, 수축기 혈압, 이완기 혈압, 맥박수 |
| 분석 방법 | 선형회귀 및 다중회귀 분석 |
""")

c1, c2, c3 = st.columns(3)

c1.metric(
    "운동 조사 응답자",
    f"{len(df):,}명"
)

c2.metric(
    "평균 운동 일수",
    f"{df['운동일수'].mean():.2f}일"
)

c3.metric(
    "평균 연령",
    f"{df['나이'].mean():.1f}세"
)

st.info(
    "PAQ706은 하루 총 60분 이상 신체활동을 한 날의 수입니다. "
    "정확한 운동 시간(분)을 측정한 변수는 아닙니다."
)


# ==========================================
# 운동량 분포
# ==========================================

st.header("3. 운동량 분포")

exercise_count = (
    df["운동일수"]
    .value_counts()
    .sort_index()
    .reindex(range(8), fill_value=0)
    .reset_index()
)

exercise_count.columns = [
    "운동한 날",
    "인원수"
]

exercise_count["비율(%)"] = (
    exercise_count["인원수"]
    / exercise_count["인원수"].sum()
    * 100
).round(2)

st.dataframe(
    exercise_count,
    use_container_width=True,
    hide_index=True
)

fig_count = px.bar(
    exercise_count,
    x="운동한 날",
    y="인원수",
    text="인원수",
    title="일주일 중 하루 60분 이상 활동한 날의 수"
)

fig_count.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_count,
    use_container_width=True
)


# ==========================================
# 회귀 분석 설명
# ==========================================

st.header("4. 회귀 분석 방법")

st.markdown("""
회귀 분석은 두 변수 사이의 관계를 직선으로 나타내는 분석 방법입니다.

- 독립변수(X): 일주일 중 하루 60분 이상 활동한 날의 수
- 종속변수(Y): BMI, 혈압, 맥박수

운동 일수가 증가할 때 건강 지표가 어떤 방향으로 변화하는지 확인합니다.
""")

st.latex(
    r"Y = a + bX"
)

st.markdown("""
- X: 운동 일수
- Y: 건강 지표
- a: 절편
- b: 회귀계수(기울기)

회귀계수가 양수이면 운동 일수가 많을수록 해당 지표가 높아지는 경향,
음수이면 낮아지는 경향을 뜻합니다.
""")


# ==========================================
# 건강 지표 설정
# ==========================================

health_variables = {
    "BMI": {
        "column": "BMI",
        "unit": "",
        "description": "체질량지수"
    },
    "수축기 혈압": {
        "column": "수축기혈압",
        "unit": "mmHg",
        "description": "수축기 혈압 평균"
    },
    "이완기 혈압": {
        "column": "이완기혈압",
        "unit": "mmHg",
        "description": "이완기 혈압 평균"
    },
    "맥박수": {
        "column": "맥박수",
        "unit": "회/분",
        "description": "분당 맥박수"
    }
}


# ==========================================
# 개별 회귀 분석
# ==========================================

st.header("5. 운동량과 건강 지표의 관계")

selected_name = st.selectbox(
    "분석할 건강 지표를 선택하세요.",
    list(health_variables.keys())
)

selected_info = health_variables[selected_name]
selected_col = selected_info["column"]

analysis_df = df[[
    "운동일수",
    selected_col,
    "나이",
    "성별"
]].dropna().copy()

if len(analysis_df) < 10:

    st.warning(
        "회귀 분석을 수행할 데이터가 부족합니다."
    )

else:

    # 산점도와 회귀선
    fig_reg = px.scatter(
        analysis_df,
        x="운동일수",
        y=selected_col,
        trendline="ols",
        opacity=0.55,
        title=f"운동 일수와 {selected_name}의 관계",
        labels={
            "운동일수": "일주일 중 하루 60분 이상 활동한 날",
            selected_col: selected_name
        },
        hover_data=["나이", "성별"]
    )

    st.plotly_chart(
        fig_reg,
        use_container_width=True
    )

    # 단순 회귀
    simple_model = smf.ols(
        f'Q("{selected_col}") ~ 운동일수',
        data=analysis_df
    ).fit()

    slope = simple_model.params["운동일수"]
    r_squared = simple_model.rsquared
    p_value = simple_model.pvalues["운동일수"]

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "회귀계수",
        f"{slope:.4f}"
    )

    m2.metric(
        "결정계수 R²",
        f"{r_squared:.4f}"
    )

    m3.metric(
        "p-value",
        f"{p_value:.4f}"
    )

    st.markdown("**회귀 분석 결과**")

    if slope > 0:
        st.write(
            f"운동 일수가 1일 증가할 때 "
            f"{selected_name}은 평균적으로 "
            f"{abs(slope):.3f}만큼 높아지는 선형 관계가 관찰됩니다."
        )

    elif slope < 0:
        st.write(
            f"운동 일수가 1일 증가할 때 "
            f"{selected_name}은 평균적으로 "
            f"{abs(slope):.3f}만큼 낮아지는 선형 관계가 관찰됩니다."
        )

    else:
        st.write(
            "운동 일수와 해당 건강 지표 사이의 "
            "선형 기울기는 거의 0입니다."
        )

    st.caption(
        "R²는 단순 회귀모형이 해당 건강 지표의 변동을 "
        "얼마나 설명하는지 나타냅니다. p-value는 통계적 "
        "유의성을 판단하는 참고값이며 인과관계를 증명하지 않습니다."
    )


# ==========================================
# 모든 건강 지표 회귀 분석
# ==========================================

st.header("6. 전체 건강 지표 회귀 분석")

all_results = []

for name, info in health_variables.items():

    col = info["column"]

    temp = df[[
        "운동일수",
        col,
        "나이",
        "성별"
    ]].dropna().copy()

    if len(temp) < 10:
        continue

    model = smf.ols(
        f'Q("{col}") ~ 운동일수',
        data=temp
    ).fit()

    adjusted_model = smf.ols(
        f'Q("{col}") ~ 운동일수 + 나이 + C(성별)',
        data=temp
    ).fit()

    all_results.append({
        "건강 지표": name,
        "분석 인원": len(temp),
        "회귀계수": model.params["운동일수"],
        "R²": model.rsquared,
        "p-value": model.pvalues["운동일수"],
        "나이·성별 보정 회귀계수":
            adjusted_model.params["운동일수"],
        "보정 R²": adjusted_model.rsquared
    })

results_df = pd.DataFrame(all_results)

st.dataframe(
    results_df.round(4),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "나이·성별 보정 회귀계수는 나이와 성별을 통제한 상태에서 "
    "운동 일수와 건강 지표의 관계를 추정한 값입니다."
)


# ==========================================
# 운동량별 평균 비교
# ==========================================

st.header("7. 운동량별 건강 지표 평균")

group_bins = [-1, 1, 3, 5, 7]

group_labels = [
    "0~1일",
    "2~3일",
    "4~5일",
    "6~7일"
]

df["운동 그룹"] = pd.cut(
    df["운동일수"],
    bins=group_bins,
    labels=group_labels
)

group_results = []

for group in group_labels:

    part = df[df["운동 그룹"] == group]

    row = {
        "운동 그룹": group,
        "운동 조사 인원": len(part)
    }

    for name, info in health_variables.items():

        col = info["column"]

        valid = part[col].dropna()

        row[name] = (
            round(valid.mean(), 2)
            if len(valid) > 0 else np.nan
        )

        row[f"{name} 분석 인원"] = len(valid)

    group_results.append(row)

group_df = pd.DataFrame(group_results)

st.dataframe(
    group_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "그룹별 평균은 참고용 비교 결과입니다. "
    "회귀 분석과는 다른 분석 방식입니다."
)


# ==========================================
# 연령과 성별
# ==========================================

st.header("8. 연령과 성별 확인")

age_gender = (
    df.groupby(
        ["나이", "성별"],
        observed=True
    )
    .size()
    .reset_index(name="인원수")
)

st.dataframe(
    age_gender,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 데이터 다운로드
# ==========================================

st.header("9. 분석 결과 다운로드")

result_csv = results_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "📥 회귀 분석 결과 다운로드",
    data=result_csv,
    file_name="regression_results.csv",
    mime="text/csv"
)

group_csv = group_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "📥 운동량별 건강 비교표 다운로드",
    data=group_csv,
    file_name="exercise_group_results.csv",
    mime="text/csv"
)

raw_csv = df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "📥 결합 원자료 다운로드",
    data=raw_csv,
    file_name="nhanes_combined_data.csv",
    mime="text/csv"
)


# ==========================================
# 결론 안내
# ==========================================

st.header("10. 분석 시 주의할 점")

st.markdown("""
1. 이 자료는 운동 시간이 아니라 하루 60분 이상 활동한 날의 수를 측정합니다.
2. BMI는 청소년의 경우 나이와 성별에 따른 백분위 해석이 필요합니다.
3. 혈압과 맥박은 실제 측정값이 있는 사람만 분석합니다.
4. 분석 변수마다 결측값이 다르므로 분석 인원이 다를 수 있습니다.
5. 나이와 성별을 고려하더라도 다른 생활 습관 등 모든 요인이 통제되는 것은 아닙니다.
6. 회귀 분석 결과는 관계를 보여줄 뿐 운동이 건강 변화의 직접적인 원인임을 증명하지 않습니다.
7. 공식 NHANES 분석에서는 조사 설계와 표본 가중치도 고려해야 합니다.
""")

st.header("11. 데이터 출처")

st.markdown("""
- [CDC NHANES 2017~2018 청소년 신체활동](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PAQY_J.htm)
- [CDC 신체계측 자료](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.htm)
- [CDC 혈압 자료](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.htm)
- [CDC 인구통계 자료](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm)
- [CDC NHANES 분석 안내](https://wwwn.cdc.gov/nchs/nhanes/tutorials/)
""")

st.divider()

st.caption(
    "운동 시간은 많을수록 정말 좋은가? | "
    "CDC NHANES 2017~2018 데이터 분석 프로젝트"
)
