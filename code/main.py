
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import plotly.express as px

st.set_page_config(
    page_title="운동 시간은 많을수록 정말 좋은가?",
    page_icon="🏃",
    layout="wide"
)

st.title("🏃 운동 시간은 많을수록 정말 좋은가?")
st.caption("CDC NHANES 2017~2018 청소년 건강 데이터 분석")

st.markdown("""
이 프로젝트는 청소년의 신체활동 빈도와 건강 지표 사이의 관계를 분석합니다.

미국 CDC의 공개 조사 데이터를 활용하며,
운동량에 따라 BMI, 혈압, 맥박이 어떻게 달라지는지 비교합니다.
""")

# --------------------------------------------------
# 데이터 주소
# --------------------------------------------------

URLS = {
    "운동": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PAQY_J.xpt",
    "신체계측": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.xpt",
    "혈압": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.xpt",
    "인구정보": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt"
}

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data(ttl=86400)
def download_data(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_sas(
        BytesIO(response.content),
        format="xport"
    )

@st.cache_data(ttl=86400)
def load_all_data():

    paq = download_data(URLS["운동"])
    bmx = download_data(URLS["신체계측"])
    bpx = download_data(URLS["혈압"])
    demo = download_data(URLS["인구정보"])

    paq = paq[["SEQN", "PAQ706"]]
    bmx = bmx[["SEQN", "BMXBMI"]]

    bpx_columns = [
        "SEQN",
        "BPXSY1",
        "BPXDI1",
        "BPXPLS"
    ]

    bpx = bpx[bpx_columns]

    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]

    # 개인 식별번호로 데이터 결합
    df = paq.merge(bmx, on="SEQN", how="left")
    df = df.merge(bpx, on="SEQN", how="left")
    df = df.merge(demo, on="SEQN", how="left")

    # 숫자형 변환
    columns = [
        "PAQ706", "BMXBMI",
        "BPXSY1", "BPXDI1",
        "BPXPLS", "RIDAGEYR"
    ]

    for col in columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # 운동 일수의 유효 응답만 사용
    df = df[df["PAQ706"].between(0, 7)].copy()

    # 운동 그룹
    bins = [-1, 1, 3, 5, 7]
    labels = [
        "A그룹 (0~1일)",
        "B그룹 (2~3일)",
        "C그룹 (4~5일)",
        "D그룹 (6~7일)"
    ]

    df["운동 그룹"] = pd.cut(
        df["PAQ706"],
        bins=bins,
        labels=labels
    )

    return df


# --------------------------------------------------
# 데이터 분석
# --------------------------------------------------

st.header("1. 데이터 불러오기")

try:
    with st.spinner("CDC 공식 데이터를 내려받고 있습니다..."):
        df = load_all_data()

    st.success("CDC 데이터 불러오기 완료!")

except Exception as e:
    st.error(
        "데이터를 불러오지 못했습니다. "
        "인터넷 연결과 CDC 서버 상태를 확인해 주세요."
    )
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# 데이터 기본 정보
# --------------------------------------------------

st.header("2. 데이터 기본 정보")

c1, c2, c3 = st.columns(3)

c1.metric(
    "운동 조사 유효 응답자",
    f"{len(df):,}명"
)

c2.metric(
    "평균 운동 일수",
    f"{df['PAQ706'].mean():.2f}일"
)

c3.metric(
    "분석 연령",
    f"{int(df['RIDAGEYR'].min())}~"
    f"{int(df['RIDAGEYR'].max())}세"
)

st.info(
    "운동 일수는 지난 7일 동안 하루 총 60분 이상 "
    "심박수를 높이고 숨이 차는 신체활동을 한 날의 수입니다."
)

# --------------------------------------------------
# 운동량별 인원수
# --------------------------------------------------

st.header("3. 운동량별 인원수")

group_order = [
    "A그룹 (0~1일)",
    "B그룹 (2~3일)",
    "C그룹 (4~5일)",
    "D그룹 (6~7일)"
]

count_table = (
    df.groupby("운동 그룹", observed=False)
    .size()
    .reindex(group_order, fill_value=0)
    .reset_index(name="인원수")
)

count_table["비율(%)"] = (
    count_table["인원수"] / len(df) * 100
).round(2)

st.dataframe(
    count_table,
    use_container_width=True,
    hide_index=True
)

fig_count = px.bar(
    count_table,
    x="운동 그룹",
    y="인원수",
    text="인원수",
    title="운동량 그룹별 응답자 수"
)

fig_count.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig_count,
    use_container_width=True
)


# --------------------------------------------------
# 건강 지표별 평균
# --------------------------------------------------

st.header("4. 운동량과 건강 지표 비교")

health_columns = {
    "BMI": "BMXBMI",
    "수축기 혈압": "BPXSY1",
    "이완기 혈압": "BPXDI1",
    "맥박수": "BPXPLS"
}

unit = {
    "BMI": "",
    "수축기 혈압": "mmHg",
    "이완기 혈압": "mmHg",
    "맥박수": "회/분"
}

result = []

for group in group_order:

    part = df[df["운동 그룹"] == group]

    row = {
        "운동 그룹": group,
        "전체 인원": len(part)
    }

    for name, col in health_columns.items():

        valid = part[col].dropna()

        row[f"{name} 평균"] = (
            round(valid.mean(), 2)
            if len(valid) > 0 else np.nan
        )

        row[f"{name} 분석 인원"] = len(valid)

    result.append(row)

result_df = pd.DataFrame(result)

st.subheader("운동량별 건강 지표 평균")

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "각 건강 지표의 분석 인원은 결측값을 제외한 인원입니다. "
    "지표마다 분석 인원이 다를 수 있습니다."
)

# --------------------------------------------------
# 개별 건강 지표 그래프
# --------------------------------------------------

st.header("5. 건강 지표별 그래프")

selected = st.selectbox(
    "비교할 건강 지표 선택",
    list(health_columns.keys())
)

selected_col = health_columns[selected]

chart_data = (
    df.groupby("운동 그룹", observed=False)[selected_col]
    .agg(["mean", "count"])
    .reindex(group_order)
    .reset_index()
)

chart_data = chart_data.dropna(subset=["mean"])

fig = px.bar(
    chart_data,
    x="운동 그룹",
    y="mean",
    text="mean",
    title=f"운동량에 따른 {selected} 평균 비교",
    labels={
        "mean": f"평균 {selected} {unit[selected]}",
        "운동 그룹": "운동량"
    }
)

fig.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# 연령과 성별 확인
# --------------------------------------------------

st.header("6. 연령 및 성별에 따른 데이터 확인")

age_table = (
    df.groupby("운동 그룹", observed=False)
    ["RIDAGEYR"]
    .agg(["mean", "count"])
    .reindex(group_order)
    .reset_index()
)

age_table.columns = [
    "운동 그룹", "평균 연령", "인원수"
]

st.dataframe(
    age_table.round(2),
    use_container_width=True,
    hide_index=True
)

gender_names = {
    1: "남성",
    2: "여성"
}

gender_df = df.copy()

gender_df["성별"] = (
    gender_df["RIAGENDR"]
    .map(gender_names)
)

gender_table = (
    gender_df.groupby(
        ["운동 그룹", "성별"],
        observed=False
    )
    .size()
    .reset_index(name="인원수")
)

st.dataframe(
    gender_table,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 분석 결과 자동 정리
# --------------------------------------------------

st.header("7. 분석 결과")

st.markdown("""
아래 내용은 실제 계산된 평균을 바탕으로 확인할 수 있습니다.
단, 평균 차이만으로 운동이 건강 변화의 원인이라고
판단할 수는 없습니다.
""")

for name in health_columns:

    avg_col = f"{name} 평균"

    valid = result_df.dropna(subset=[avg_col])

    if len(valid) >= 2:

        best = valid.loc[
            valid[avg_col].idxmin()
        ]

        worst = valid.loc[
            valid[avg_col].idxmax()
        ]

        st.write(
            f"**{name}**: 평균이 가장 낮은 그룹은 "
            f"{best['운동 그룹']}이며 "
            f"{best[avg_col]:.2f}, "
            f"가장 높은 그룹은 "
            f"{worst['운동 그룹']}이며 "
            f"{worst[avg_col]:.2f}입니다."
        )

# --------------------------------------------------
# 데이터 다운로드
# --------------------------------------------------

st.header("8. 분석 데이터 다운로드")

csv = result_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="📥 건강 지표 비교표 CSV 다운로드",
    data=csv,
    file_name="exercise_health_analysis.csv",
    mime="text/csv"
)

raw_csv = df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="📥 결합된 원자료 CSV 다운로드",
    data=raw_csv,
    file_name="exercise_health_raw_data.csv",
    mime="text/csv"
)

# --------------------------------------------------
# 출처
# --------------------------------------------------

st.header("9. 데이터 출처")

st.markdown("""
- [CDC NHANES 2017~2018 청소년 신체활동](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PAQY_J.htm)
- [CDC 신체계측 자료](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.htm)
- [CDC 혈압 자료](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.htm)
- [CDC 인구통계 자료](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm)

**조사 기간:** 2017~2018년

**조사 기관:** 미국 질병통제예방센터(CDC)

**분석 방법:** 개인 식별번호(SEQN)를 이용한 데이터 결합 및 그룹별 평균 비교
""")

st.caption(
    "주의: 이 분석은 관찰 자료의 집단별 평균 비교이며 "
    "인과관계를 증명하지 않습니다. 청소년 BMI는 "
    "연령과 성별을 고려한 백분위 해석이 필요합니다."
)
