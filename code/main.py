
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats

st.set_page_config(
    page_title="운동 시간은 많을수록 정말 좋은가?",
    page_icon="🏃",
    layout="wide"
)

st.title("🏃 운동 시간은 많을수록 정말 좋은가?")
st.subheader("운동량에 따른 청소년의 제지방량 비교")

st.write(
    "미국 CDC의 NHANES 2017~2018년 자료를 활용하여 "
    "청소년의 신체활동 일수와 제지방량 사이의 관계를 분석합니다."
)

st.info(
    "분석 대상: 13~17세 청소년 | "
    "운동량: 일주일 중 하루 60분 이상 신체활동을 한 날의 수"
)

# --------------------------------------------------
# 1. 데이터 다운로드 및 전처리
# --------------------------------------------------

@st.cache_data
def load_data():
    base_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"

    # 신체활동 자료
    activity = pd.read_sas(
        base_url + "PAQY_J.xpt",
        format="xport"
    )

    # DXA 체성분 자료
    body = pd.read_sas(
        base_url + "DXX_J.xpt",
        format="xport"
    )

    # 나이와 성별 자료
    demo = pd.read_sas(
        base_url + "DEMO_J.xpt",
        format="xport"
    )

    activity = activity[["SEQN", "PAQ706"]]
    body = body[["SEQN", "DXDTOLE"]]
    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]

    # 개인 식별번호(SEQN)를 기준으로 자료 결합
    df = activity.merge(body, on="SEQN", how="inner")
    df = df.merge(demo, on="SEQN", how="inner")

    # 숫자형으로 변환
    for col in ["PAQ706", "DXDTOLE", "RIDAGEYR", "RIAGENDR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 13~17세 청소년만 선택
    df = df[
        (df["RIDAGEYR"] >= 13) &
        (df["RIDAGEYR"] <= 17)
    ].copy()

    # 유효한 운동 일수(0~7일)만 선택
    df = df[
        df["PAQ706"].between(0, 7) &
        df["DXDTOLE"].notna()
    ].copy()

    # 제지방량 g -> kg
    df["제지방량(kg)"] = df["DXDTOLE"] / 1000

    # 운동량 그룹
    def classify(days):
        if days <= 1:
            return "운동 적음 (0~1일)"
        elif days <= 5:
            return "운동 보통 (2~5일)"
        else:
            return "운동 많음 (6~7일)"

    df["운동량 그룹"] = df["PAQ706"].apply(classify)

    # 성별 이름
    df["성별"] = df["RIAGENDR"].map({
        1: "남성",
        2: "여성"
    })

    return df


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

with st.spinner("CDC 원자료를 내려받고 분석하고 있습니다..."):
    try:
        data = load_data()
    except Exception as e:
        st.error("데이터를 불러오는 중 오류가 발생했습니다.")
        st.code(str(e))
        st.info(
            "인터넷 연결을 확인한 뒤 다시 실행해 주세요. "
            "처음 실행할 때는 자료 다운로드에 시간이 걸릴 수 있습니다."
        )
        st.stop()

if data.empty:
    st.warning("조건에 맞는 데이터가 없습니다.")
    st.stop()

st.success(f"분석 가능한 청소년 데이터: {len(data):,}명")

# --------------------------------------------------
# 3. 전체 데이터 요약
# --------------------------------------------------

st.divider()
st.header("1. 운동량에 따른 제지방량 비교")

summary = (
    data.groupby("운동량 그룹", observed=True)
    .agg(
        인원수=("제지방량(kg)", "count"),
        평균_제지방량=("제지방량(kg)", "mean"),
        중앙값=("제지방량(kg)", "median"),
        표준편차=("제지방량(kg)", "std")
    )
    .reindex([
        "운동 적음 (0~1일)",
        "운동 보통 (2~5일)",
        "운동 많음 (6~7일)"
    ])
    .dropna(subset=["인원수"])
    .reset_index()
)

summary_display = summary.rename(columns={
    "운동량 그룹": "운동량",
    "인원수": "인원(명)",
    "평균_제지방량": "평균 제지방량(kg)",
    "중앙값": "중앙 제지방량(kg)",
    "표준편차": "표준편차(kg)"
})

for col in [
    "평균 제지방량(kg)",
    "중앙 제지방량(kg)",
    "표준편차(kg)"
]:
    summary_display[col] = summary_display[col].round(2)

st.dataframe(
    summary_display,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 4. 평균 비교 그래프
# --------------------------------------------------

st.subheader("운동량 그룹별 평균 제지방량")

fig = px.bar(
    summary,
    x="운동량 그룹",
    y="평균_제지방량",
    text=summary["평균_제지방량"].round(2),
    labels={
        "운동량 그룹": "운동량",
        "평균_제지방량": "평균 제지방량(kg)"
    },
    title="운동량 그룹별 평균 제지방량 비교"
)

fig.update_traces(
    textposition="outside"
)

fig.update_layout(
    yaxis_title="평균 제지방량(kg)",
    xaxis_title="운동량 그룹",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 5. 운동 일수별 제지방량
# --------------------------------------------------

st.divider()
st.header("2. 운동 일수와 제지방량의 관계")

daily = (
    data.groupby("PAQ706")
    .agg(
        인원수=("제지방량(kg)", "count"),
        평균_제지방량=("제지방량(kg)", "mean")
    )
    .reset_index()
)

daily = daily.rename(columns={
    "PAQ706": "운동 일수",
    "평균_제지방량": "평균 제지방량(kg)"
})

fig2 = px.scatter(
    data,
    x="PAQ706",
    y="제지방량(kg)",
    opacity=0.35,
    labels={
        "PAQ706": "일주일 중 60분 이상 활동한 날",
        "제지방량(kg)": "제지방량(kg)"
    },
    title="개인별 운동 일수와 제지방량"
)

fig2.add_scatter(
    x=daily["운동 일수"],
    y=daily["평균 제지방량(kg)"],
    mode="lines+markers",
    name="일수별 평균"
)

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# 6. 상관관계 분석
# --------------------------------------------------

st.divider()
st.header("3. 운동 일수와 제지방량의 상관관계")

correlation, p_value = stats.pearsonr(
    data["PAQ706"],
    data["제지방량(kg)"]
)

c1, c2 = st.columns(2)

c1.metric(
    "상관계수",
    f"{correlation:.3f}"
)

c2.metric(
    "분석 대상 인원",
    f"{len(data):,}명"
)

if correlation > 0:
    st.write("운동 일수가 많을수록 제지방량이 증가하는 양의 상관관계가 관찰됩니다.")
elif correlation < 0:
    st.write("운동 일수가 많을수록 제지방량이 감소하는 음의 상관관계가 관찰됩니다.")
else:
    st.write("운동 일수와 제지방량 사이에 선형 상관관계가 거의 관찰되지 않습니다.")

st.caption(
    f"통계적 검정의 p-value: {p_value:.4f}"
)

# --------------------------------------------------
# 7. 성별에 따른 비교
# --------------------------------------------------

st.divider()
st.header("4. 성별에 따른 운동량과 제지방량")

gender_summary = (
    data.groupby(["성별", "운동량 그룹"], observed=True)
    ["제지방량(kg)"]
    .agg(["count", "mean"])
    .reset_index()
)

gender_summary = gender_summary.rename(columns={
    "count": "인원(명)",
    "mean": "평균 제지방량(kg)"
})

gender_summary["평균 제지방량(kg)"] = (
    gender_summary["평균 제지방량(kg)"].round(2)
)

st.dataframe(
    gender_summary,
    use_container_width=True,
    hide_index=True
)

fig3 = px.bar(
    gender_summary,
    x="운동량 그룹",
    y="평균 제지방량(kg)",
    color="성별",
    barmode="group",
    text="평균 제지방량(kg)",
    title="성별 및 운동량 그룹별 평균 제지방량"
)

st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------
# 8. 원자료 확인
# --------------------------------------------------

st.divider()
st.header("5. 분석에 사용한 데이터")

st.write("분석에 사용한 일부 원자료입니다.")

st.dataframe(
    data[
        [
            "RIDAGEYR",
            "성별",
            "PAQ706",
            "운동량 그룹",
            "제지방량(kg)"
        ]
    ].rename(columns={
        "RIDAGEYR": "나이",
        "PAQ706": "일주일 운동 일수"
    }).head(30),
    use_container_width=True,
    hide_index=True
)

csv = data.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="분석 데이터 CSV 다운로드",
    data=csv,
    file_name="운동량_제지방량_분석.csv",
    mime="text/csv"
)

# --------------------------------------------------
# 9. 해석 및 주의사항
# --------------------------------------------------

st.divider()
st.header("6. 분석할 때 주의할 점")

st.warning(
    """
    이 자료의 제지방량은 순수 근육량과 같은 개념이 아닙니다.

    DXA로 측정한 전체 제지방량에는 근육뿐 아니라 수분과 장기 등
    지방이 아닌 다른 조직도 포함됩니다.

    또한 나이, 성별, 키, 성장 단계 등에 따라 제지방량이 달라질 수 있습니다.
    따라서 운동량 그룹의 평균 차이만으로 운동이 근육량을 증가시켰다고
    단정할 수 없습니다.
    """
)

st.caption(
    "자료 출처: 미국 질병통제예방센터(CDC), "
    "NHANES 2017-2018"
)

st.markdown(
    "[신체활동 자료 설명](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PAQY_J.htm)"
)

st.markdown(
    "[체성분 자료 설명](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DXX_J.htm)"
)
