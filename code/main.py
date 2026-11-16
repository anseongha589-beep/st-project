
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from io import BytesIO

# -------------------------------------
# 기본 설정
# -------------------------------------

st.set_page_config(
    page_title="운동 시간은 많을수록 정말 좋은가?",
    page_icon="💪",
    layout="wide"
)

st.title("💪 운동 시간은 많을수록 정말 좋은가?")
st.markdown(
    "청소년의 신체활동 일수에 따라 제지방량에 차이가 있는지 분석합니다."
)

st.info(
    "CDC NHANES 2017~2018 실제 조사 자료를 사용합니다. "
    "데이터를 자동으로 내려받아 분석합니다."
)

# -------------------------------------
# 데이터 주소
# -------------------------------------

BASE_URL = (
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/"
    "Public/2017/DataFiles/"
)

FILES = {
    "PAQY_J": BASE_URL + "PAQY_J.XPT",
    "DXX_J": BASE_URL + "DXX_J.XPT",
    "DEMO_J": BASE_URL + "DEMO_J.XPT"
}


# -------------------------------------
# 데이터 다운로드 함수
# -------------------------------------

@st.cache_data
def download_xpt(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    data = pd.read_sas(
        BytesIO(response.content),
        format="xport"
    )

    data.columns = data.columns.str.upper()

    return data


# -------------------------------------
# 데이터 불러오기 및 결합
# -------------------------------------

@st.cache_data
def load_data():

    activity = download_xpt(FILES["PAQY_J"])
    body = download_xpt(FILES["DXX_J"])
    demo = download_xpt(FILES["DEMO_J"])

    activity = activity[["SEQN", "PAQ706"]]
    body = body[["SEQN", "DXDTOLE"]]
    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]

    # 개인 식별번호를 기준으로 데이터 결합
    data = activity.merge(
        body,
        on="SEQN",
        how="inner"
    )

    data = data.merge(
        demo,
        on="SEQN",
        how="inner"
    )

    # 숫자형 변환
    columns = [
        "PAQ706",
        "DXDTOLE",
        "RIDAGEYR",
        "RIAGENDR"
    ]

    for column in columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # 13~17세 청소년만 분석
    data = data[
        (data["RIDAGEYR"] >= 13) &
        (data["RIDAGEYR"] <= 17)
    ].copy()

    # 결측값 제거
    data = data.dropna(
        subset=["PAQ706", "DXDTOLE"]
    )

    # 유효한 값만 사용
    data = data[
        (data["PAQ706"] >= 0) &
        (data["PAQ706"] <= 7) &
        (data["DXDTOLE"] > 0)
    ].copy()

    # 제지방량 g -> kg
    data["제지방량(kg)"] = data["DXDTOLE"] / 1000

    # 운동 그룹 분류
    def classify(days):
        if days <= 1:
            return "운동 적은 그룹 (0~1일)"
        elif days <= 3:
            return "운동 중간 그룹 (2~3일)"
        elif days <= 5:
            return "운동 많은 그룹 (4~5일)"
        else:
            return "운동 매우 많은 그룹 (6~7일)"

    data["운동 그룹"] = data["PAQ706"].apply(classify)

    return data


# -------------------------------------
# 데이터 불러오기
# -------------------------------------

with st.spinner("CDC 실제 데이터를 다운로드하고 분석하고 있습니다..."):

    try:
        df = load_data()

    except Exception as e:
        st.error("데이터를 불러오는 중 오류가 발생했습니다.")
        st.code(str(e))
        st.stop()


# -------------------------------------
# 1. 분석 데이터 요약
# -------------------------------------

st.subheader("1. 분석 데이터")

col1, col2, col3 = st.columns(3)

col1.metric(
    "분석 대상 인원",
    f"{len(df):,}명"
)

col2.metric(
    "평균 운동 일수",
    f"{df['PAQ706'].mean():.2f}일"
)

col3.metric(
    "평균 제지방량",
    f"{df['제지방량(kg)'].mean():.2f}kg"
)

st.caption(
    "운동량과 체성분 자료가 모두 확인되는 13~17세 청소년을 대상으로 합니다."
)


# -------------------------------------
# 2. 운동 그룹별 통계
# -------------------------------------

st.subheader("2. 운동량에 따른 제지방량")

group_order = [
    "운동 적은 그룹 (0~1일)",
    "운동 중간 그룹 (2~3일)",
    "운동 많은 그룹 (4~5일)",
    "운동 매우 많은 그룹 (6~7일)"
]

summary = (
    df.groupby("운동 그룹")
    .agg(
        인원수=("제지방량(kg)", "count"),
        평균_제지방량=("제지방량(kg)", "mean"),
        중앙값=("제지방량(kg)", "median"),
        표준편차=("제지방량(kg)", "std")
    )
    .reindex(group_order)
    .dropna(subset=["인원수"])
    .reset_index()
)

summary["평균 제지방량(kg)"] = summary["평균_제지방량"].round(2)
summary["중앙값(kg)"] = summary["중앙값"].round(2)
summary["표준편차(kg)"] = summary["표준편차"].round(2)

result_table = summary[
    [
        "운동 그룹",
        "인원수",
        "평균 제지방량(kg)",
        "중앙값(kg)",
        "표준편차(kg)"
    ]
]

st.dataframe(
    result_table,
    use_container_width=True,
    hide_index=True
)


# -------------------------------------
# 3. 운동 적은 그룹과 많은 그룹 비교
# -------------------------------------

st.subheader("3. 운동을 적게 한 사람과 많이 한 사람 비교")

low = df[df["PAQ706"].between(0, 1)]
high = df[df["PAQ706"].between(6, 7)]

if len(low) == 0 or len(high) == 0:

    st.warning(
        "운동 적은 그룹 또는 운동 많은 그룹의 데이터가 부족합니다."
    )

else:

    low_mean = low["제지방량(kg)"].mean()
    high_mean = high["제지방량(kg)"].mean()

    low_median = low["제지방량(kg)"].median()
    high_median = high["제지방량(kg)"].median()

    difference = high_mean - low_mean

    comparison = pd.DataFrame({
        "운동 그룹": [
            "운동 적게 한 사람 (0~1일)",
            "운동 많이 한 사람 (6~7일)"
        ],
        "인원수": [
            len(low),
            len(high)
        ],
        "평균 제지방량(kg)": [
            low_mean,
            high_mean
        ],
        "중앙값(kg)": [
            low_median,
            high_median
        ]
    })

    # 빨간색과 파란색 행 스타일
    def color_group(row):

        if "적게" in row["운동 그룹"]:
            return [
                "background-color: #FDE2E2; color: #B91C1C"
            ] * len(row)

        return [
            "background-color: #DBEAFE; color: #1D4ED8"
        ] * len(row)

    styled_table = (
        comparison.style
        .apply(color_group, axis=1)
        .format({
            "평균 제지방량(kg)": "{:.2f}",
            "중앙값(kg)": "{:.2f}"
        })
    )

    st.markdown(
        """
        🔴 **빨간색: 운동을 적게 한 그룹**

        🔵 **파란색: 운동을 많이 한 그룹**
        """
    )

    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True
    )

    # 평균 차이
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "운동 적은 그룹 평균",
        f"{low_mean:.2f}kg",
        f"{len(low):,}명"
    )

    c2.metric(
        "운동 많은 그룹 평균",
        f"{high_mean:.2f}kg",
        f"{len(high):,}명"
    )

    c3.metric(
        "평균 제지방량 차이",
        f"{difference:+.2f}kg"
    )


# -------------------------------------
# 4. 운동 그룹별 평균 그래프
# -------------------------------------

st.subheader("4. 운동 그룹별 평균 제지방량 그래프")

fig = px.bar(
    summary,
    x="운동 그룹",
    y="평균_제지방량",
    text="평균_제지방량",
    labels={
        "운동 그룹": "운동 일수 그룹",
        "평균_제지방량": "평균 제지방량(kg)"
    },
    title="운동 일수 그룹별 평균 제지방량"
)

fig.update_traces(
    texttemplate="%{y:.2f}kg",
    textposition="outside"
)

fig.update_layout(
    xaxis_title="운동 그룹",
    yaxis_title="평균 제지방량(kg)",
    showlegend=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -------------------------------------
# 5. 운동 일수와 제지방량 산점도
# -------------------------------------

st.subheader("5. 운동 일수와 제지방량의 관계")

scatter = px.scatter(
    df,
    x="PAQ706",
    y="제지방량(kg)",
    color="운동 그룹",
    hover_data=["RIDAGEYR"],
    labels={
        "PAQ706": "일주일 중 하루 60분 이상 운동한 날",
        "제지방량(kg)": "제지방량(kg)",
        "RIDAGEYR": "나이"
    },
    title="운동 일수와 제지방량의 관계"
)

scatter.update_layout(
    xaxis=dict(dtick=1),
    xaxis_title="일주일 중 하루 60분 이상 운동한 날",
    yaxis_title="제지방량(kg)"
)

st.plotly_chart(
    scatter,
    use_container_width=True
)


# -------------------------------------
# 6. 분석 결과
# -------------------------------------

st.subheader("6. 분석 결과")

if len(low) > 0 and len(high) > 0:

    if difference > 0:

        st.write(
            f"운동을 6~7일 한 그룹의 평균 제지방량이 "
            f"0~1일 한 그룹보다 {difference:.2f}kg 높게 나타났습니다."
        )

    elif difference < 0:

        st.write(
            f"운동을 6~7일 한 그룹의 평균 제지방량이 "
            f"0~1일 한 그룹보다 {abs(difference):.2f}kg 낮게 나타났습니다."
        )

    else:

        st.write(
            "두 그룹의 평균 제지방량은 같습니다."
        )

    st.caption(
        "이 결과는 관찰된 평균 차이이며 운동이 차이의 직접적인 원인임을 증명하지는 않습니다."
    )


# -------------------------------------
# 7. 자료 출처 및 주의사항
# -------------------------------------

st.divider()

st.subheader("자료 출처 및 분석 시 주의사항")

st.markdown("""
- 자료 출처: 미국 질병통제예방센터(CDC), NHANES 2017~2018
- 운동 자료: Physical Activity - Youth (PAQY_J)
- 체성분 자료: DXA - Whole Body (DXX_J)
- 인구통계 자료: Demographics (DEMO_J)

**분석 지표 안내**

`DXDTOLE`은 뼈의 무기질 성분을 제외한 전체 제지방량입니다.

제지방량에는 골격근뿐 아니라 수분과 기타 제지방 조직도 포함되므로
순수한 근육량과 동일하지 않습니다.

또한 나이와 성별에 따라서도 제지방량은 달라질 수 있습니다.
따라서 이 분석은 운동량 그룹 간의 관찰된 차이를 보여주는 것이며,
운동 시간만으로 근육량 차이를 설명할 수는 없습니다.
""")
