
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
    "운동 일수에 따라 청소년의 근육량 추정 지표가 어떻게 달라지는지 비교합니다."
)

st.info(
    "남녀를 구분하지 않고 13~17세 전체 데이터를 분석합니다. "
    "CDC NHANES 2017~2018 실제 조사 자료를 사용합니다."
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
# XPT 데이터 다운로드
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

    # 운동 일수
    activity = activity[
        ["SEQN", "PAQ706"]
    ]

    # 팔과 다리의 제지방량
    body = body[
        [
            "SEQN",
            "DXDLALE",
            "DXDRALE",
            "DXDLLLE",
            "DXDRLLE"
        ]
    ]

    # 나이
    demo = demo[
        ["SEQN", "RIDAGEYR"]
    ]

    # 개인 식별번호로 데이터 결합
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
    numeric_columns = [
        "PAQ706",
        "DXDLALE",
        "DXDRALE",
        "DXDLLLE",
        "DXDRLLE",
        "RIDAGEYR"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # 13~17세 청소년만 선택
    data = data[
        (data["RIDAGEYR"] >= 13) &
        (data["RIDAGEYR"] <= 17)
    ].copy()

    # 운동 일수와 네 부위의 측정값이 모두 있는 사람
    data = data.dropna(
        subset=[
            "PAQ706",
            "DXDLALE",
            "DXDRALE",
            "DXDLLLE",
            "DXDRLLE"
        ]
    )

    # 유효한 운동 일수
    data = data[
        data["PAQ706"].between(0, 7)
    ].copy()

    # 네 부위의 제지방량 합산
    # 단위: g
    data["사지 제지방량(g)"] = (
        data["DXDLALE"] +
        data["DXDRALE"] +
        data["DXDLLLE"] +
        data["DXDRLLE"]
    )

    # g -> kg
    data["근육량 추정 지표(kg)"] = (
        data["사지 제지방량(g)"] / 1000
    )

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

with st.spinner("CDC 데이터를 다운로드하고 분석하고 있습니다..."):

    try:
        df = load_data()

    except Exception as e:
        st.error("데이터를 불러오는 중 오류가 발생했습니다.")
        st.code(str(e))
        st.stop()


# -------------------------------------
# 1. 분석 데이터
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
    "평균 근육량 추정 지표",
    f"{df['근육량 추정 지표(kg)'].mean():.2f}kg"
)

st.caption(
    "13~17세 남녀 전체 자료를 합쳐 분석합니다."
)


# -------------------------------------
# 2. 운동 그룹별 통계
# -------------------------------------

st.subheader("2. 운동량에 따른 근육량 추정 지표")

group_order = [
    "운동 적은 그룹 (0~1일)",
    "운동 중간 그룹 (2~3일)",
    "운동 많은 그룹 (4~5일)",
    "운동 매우 많은 그룹 (6~7일)"
]

summary = (
    df.groupby("운동 그룹")
    .agg(
        인원수=("근육량 추정 지표(kg)", "count"),
        평균=("근육량 추정 지표(kg)", "mean"),
        중앙값=("근육량 추정 지표(kg)", "median"),
        표준편차=("근육량 추정 지표(kg)", "std")
    )
    .reindex(group_order)
    .dropna(subset=["인원수"])
    .reset_index()
)

summary = summary.rename(
    columns={
        "평균": "평균 근육량 추정 지표(kg)",
        "중앙값": "중앙값(kg)",
        "표준편차": "표준편차(kg)"
    }
)

st.dataframe(
    summary.round(2),
    use_container_width=True,
    hide_index=True
)


# -------------------------------------
# 3. 운동 적은 그룹 vs 운동 많은 그룹
# -------------------------------------

st.subheader("3. 운동을 적게 한 사람과 많이 한 사람 비교")

low = df[
    df["PAQ706"].between(0, 1)
]

high = df[
    df["PAQ706"].between(6, 7)
]

if len(low) == 0 or len(high) == 0:

    st.warning(
        "두 그룹 중 데이터가 부족한 그룹이 있습니다."
    )

else:

    low_mean = low["근육량 추정 지표(kg)"].mean()
    high_mean = high["근육량 추정 지표(kg)"].mean()

    low_median = low["근육량 추정 지표(kg)"].median()
    high_median = high["근육량 추정 지표(kg)"].median()

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
        "평균 근육량 추정 지표(kg)": [
            low_mean,
            high_mean
        ],
        "중앙값(kg)": [
            low_median,
            high_median
        ]
    })

    # 그룹별 색상
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
            "평균 근육량 추정 지표(kg)": "{:.2f}",
            "중앙값(kg)": "{:.2f}"
        })
    )

    st.markdown(
        """
        🔴 **빨간색: 운동을 적게 한 그룹 (0~1일)**

        🔵 **파란색: 운동을 많이 한 그룹 (6~7일)**
        """
    )

    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True
    )

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
        "평균 차이",
        f"{difference:+.2f}kg"
    )


# -------------------------------------
# 4. 운동 일수별 평균 막대그래프
# -------------------------------------

st.subheader("4. 운동 일수별 평균 근육량 추정 지표")

daily_summary = (
    df.groupby("PAQ706")
    .agg(
        평균=("근육량 추정 지표(kg)", "mean"),
        인원수=("근육량 추정 지표(kg)", "count")
    )
    .reindex(range(8))
    .dropna()
    .reset_index()
)

daily_summary["운동 일수"] = (
    daily_summary["PAQ706"].astype(int).astype(str) + "일"
)


# 운동 일수별 색상
def get_bar_color(days):

    if days <= 1:
        return "#EF4444"

    elif days <= 3:
        return "#9CA3AF"

    elif days <= 5:
        return "#7DD3FC"

    else:
        return "#2563EB"


daily_summary["색상"] = daily_summary["PAQ706"].apply(
    get_bar_color
)

fig = px.bar(
    daily_summary,
    x="운동 일수",
    y="평균",
    text="평균",
    custom_data=["인원수"],
    labels={
        "운동 일수": "일주일 중 하루 60분 이상 운동한 날",
        "평균": "평균 근육량 추정 지표(kg)"
    },
    title="운동 일수별 평균 근육량 추정 지표"
)

fig.update_traces(
    marker_color=daily_summary["색상"],
    texttemplate="%{y:.2f}kg",
    textposition="outside",
    hovertemplate=(
        "운동 일수: %{x}<br>"
        "평균 근육량 추정 지표: %{y:.2f}kg<br>"
        "인원수: %{customdata[0]}명"
        "<extra></extra>"
    )
)

fig.update_layout(
    xaxis_title="일주일 중 하루 60분 이상 운동한 날",
    yaxis_title="평균 근육량 추정 지표(kg)",
    showlegend=False,
    xaxis=dict(
        categoryorder="array",
        categoryarray=[
            "0일", "1일", "2일", "3일",
            "4일", "5일", "6일", "7일"
        ]
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "빨간색: 0~1일 / 회색: 2~3일 / "
    "하늘색: 4~5일 / 파란색: 6~7일"
)


# -------------------------------------
# 5. 운동 일수별 조사 인원
# -------------------------------------

st.subheader("5. 운동 일수별 조사 인원")

count_fig = px.bar(
    daily_summary,
    x="운동 일수",
    y="인원수",
    text="인원수",
    labels={
        "운동 일수": "일주일 중 하루 60분 이상 운동한 날",
        "인원수": "인원수(명)"
    },
    title="운동 일수별 조사 참여 인원"
)

count_fig.update_traces(
    marker_color="#64748B",
    texttemplate="%{y:,}명",
    textposition="outside"
)

count_fig.update_layout(
    xaxis_title="일주일 중 하루 60분 이상 운동한 날",
    yaxis_title="인원수(명)",
    showlegend=False
)

st.plotly_chart(
    count_fig,
    use_container_width=True
)


# -------------------------------------
# 6. 분석 결과
# -------------------------------------

st.subheader("6. 분석 결과")

if len(low) > 0 and len(high) > 0:

    if difference > 0:

        st.write(
            f"운동을 6~7일 한 그룹의 평균 근육량 추정 지표가 "
            f"0~1일 한 그룹보다 {difference:.2f}kg 높게 나타났습니다."
        )

    elif difference < 0:

        st.write(
            f"운동을 6~7일 한 그룹의 평균 근육량 추정 지표가 "
            f"0~1일 한 그룹보다 {abs(difference):.2f}kg 낮게 나타났습니다."
        )

    else:

        st.write(
            "두 그룹의 평균 근육량 추정 지표는 같습니다."
        )

    st.caption(
        "두 그룹의 평균을 비교한 결과이며, 운동이 차이의 직접적인 원인이라는 뜻은 아닙니다."
    )


# -------------------------------------
# 7. 자료 출처 및 주의사항
# -------------------------------------

st.divider()

st.subheader("자료 출처 및 분석 시 주의사항")

st.markdown("""
**자료 출처**
- 미국 질병통제예방센터(CDC), NHANES 2017~2018
- 운동 자료: Physical Activity - Youth (PAQY_J)
- 체성분 자료: DXA - Whole Body (DXX_J)
- 인구통계 자료: Demographics (DEMO_J)

**근육량 추정 지표 계산 방법**

- 오른팔 제지방량: DXDRALE
- 왼팔 제지방량: DXDLALE
- 오른다리 제지방량: DXDRLLE
- 왼다리 제지방량: DXDLLLE

위 네 부위의 제지방량을 합산한 뒤 kg 단위로 변환했습니다.

이 값은 팔과 다리의 제지방량을 이용한 골격근량 추정 지표입니다.
DXA가 근육 조직만을 직접 분리해 측정하는 것은 아니므로
실제 순수 골격근량과 완전히 같지는 않습니다.

이번 분석은 남녀를 구분하지 않고 전체 자료를 합쳐 계산했습니다.
나이와 성별 등 다른 요인도 결과에 영향을 줄 수 있으므로
운동 일수와 근육량 추정 지표 사이의 관계를 관찰하는 분석으로 해석해야 합니다.
""")
