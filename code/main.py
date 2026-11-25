
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import numpy as np
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
st.write(
    "운동 일수에 따라 청소년의 근육량 추정 지표가 "
    "어떻게 달라지는지 분석합니다."
)

st.info(
    "CDC NHANES 2017~2018 자료를 사용합니다. "
    "분석 대상은 13~17세이며 남녀를 구분하지 않습니다."
)

# -------------------------------------
# 데이터 주소 및 변수
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

DXA_COLUMNS = [
    "DXDLALE",
    "DXDRALE",
    "DXDLLLE",
    "DXDRLLE"
]

REQUIRED_COLUMNS = [
    "PAQ706",
    "RIDAGEYR",
    *DXA_COLUMNS
]

METRIC_COLUMN = "근육량 추정 지표(kg)"


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
# 데이터 수집 및 전처리
# -------------------------------------

@st.cache_data
def load_data():

    # 1. CDC 자료 다운로드
    activity = download_xpt(FILES["PAQY_J"])
    body = download_xpt(FILES["DXX_J"])
    demo = download_xpt(FILES["DEMO_J"])

    # 2. 필요한 열 선택
    activity = activity[
        ["SEQN", "PAQ706"]
    ]

    body = body[
        ["SEQN"] + DXA_COLUMNS
    ]

    demo = demo[
        ["SEQN", "RIDAGEYR"]
    ]

    # 3. 개인 식별번호로 자료 결합
    merged = activity.merge(
        body,
        on="SEQN",
        how="inner"
    )

    merged = merged.merge(
        demo,
        on="SEQN",
        how="inner"
    )

    merged_count = len(merged)

    # 숫자형으로 변환
    for column in REQUIRED_COLUMNS:
        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce"
        )

    # 4. 13~17세 자료 선택
    age_filtered = merged[
        merged["RIDAGEYR"].between(13, 17)
    ].copy()

    age_count = len(age_filtered)

    # 5. 운동 일수 0~7일 자료 선택
    activity_filtered = age_filtered[
        age_filtered["PAQ706"].between(0, 7)
    ].copy()

    activity_count = len(activity_filtered)

    # 6. 근육량 추정 지표 계산
    # 팔과 다리의 제지방량을 합산
    # 단위: g -> kg
    activity_filtered[METRIC_COLUMN] = (
        activity_filtered[DXA_COLUMNS].sum(
            axis=1,
            min_count=4
        ) / 1000
    )

    # 7. 결측값 제거 전 데이터의 결측값 개수 저장
    missing_info = pd.DataFrame({
        "변수": REQUIRED_COLUMNS + [METRIC_COLUMN],
        "결측값 개수": [
            int(activity_filtered[column].isna().sum())
            for column in REQUIRED_COLUMNS
        ] + [
            int(activity_filtered[METRIC_COLUMN].isna().sum())
        ]
    })

    # 8. 분석에 필요한 값이 모두 있는 행만 선택
    complete_data = activity_filtered.dropna(
        subset=REQUIRED_COLUMNS + [METRIC_COLUMN]
    ).copy()

    complete_count = len(complete_data)

    # 9. 운동 그룹 분류
    def classify(days):

        if days <= 1:
            return "운동 적은 그룹 (0~1일)"

        elif days <= 3:
            return "운동 중간 그룹 (2~3일)"

        elif days <= 5:
            return "운동 많은 그룹 (4~5일)"

        else:
            return "운동 매우 많은 그룹 (6~7일)"

    complete_data["운동 그룹"] = (
        complete_data["PAQ706"].apply(classify)
    )

    # 10. 전처리 단계별 행 수
    preprocessing = pd.DataFrame({
        "전처리 단계": [
            "세 자료 결합 후",
            "13~17세 선택 후",
            "운동 일수 유효 범위 선택 후",
            "결측값 제거 후"
        ],
        "남은 행 수": [
            merged_count,
            age_count,
            activity_count,
            complete_count
        ]
    })

    # 중요: activity_filtered와 missing_info도 함수 밖으로 반환
    return (
        complete_data,
        activity_filtered,
        preprocessing,
        missing_info,
        merged_count,
        age_count,
        activity_count,
        complete_count
    )


# -------------------------------------
# 데이터 불러오기
# -------------------------------------

with st.spinner("CDC 데이터를 다운로드하고 분석하고 있습니다..."):

    try:
        (
            df,
            activity_filtered,
            preprocessing,
            missing_info,
            merged_count,
            age_count,
            activity_count,
            complete_count
        ) = load_data()

    except Exception as e:
        st.error("데이터를 불러오는 중 오류가 발생했습니다.")
        st.code(str(e))
        st.stop()

if len(df) < 5:
    st.error("분석에 필요한 데이터가 충분하지 않습니다.")
    st.stop()


# -------------------------------------
# 1. 데이터 개요
# -------------------------------------

st.subheader("1. 데이터 개요")

col1, col2, col3, col4 = st.columns(4)

col1.metric("최종 분석 인원", f"{len(df):,}명")
col2.metric("데이터 열 수", f"{len(df.columns)}개")
col3.metric(
    "전처리 전 결측값",
    f"{int(activity_filtered.isna().sum().sum()):,}개"
)
col4.metric(
    "평균 운동 일수",
    f"{df['PAQ706'].mean():.2f}일"
)

st.markdown("#### 사용한 데이터 열")

# 결측값 제거 전 데이터를 기준으로 표시
column_info = pd.DataFrame({
    "열 이름": activity_filtered.columns,
    "자료형": [
        str(activity_filtered[column].dtype)
        for column in activity_filtered.columns
    ],
    "결측값 개수": [
        int(activity_filtered[column].isna().sum())
        for column in activity_filtered.columns
    ]
})

st.dataframe(
    column_info,
    use_container_width=True,
    hide_index=True
)


# -------------------------------------
# 2. 데이터 전처리 과정
# -------------------------------------

st.subheader("2. 데이터 전처리 과정")

st.write(
    """
    1. 운동량, 체성분, 인구통계 자료를 개인 식별번호로 결합했습니다.
    2. 13~17세 청소년 자료를 선택했습니다.
    3. 운동 일수가 0~7일인 자료를 선택했습니다.
    4. 분석에 필요한 값이 비어 있는 행을 제거했습니다.
    5. 팔과 다리의 제지방량을 합산해 근육량 추정 지표를 계산했습니다.
    """
)

st.markdown("#### 전처리 단계별 행 수")

st.dataframe(
    preprocessing,
    use_container_width=True,
    hide_index=True
)

removed_age = merged_count - age_count
removed_activity = age_count - activity_count
removed_missing = activity_count - complete_count
total_removed = merged_count - complete_count

c1, c2, c3 = st.columns(3)

c1.metric("나이 조건으로 제외", f"{removed_age:,}행")
c2.metric("운동 일수 조건으로 제외", f"{removed_activity:,}행")
c3.metric("결측값으로 제외", f"{removed_missing:,}행")

st.write(f"**전체 제외 행 수:** {total_removed:,}행")
st.write(f"**최종 분석 행 수:** {complete_count:,}행")


# -------------------------------------
# 3. 결측값 처리 기준
# -------------------------------------

st.subheader("3. 결측값 처리 기준")

st.dataframe(
    missing_info,
    use_container_width=True,
    hide_index=True
)

st.markdown(
    """
    - 처리 기준: 운동 일수, 나이, 팔·다리 제지방량 중 하나라도 비어 있는 행은 제외했습니다.
    - 이상치 처리: 이상치는 별도로 삭제하지 않았습니다.
    - 결측값 개수는 결측값을 제거하기 전 데이터를 기준으로 표시합니다.
    """
)


# -------------------------------------
# 4. 분석 데이터 요약
# -------------------------------------

st.subheader("4. 분석 데이터 요약")

col1, col2, col3 = st.columns(3)

col1.metric("분석 대상 인원", f"{len(df):,}명")
col2.metric(
    "평균 운동 일수",
    f"{df['PAQ706'].mean():.2f}일"
)
col3.metric(
    "평균 근육량 추정 지표",
    f"{df[METRIC_COLUMN].mean():.2f}kg"
)


# -------------------------------------
# 5. 운동 그룹별 통계
# -------------------------------------

st.subheader("5. 운동량에 따른 근육량 추정 지표")

group_order = [
    "운동 적은 그룹 (0~1일)",
    "운동 중간 그룹 (2~3일)",
    "운동 많은 그룹 (4~5일)",
    "운동 매우 많은 그룹 (6~7일)"
]

summary = (
    df.groupby("운동 그룹")
    .agg(
        인원수=(METRIC_COLUMN, "count"),
        평균=(METRIC_COLUMN, "mean"),
        중앙값=(METRIC_COLUMN, "median"),
        표준편차=(METRIC_COLUMN, "std")
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
# 6. 운동 적은 그룹과 많은 그룹 비교
# -------------------------------------

st.subheader("6. 운동을 적게 한 사람과 많이 한 사람 비교")

low = df[df["PAQ706"].between(0, 1)]
high = df[df["PAQ706"].between(6, 7)]

if len(low) > 0 and len(high) > 0:

    low_mean = low[METRIC_COLUMN].mean()
    high_mean = high[METRIC_COLUMN].mean()
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
        ]
    })

    def color_group(row):
        if "적게" in row["운동 그룹"]:
            color = (
                "background-color: #FDE2E2; "
                "color: #B91C1C"
            )
        else:
            color = (
                "background-color: #DBEAFE; "
                "color: #1D4ED8"
            )

        return [color] * len(row)

    st.dataframe(
        comparison.style
        .apply(color_group, axis=1)
        .format({
            "평균 근육량 추정 지표(kg)": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("운동 적은 그룹 평균", f"{low_mean:.2f}kg")
    c2.metric("운동 많은 그룹 평균", f"{high_mean:.2f}kg")
    c3.metric("평균 차이", f"{difference:+.2f}kg")

else:
    st.warning("비교할 그룹의 자료가 충분하지 않습니다.")


# -------------------------------------
# 7. 운동 일수별 평균 막대그래프
# -------------------------------------

st.subheader("7. 운동 일수별 평균 근육량 추정 지표")

daily_summary = (
    df.groupby("PAQ706")
    .agg(
        평균=(METRIC_COLUMN, "mean"),
        인원수=(METRIC_COLUMN, "count")
    )
    .reindex(range(8))
    .dropna()
    .reset_index()
)

daily_summary["운동 일수"] = (
    daily_summary["PAQ706"].astype(int).astype(str) + "일"
)

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

st.plotly_chart(fig, use_container_width=True)


# -------------------------------------
# 8. 운동 일수별 표
# -------------------------------------

st.subheader("8. 운동 일수별 근육량 추정 지표 표")

table_data = daily_summary[
    ["운동 일수", "인원수", "평균"]
].copy()

table_data = table_data.rename(
    columns={
        "인원수": "조사 인원(명)",
        "평균": "평균 근육량 추정 지표(kg)"
    }
)

def color_daily_table(row):

    days = int(row["운동 일수"].replace("일", ""))

    if days <= 1:
        color = "background-color: #FDE2E2; color: #B91C1C"
    elif days >= 6:
        color = "background-color: #DBEAFE; color: #1D4ED8"
    else:
        color = ""

    return [color] * len(row)

st.dataframe(
    table_data.style
    .apply(color_daily_table, axis=1)
    .format({
        "평균 근육량 추정 지표(kg)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# -------------------------------------
# 9. 선형회귀분석
# -------------------------------------

st.subheader("9. 선형회귀분석")

st.write(
    "운동 일수를 독립변수(X), 근육량 추정 지표를 종속변수(Y)로 설정합니다."
)

X = df["PAQ706"].to_numpy(dtype=float)
Y = df[METRIC_COLUMN].to_numpy(dtype=float)

rng = np.random.default_rng(42)
indices = rng.permutation(len(df))

split_index = int(len(df) * 0.8)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X[train_indices]
Y_train = Y[train_indices]

X_test = X[test_indices]
Y_test = Y[test_indices]

# 회귀 모델 학습
slope, intercept = np.polyfit(
    X_train,
    Y_train,
    1
)

# 평가 데이터 예측
Y_pred = slope * X_test + intercept

# 오차 계산
mae = np.mean(np.abs(Y_test - Y_pred))
rmse = np.sqrt(np.mean((Y_test - Y_pred) ** 2))

ss_res = np.sum((Y_test - Y_pred) ** 2)
ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)

if ss_tot != 0:
    r2 = 1 - (ss_res / ss_tot)
else:
    r2 = np.nan

# 회귀 그래프
x_line = np.linspace(X.min(), X.max(), 100)
y_line = slope * x_line + intercept

regression_fig = go.Figure()

regression_fig.add_trace(
    go.Scatter(
        x=X,
        y=Y,
        mode="markers",
        name="조사 데이터",
        marker=dict(
            color="#60A5FA",
            opacity=0.55,
            size=7
        )
    )
)

regression_fig.add_trace(
    go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name="선형회귀 직선",
        line=dict(
            color="#EF4444",
            width=3
        )
    )
)

regression_fig.update_layout(
    title="운동 일수와 근육량 추정 지표의 회귀분석",
    xaxis_title="일주일 중 하루 60분 이상 운동한 날",
    yaxis_title="근육량 추정 지표(kg)"
)

st.plotly_chart(
    regression_fig,
    use_container_width=True
)

equation_sign = "+" if intercept >= 0 else "-"

st.markdown("#### 회귀식")

st.code(
    f"근육량 추정 지표 = {slope:.4f} × 운동 일수 "
    f"{equation_sign} {abs(intercept):.4f}"
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "결정계수 (R²)",
    f"{r2:.4f}" if not np.isnan(r2) else "계산 불가"
)

c2.metric("평균절대오차 (MAE)", f"{mae:.2f}kg")
c3.metric("평균제곱근오차 (RMSE)", f"{rmse:.2f}kg")

st.markdown("#### 회귀분석 해석")

if slope > 0:
    st.write(
        f"회귀식의 기울기는 {slope:.4f}입니다. "
        "운동 일수가 증가할수록 근육량 추정 지표가 증가하는 "
        "방향의 관계가 나타났습니다."
    )
elif slope < 0:
    st.write(
        f"회귀식의 기울기는 {slope:.4f}입니다. "
        "운동 일수가 증가할수록 근육량 추정 지표가 감소하는 "
        "방향의 관계가 나타났습니다."
    )
else:
    st.write(
        "회귀식의 기울기가 0에 가까워 "
        "선형적인 관계가 거의 나타나지 않았습니다."
    )

st.caption(
    "R², MAE, RMSE는 평가 데이터에서 계산했습니다. "
    "이 결과는 관계를 보여줄 뿐 운동이 근육량을 증가시킨다는 "
    "원인을 증명하지 않습니다."
)


# -------------------------------------
# 10. 분석 결과와 한계
# -------------------------------------

st.subheader("10. 분석 결과 및 한계")

st.write(
    """
    이번 분석에서는 운동 일수와 근육량 추정 지표의 관계를
    그룹별 평균 비교와 선형회귀분석으로 확인했습니다.

    하지만 운동 일수 외에도 나이, 성장 상태, 식습관, 운동 강도 등
    다양한 요인이 근육량 추정 지표에 영향을 줄 수 있습니다.

    따라서 운동 일수만으로 근육량 차이의 원인을
    모두 설명할 수는 없습니다.
    """
)


# -------------------------------------
# 11. 데이터 출처
# -------------------------------------

st.divider()

st.subheader("11. 데이터 출처 및 주의사항")

st.markdown("""
**자료 출처**
- 미국 질병통제예방센터(CDC), NHANES 2017~2018
- 운동 자료: Physical Activity - Youth (PAQY_J)
- 체성분 자료: DXA - Whole Body (DXX_J)
- 인구통계 자료: Demographics (DEMO_J)

**사용한 변수**
- PAQ706: 지난 7일 동안 하루 60분 이상 신체활동을 한 날의 수
- DXDLALE: 왼팔 제지방량
- DXDRALE: 오른팔 제지방량
- DXDLLLE: 왼다리 제지방량
- DXDRLLE: 오른다리 제지방량
- RIDAGEYR: 나이

**근육량 추정 지표 계산**
왼팔, 오른팔, 왼다리, 오른다리의 제지방량을 합산한 뒤 kg 단위로 변환했습니다.

이 값은 순수 골격근량과 같지 않으며 근육량을 추정하기 위한 지표입니다.
남녀를 구분하지 않고 분석했습니다.
""")

st.caption(
    "이 앱의 분석 결과는 조사 자료에 나타난 관계를 보여주며, "
    "모든 사람에게 동일하게 적용되는 결과는 아닙니다."
)
