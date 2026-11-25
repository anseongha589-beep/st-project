import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import numpy as np
from io import BytesIO

# =====================================
# 1. 기본 설정
# =====================================

st.set_page_config(
    page_title="운동 시간은 많을수록 정말 좋은가?",
    page_icon="💪",
    layout="wide"
)

st.title("💪 운동 시간은 많을수록 정말 좋은가?")
st.write(
    "운동 일수에 따라 청소년의 근육량 추정 지표가 "
    "어떻게 달라지는지 알아봅니다."
)

st.info(
    "CDC NHANES 2017~2018 자료를 사용합니다. "
    "분석 대상은 13~17세이며 남녀를 구분하지 않습니다."
)

# =====================================
# 2. 데이터 주소 및 변수
# =====================================

BASE_URL = (
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/"
    "Public/2017/DataFiles/"
)

FILES = {
    "운동 자료": BASE_URL + "PAQY_J.XPT",
    "체성분 자료": BASE_URL + "DXX_J.XPT",
    "나이 자료": BASE_URL + "DEMO_J.XPT"
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

KOREAN_NAMES = {
    "SEQN": "개인 식별번호",
    "PAQ706": "운동 일수",
    "RIDAGEYR": "나이",
    "DXDLALE": "왼팔 제지방량",
    "DXDRALE": "오른팔 제지방량",
    "DXDLLLE": "왼쪽 다리 제지방량",
    "DXDRLLE": "오른쪽 다리 제지방량",
    METRIC_COLUMN: METRIC_COLUMN,
    "운동 그룹": "운동 그룹"
}


# =====================================
# 3. XPT 데이터 다운로드
# =====================================

@st.cache_data
def download_xpt(url):

    response = requests.get(
        url,
        timeout=60
    )

    response.raise_for_status()

    data = pd.read_sas(
        BytesIO(response.content),
        format="xport"
    )

    data.columns = data.columns.str.upper()

    return data


# =====================================
# 4. 데이터 수집 및 전처리
# =====================================

@st.cache_data
def load_data():

    # 자료 다운로드
    activity = download_xpt(FILES["운동 자료"])
    body = download_xpt(FILES["체성분 자료"])
    demo = download_xpt(FILES["나이 자료"])

    # 필요한 열 선택
    activity = activity[
        ["SEQN", "PAQ706"]
    ]

    body = body[
        ["SEQN"] + DXA_COLUMNS
    ]

    demo = demo[
        ["SEQN", "RIDAGEYR"]
    ]

    # 개인 식별번호로 자료 결합
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

    # 13~17세 선택
    age_filtered = merged[
        merged["RIDAGEYR"].between(13, 17)
    ].copy()

    age_count = len(age_filtered)

    # 운동 일수 0~7일 선택 (부동소수점 오차 방지를 위해 반올림 후 정수 처리)
    activity_filtered = age_filtered[
        age_filtered["PAQ706"].between(0, 7)
    ].copy()
    activity_filtered["PAQ706"] = activity_filtered["PAQ706"].round().astype(int)

    activity_count = len(activity_filtered)

    # 팔과 다리의 제지방량 합산
    # 단위 g -> kg
    activity_filtered[METRIC_COLUMN] = (
        activity_filtered[DXA_COLUMNS].sum(
            axis=1,
            min_count=4
        ) / 1000
    )

    # 결측값 제거 전 결측값 개수
    missing_columns = REQUIRED_COLUMNS + [
        METRIC_COLUMN
    ]

    missing_info = pd.DataFrame({
        "변수": missing_columns,
        "결측값 개수": [
            int(activity_filtered[column].isna().sum())
            for column in REQUIRED_COLUMNS
        ] + [
            int(activity_filtered[METRIC_COLUMN].isna().sum())
        ]
    })

    # 분석에 필요한 값이 모두 있는 행만 선택
    complete_data = activity_filtered.dropna(
        subset=REQUIRED_COLUMNS + [METRIC_COLUMN]
    ).copy()

    complete_count = len(complete_data)

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

    complete_data["운동 그룹"] = (
        complete_data["PAQ706"].apply(classify)
    )

    # 전처리 단계별 행 수
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


# =====================================
# 5. 데이터 불러오기
# =====================================

with st.spinner("CDC 데이터를 불러오고 있습니다..."):

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


# =====================================
# 6. 데이터 개요
# =====================================

st.header("1. 데이터 개요")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "최종 분석 인원",
    f"{len(df):,}명"
)

col2.metric(
    "데이터 열 수",
    f"{len(df.columns)}개"
)

col3.metric(
    "전처리 전 결측값",
    f"{int(activity_filtered.isna().sum().sum()):,}개"
)

col4.metric(
    "평균 운동 일수",
    f"{df['PAQ706'].mean():.2f}일"
)

st.subheader("사용한 데이터 열")

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

column_info["열 이름"] = column_info["열 이름"].replace(
    KOREAN_NAMES
)

column_info["자료형"] = column_info["자료형"].replace({
    "float64": "숫자형",
    "int64": "정수형",
    "int32": "정수형",
    "object": "문자형"
})

st.dataframe(
    column_info,
    use_container_width=True,
    hide_index=True
)


# =====================================
# 7. 데이터 전처리 과정
# =====================================

st.header("2. 데이터 전처리 과정")

st.write(
    """
1. 운동량, 체성분, 인구통계 자료를 개인 식별번호로 결합했습니다.
2. 13~17세 청소년 자료를 선택했습니다.
3. 운동 일수가 0~7일인 자료를 선택했습니다.
4. 분석에 필요한 값이 비어 있는 행을 제거했습니다.
5. 팔과 다리의 제지방량을 합산해 근육량 추정 지표를 계산했습니다.
    """
)

st.subheader("전처리 단계별 남은 행 수")

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

c1.metric(
    "나이 조건으로 제외",
    f"{removed_age:,}행"
)

c2.metric(
    "운동 일수 조건으로 제외",
    f"{removed_activity:,}행"
)

c3.metric(
    "결측값으로 제외",
    f"{removed_missing:,}행"
)

st.write(f"**전체 제외 행 수:** {total_removed:,}행")
st.write(f"**최종 분석 행 수:** {complete_count:,}행")

st.subheader("결측값 처리 기준")

missing_info["변수"] = missing_info["변수"].replace(
    KOREAN_NAMES
)

st.dataframe(
    missing_info,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "결측값은 분석에 필요한 값이 비어 있는 경우입니다. "
    "이상치는 임의로 삭제하지 않았습니다."
)


# =====================================
# 8. 분석 데이터 요약
# =====================================

st.header("3. 분석 데이터 요약")

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
    f"{df[METRIC_COLUMN].mean():.2f}kg"
)


# =====================================
# 9. 운동 그룹별 통계
# =====================================

st.header("4. 운동 그룹별 평균 비교")

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

# 인원수를 정수로 변환
summary["인원수"] = summary["인원수"].astype(int)

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


# =====================================
# 10. 운동 일수별 평균 막대그래프
# =====================================

st.header("5. 운동 일수별 평균 근육량 추정 지표")

st.write(
    "막대가 높을수록 해당 운동 일수 그룹의 평균 "
    "근육량 추정 지표가 높다는 뜻입니다."
)

daily_summary = (
    df.groupby("PAQ706")
    .agg(
        평균=(METRIC_COLUMN, "mean"),
        인원수=(METRIC_COLUMN, "count"),
        표준편차=(METRIC_COLUMN, "std")
    )
    .reindex(range(8))
    .dropna(subset=["평균"])
    .reset_index()
)

# 인원수를 정수로 변환
daily_summary["인원수"] = daily_summary["인원수"].astype(int)
daily_summary["운동 일수"] = (
    daily_summary["PAQ706"].astype(int).astype(str) + "일"
)

# 색상 지정
bar_colors = []

for day in daily_summary["PAQ706"]:

    if day <= 1:
        bar_colors.append("#EF4444")

    elif day <= 3:
        bar_colors.append("#A1A1AA")

    elif day <= 5:
        bar_colors.append("#7DD3FC")

    else:
        bar_colors.append("#2563EB")


fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=daily_summary["운동 일수"],
        y=daily_summary["평균"],
        marker=dict(
            color=bar_colors,
            line=dict(
                color="white",
                width=1
            )
        ),
        text=[
            f"{value:.1f}kg"
            for value in daily_summary["평균"]
        ],
        textposition="outside",
        textfont=dict(size=13),
        customdata=np.column_stack([
            daily_summary["인원수"],
            daily_summary["표준편차"]
        ]),
        hovertemplate=(
            "<b>운동 일수: %{x}</b><br>"
            "평균 근육량 추정 지표: %{y:.2f}kg<br>"
            "조사 인원: %{customdata[0]:.0f}명<br>"
            "표준편차: %{customdata[1]:.2f}kg"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    height=550,
    title=dict(
        text="운동 일수에 따른 평균 비교",
        font=dict(size=23)
    ),
    xaxis=dict(
        title=dict(
            text="일주일 중 하루 60분 이상 운동한 날",
            font=dict(size=15)
        ),
        tickfont=dict(size=14),
        categoryorder="array",
        categoryarray=[
            "0일", "1일", "2일", "3일",
            "4일", "5일", "6일", "7일"
        ]
    ),
    yaxis=dict(
        title=dict(
            text="평균 근육량 추정 지표(kg)",
            font=dict(size=15)
        ),
        tickfont=dict(size=13),
        rangemode="tozero",
        gridcolor="#E5E7EB"
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(
        l=60,
        r=35,
        t=90,
        b=80
    ),
    showlegend=False,
    hoverlabel=dict(
        font_size=14
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown(
    """
    🔴 **빨간색:** 운동 적은 그룹 (0~1일)

    ⚪ **회색:** 운동 중간 그룹 (2~3일)

    🔵 **연한 파란색:** 운동 많은 그룹 (4~5일)

    🔵 **진한 파란색:** 운동 매우 많은 그룹 (6~7일)
    """
)


# =====================================
# 11. 운동 적은 그룹과 많은 그룹 비교
# =====================================

st.header("6. 운동을 적게 한 사람과 많이 한 사람 비교")

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

    st.dataframe(
        comparison.style
        .apply(
            lambda row: [
                (
                    "background-color: #FEE2E2; color: #991B1B"
                    if "적게" in row["운동 그룹"]
                    else "background-color: #DBEAFE; color: #1D4ED8"
                )
            ] * len(row),
            axis=1
        )
        .format({
            "인원수": "{:,}",
            "평균 근육량 추정 지표(kg)": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "운동 적은 그룹 평균",
        f"{low_mean:.2f}kg"
    )

    c2.metric(
        "운동 많은 그룹 평균",
        f"{high_mean:.2f}kg"
    )

    c3.metric(
        "평균 차이",
        f"{difference:+.2f}kg"
    )

    # 비교 전용 그래프
    compare_fig = go.Figure()

    compare_fig.add_trace(
        go.Bar(
            x=["운동 적은 그룹", "운동 많은 그룹"],
            y=[low_mean, high_mean],
            marker_color=["#EF4444", "#2563EB"],
            text=[
                f"{low_mean:.2f}kg",
                f"{high_mean:.2f}kg"
            ],
            textposition="outside",
            textfont=dict(size=16),
            customdata=[
                [len(low)],
                [len(high)]
            ],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "평균: %{y:.2f}kg<br>"
                "인원: %{customdata[0]}명"
                "<extra></extra>"
            )
        )
    )

    compare_fig.update_layout(
        title=dict(
            text="운동량이 적은 그룹과 많은 그룹의 평균 비교",
            font=dict(size=21)
        ),
        height=450,
        yaxis=dict(
            title="평균 근육량 추정 지표(kg)",
            rangemode="tozero",
            gridcolor="#E5E7EB"
        ),
        xaxis=dict(
            title="운동 그룹",
            tickfont=dict(size=14)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(
            l=60,
            r=30,
            t=80,
            b=60
        )
    )

    st.plotly_chart(
        compare_fig,
        use_container_width=True
    )

else:

    st.warning(
        "운동 적은 그룹 또는 운동 많은 그룹의 자료가 부족합니다."
    )


# =====================================
# 12. 운동 일수별 표
# =====================================

st.header("7. 운동 일수별 근육량 추정 지표 표")

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
        color = "background-color: #FEE2E2; color: #991B1B"

    elif days >= 6:
        color = "background-color: #DBEAFE; color: #1D4ED8"

    else:
        color = ""

    return [color] * len(row)

st.dataframe(
    table_data.style
    .apply(color_daily_table, axis=1)
    .format({
        "조사 인원(명)": "{:,}",
        "평균 근육량 추정 지표(kg)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =====================================
# 13. 선형회귀분석
# =====================================

st.header("8. 선형회귀분석")

st.write(
    "운동 일수를 독립변수(X), 근육량 추정 지표를 "
    "종속변수(Y)로 설정했습니다. 점 하나는 학생 1명이며, "
    "검은색 실선은 전체 추세선입니다."
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

slope, intercept = np.polyfit(
    X_train,
    Y_train,
    1
)

Y_pred = slope * X_test + intercept

mae = np.mean(np.abs(Y_test - Y_pred))

rmse = np.sqrt(
    np.mean((Y_test - Y_pred) ** 2)
)

ss_res = np.sum((Y_test - Y_pred) ** 2)
ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)

if ss_tot != 0:
    r2 = 1 - (ss_res / ss_tot)
else:
    r2 = np.nan

# 회귀선
x_line = np.linspace(
    X.min(),
    X.max(),
    100
)

y_line = slope * x_line + intercept

# 점들이 겹치지 않도록 가로 방향에 살짝 간격(jitter) 추가
jitter = rng.uniform(-0.18, 0.18, size=len(X))
X_jittered = X + jitter

# 그룹별 점 색상 지정 (위 막대그래프와 색상 통일)
point_colors = [
    "#EF4444" if day <= 1 else
    "#A1A1AA" if day <= 3 else
    "#7DD3FC" if day <= 5 else
    "#2563EB"
    for day in X
]

regression_fig = go.Figure()

regression_fig.add_trace(
    go.Scatter(
        x=X_jittered,
        y=Y,
        mode="markers",
        name="학생 개별 데이터 (1명 = 점 1개)",
        marker=dict(
            color=point_colors,
            opacity=0.65,
            size=8,
            line=dict(color="white", width=0.5)
        ),
        customdata=X.astype(int),
        hovertemplate=(
            "운동 일수: %{customdata}일<br>"
            "근육량 추정 지표: %{y:.2f}kg"
            "<extra></extra>"
        )
    )
)

regression_fig.add_trace(
    go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name="전체 평균 추세선 (회귀 직선)",
        line=dict(
            color="#111827",
            width=4,
            dash="solid"
        )
    )
)

regression_fig.update_layout(
    title=dict(
        text="운동 일수와 근육량 추정 지표의 관계 (개인 분포 및 추세선)",
        font=dict(size=22)
    ),
    height=550,
    xaxis_title="일주일 중 하루 60분 이상 운동한 날",
    yaxis_title="근육량 추정 지표(kg)",
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(
        tickmode="array",
        tickvals=[0, 1, 2, 3, 4, 5, 6, 7],
        ticktext=["0일", "1일", "2일", "3일", "4일", "5일", "6일", "7일"],
        range=[-0.5, 7.5],
        gridcolor="#E5E7EB"
    ),
    yaxis=dict(
        gridcolor="#E5E7EB"
    ),
    legend=dict(
        orientation="h",
        y=1.12,
        font=dict(size=14)
    )
)

st.plotly_chart(
    regression_fig,
    use_container_width=True
)

equation_sign = "+" if intercept >= 0 else "-"

st.subheader("회귀식")

st.code(
    f"근육량 추정 지표 = {slope:.4f} × 운동 일수 "
    f"{equation_sign} {abs(intercept):.4f}"
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "결정계수 (R²)",
    f"{r2:.4f}" if not np.isnan(r2) else "계산 불가"
)

c2.metric(
    "평균절대오차 (MAE)",
    f"{mae:.2f}kg"
)

c3.metric(
    "평균제곱근오차 (RMSE)",
    f"{rmse:.2f}kg"
)

st.subheader("회귀분석 해석")

if slope > 0:

    st.write(
        f"회귀식의 기울기는 {slope:.4f}입니다. "
        "운동 일수가 증가할수록 근육량 추정 지표가 "
        "증가하는 방향의 관계가 나타났습니다."
    )

elif slope < 0:

    st.write(
        f"회귀식의 기울기는 {slope:.4f}입니다. "
        "운동 일수가 증가할수록 근육량 추정 지표가 "
        "감소하는 방향의 관계가 나타났습니다."
    )

else:

    st.write(
        "회귀식의 기울기가 0에 가까워 "
        "선형적인 관계가 거의 나타나지 않았습니다."
    )

st.caption(
    "회귀분석은 두 변수의 관계를 보여줄 뿐, "
    "운동이 근육량을 변화시켰다는 원인을 증명하지는 않습니다."
)


# =====================================
# 14. 결과 및 한계
# =====================================

st.header("9. 분석 결과 및 한계")

st.write(
    """
    이번 분석에서는 운동 일수와 근육량 추정 지표의 관계를
    그룹별 평균 비교와 선형회귀분석으로 확인했습니다.

    하지만 운동 일수 외에도 나이, 성장 상태, 식습관,
    운동 강도 등 다양한 요인이 영향을 줄 수 있습니다.

    따라서 운동 일수만으로 근육량 차이의 원인을
    모두 설명할 수는 없습니다.
    """
)


# =====================================
# 15. 데이터 출처
# =====================================

st.divider()

st.header("10. 데이터 출처 및 주의사항")

st.markdown("""
**자료 출처**
- 미국 질병통제예방센터(CDC), NHANES 2017~2018
- 운동 자료: Physical Activity - Youth (PAQY_J)
- 체성분 자료: DXA - Whole Body (DXX_J)
- 인구통계 자료: Demographics (DEMO_J)

**사용한 변수**
- PAQ706: 일주일 중 하루 60분 이상 운동한 날의 수
- DXDLALE: 왼팔 제지방량
- DXDRALE: 오른팔 제지방량
- DXDLLLE: 왼쪽 다리 제지방량
- DXDRLLE: 오른쪽 다리 제지방량
- RIDAGEYR: 나이

**근육량 추정 지표 계산**
왼팔, 오른팔, 왼쪽 다리, 오른쪽 다리의 제지방량을 합산한 뒤 kg 단위로 변환했습니다.

이 값은 순수 골격근량과 같지 않으며, 근육량을 추정하기 위한 지표입니다.
남녀를 구분하지 않고 분석했습니다.
""")

st.caption(
    "분석 결과는 조사 자료에 나타난 관계를 보여주며 "
    "모든 사람에게 동일하게 적용되는 결과는 아닙니다."
)
