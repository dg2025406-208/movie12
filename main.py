import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide"
)

DAILY_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/"
    "kobis_daily.csv"
)

MOVIES_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/"
    "kobis_movies.csv"
)


# =========================================================
# 2. 제목
# =========================================================

st.title("🎬 영화 흥행 예측기")

st.write(
    "영화 정보를 이용하여 영화의 **총 관객 수**를 예측하는 "
    "다중 회귀 모델입니다."
)

st.warning(
    "⚠️ **주의:** 이 데이터는 영화 개봉 후 집계된 사후 데이터입니다. "
    "따라서 이 모델의 결과는 실제 영화 개봉 전에 흥행을 예측하는 성능을 "
    "의미하지 않습니다. 특히 첫 주 관객 수나 TOP10 진입 일수처럼 "
    "개봉 후 알 수 있는 정보가 포함될 수 있으므로, "
    "이 결과는 **데이터 분석과 변수의 관계를 확인하기 위한 예측 실험**입니다."
)


# =========================================================
# 3. 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():

    daily = pd.read_csv(
        DAILY_URL,
        encoding="utf-8"
    )

    movies = pd.read_csv(
        MOVIES_URL,
        encoding="utf-8"
    )

    return daily, movies


try:

    daily, movies = load_data()

except Exception as e:

    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()


# =========================================================
# 4. 열 이름 정리
# =========================================================

daily.columns = daily.columns.astype(str).str.strip()
movies.columns = movies.columns.astype(str).str.strip()


# =========================================================
# 5. 일별 데이터에서 분석 기간 찾기
# =========================================================

# 사용자가 알려준 구조상 첫 번째 열이 날짜
daily_date_col = daily.columns[0]

daily_dates = pd.to_datetime(
    daily[daily_date_col].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

valid_dates = daily_dates.dropna()

if len(valid_dates) > 0:

    start_date = valid_dates.min()
    end_date = valid_dates.max()

    period_text = (
        f"{start_date.strftime('%Y년 %m월 %d일')} ~ "
        f"{end_date.strftime('%Y년 %m월 %d일')}"
    )

else:

    period_text = "날짜 정보를 확인할 수 없습니다."


# =========================================================
# 6. 영화코드 정리
# =========================================================

if "movieCd" not in movies.columns:

    st.error(
        "kobis_movies.csv에서 movieCd 열을 찾을 수 없습니다."
    )
    st.stop()


movies["movieCd"] = (
    movies["movieCd"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.strip()
)


# =========================================================
# 7. 분석 기간 표시
# =========================================================

st.subheader("📅 분석 기준")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "기준 기간",
        period_text
    )

with c2:
    st.metric(
        "영화별 영화 수",
        f"{len(movies):,}편"
    )

with c3:
    st.metric(
        "일별 데이터",
        f"{len(daily):,}행"
    )


# =========================================================
# 8. 영화별 표의 맨 위 행 그대로 표시
# =========================================================

st.subheader("🎞️ 영화별 표의 첫 번째 행")

st.dataframe(
    movies.head(1),
    use_container_width=True
)


# =========================================================
# 9. 영화코드 순으로 정렬
# =========================================================

movies = (
    movies
    .sort_values("movieCd")
    .reset_index(drop=True)
)


# =========================================================
# 10. 학습 / 테스트 데이터 분리
#
# 영화코드 순으로 정렬한 뒤
# 10편마다 앞의 3편 = 테스트
# 나머지 7편 = 학습
# =========================================================

test_mask = np.array(
    [(i % 10) < 3 for i in range(len(movies))]
)

test_df = movies.loc[test_mask].copy()
train_df = movies.loc[~test_mask].copy()


# =========================================================
# 11. 목표 변수 확인
# =========================================================

target_col = "total_audi"

if target_col not in movies.columns:

    st.error(
        "kobis_movies.csv에서 total_audi 열을 찾을 수 없습니다."
    )
    st.stop()


# =========================================================
# 12. 기본 변수 설정
#
# 기본 3개:
# first_scrn
# first_show
# days_in_top10
# =========================================================

basic_features = [
    "first_scrn",
    "first_show",
    "days_in_top10"
]

basic_features = [
    col for col in basic_features
    if col in movies.columns
]


if len(basic_features) < 3:

    st.warning(
        "기본 변수 3개 중 일부를 데이터에서 찾을 수 없습니다."
    )


# 첫 주 관객 수 추가
basic_plus_week_features = basic_features.copy()

if "first_week_audi" in movies.columns:

    basic_plus_week_features.append(
        "first_week_audi"
    )


# =========================================================
# 13. 사용자가 직접 선택할 변수
# =========================================================

st.subheader("🔧 예측에 사용할 변수 선택")

excluded_columns = {
    "movieCd",
    "movieNm",
    "total_audi"
}

available_features = [
    col for col in movies.columns
    if col not in excluded_columns
]


selected_features = []

feature_columns = st.columns(3)

for i, feature in enumerate(available_features):

    with feature_columns[i % 3]:

        checked = st.checkbox(
            feature,
            value=feature in basic_features,
            key=f"select_{feature}"
        )

        if checked:
            selected_features.append(feature)


if len(selected_features) == 0:

    st.warning(
        "예측에 사용할 변수를 최소 1개 선택해주세요."
    )
    st.stop()


st.info(
    "**현재 선택된 변수:** "
    + ", ".join(selected_features)
)


# =========================================================
# 14. 회귀 모델 만드는 함수
# =========================================================

def train_regression_model(
    train_data,
    test_data,
    features
):

    X_train = train_data[features].copy()
    X_test = test_data[features].copy()

    y_train = pd.to_numeric(
        train_data[target_col],
        errors="coerce"
    )

    y_test = pd.to_numeric(
        test_data[target_col],
        errors="coerce"
    )


    # 목표값이 없는 행 제외
    train_valid = y_train.notna()
    test_valid = y_test.notna()

    X_train = X_train.loc[train_valid]
    y_train = y_train.loc[train_valid]

    X_test = X_test.loc[test_valid]
    y_test = y_test.loc[test_valid]


    # 숫자형 / 문자형 변수 구분
    numeric_features = []
    categorical_features = []

    for col in features:

        if pd.api.types.is_numeric_dtype(
            X_train[col]
        ):

            numeric_features.append(col)

        else:

            categorical_features.append(col)


    # 숫자형 전처리
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    # 문자형 전처리
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )


    transformers = []


    if len(numeric_features) > 0:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )


    if len(categorical_features) > 0:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )


    preprocessor = ColumnTransformer(
        transformers=transformers
    )


    # 다중 선형 회귀
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "regression",
                LinearRegression()
            )
        ]
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    # 관객 수가 음수가 되지 않도록 처리
    predictions = np.maximum(
        predictions,
        0
    )


    # 평가 점수
    r2 = r2_score(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    # MAPE
    nonzero = y_test != 0

    if nonzero.sum() > 0:

        mape = (
            np.mean(
                np.abs(
                    (
                        y_test[nonzero].values
                        - predictions[nonzero]
                    )
                    /
                    y_test[nonzero].values
                )
            )
            * 100
        )

    else:

        mape = np.nan


    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }


# =========================================================
# 15. 사용자가 선택한 변수로 모델 학습
# =========================================================

try:

    main_result = train_regression_model(
        train_df,
        test_df,
        selected_features
    )

except Exception as e:

    st.error("선택한 변수로 모델을 학습하는 중 오류가 발생했습니다.")
    st.code(str(e))
    st.stop()


# =========================================================
# 16. 학습 / 평가 결과
# =========================================================

st.subheader("📊 선택한 변수로 평가한 결과")

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "학습에 사용한 영화",
        f"{len(main_result['X_train']):,}편"
    )

with c2:

    st.metric(
        "평가한 영화",
        f"{len(main_result['X_test']):,}편"
    )

with c3:

    st.metric(
        "R² 점수",
        f"{main_result['r2']:.3f}"
    )

with c4:

    st.metric(
        "평균 절대 오차",
        f"{main_result['mae']:,.0f}명"
    )


st.write(
    f"**기준 기간:** {period_text}"
)


# =========================================================
# 17. 기본 변수 3개 vs 첫 주 관객 추가 비교
# =========================================================

st.subheader(
    "🔍 기본 변수 3개와 첫 주 관객 수 추가 모델 비교"
)

try:

    basic_result = train_regression_model(
        train_df,
        test_df,
        basic_features
    )

    basic_plus_week_result = train_regression_model(
        train_df,
        test_df,
        basic_plus_week_features
    )

except Exception as e:

    st.error("두 모델을 비교하는 중 오류가 발생했습니다.")
    st.code(str(e))
    st.stop()


# =========================================================
# 비교표
# =========================================================

comparison_df = pd.DataFrame(
    {
        "모델": [
            "기본 변수 3개",
            "기본 변수 3개 + 첫 주 관객 수"
        ],
        "사용 변수": [
            ", ".join(basic_features),
            ", ".join(basic_plus_week_features)
        ],
        "R² 점수": [
            basic_result["r2"],
            basic_plus_week_result["r2"]
        ],
        "평균 절대 오차(MAE)": [
            basic_result["mae"],
            basic_plus_week_result["mae"]
        ],
        "RMSE": [
            basic_result["rmse"],
            basic_plus_week_result["rmse"]
        ]
    }
)


st.dataframe(
    comparison_df.style.format(
        {
            "R² 점수": "{:.3f}",
            "평균 절대 오차(MAE)": "{:,.0f}",
            "RMSE": "{:,.0f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 점수 차이
# =========================================================

r2_change = (
    basic_plus_week_result["r2"]
    - basic_result["r2"]
)

mae_change = (
    basic_plus_week_result["mae"]
    - basic_result["mae"]
)

rmse_change = (
    basic_plus_week_result["rmse"]
    - basic_result["rmse"]
)


st.info(
    f"""
### 📌 첫 주 관객 수를 추가했을 때

- **R² 변화:** {r2_change:+.3f}
- **MAE 변화:** {mae_change:+,.0f}명
- **RMSE 변화:** {rmse_change:+,.0f}명

R²는 **높을수록 좋고**, MAE와 RMSE는 **낮을수록 좋습니다.**
"""
)


# =========================================================
# 18. 테스트 영화의 예측 결과
# =========================================================

result_df = test_df.loc[
    main_result["y_test"].index
].copy()


result_df["실제 총 관객 수"] = (
    main_result["y_test"].values
)

result_df["예측 총 관객 수"] = (
    main_result["predictions"]
)

result_df["예측 오차"] = (
    result_df["예측 총 관객 수"]
    - result_df["실제 총 관객 수"]
)

result_df["절대 오차"] = (
    np.abs(
        result_df["예측 오차"]
    )
)

result_df["오차율(%)"] = np.where(
    result_df["실제 총 관객 수"] != 0,
    (
        result_df["절대 오차"]
        /
        result_df["실제 총 관객 수"]
        * 100
    ),
    np.nan
)


display_columns = [
    "movieCd",
    "movieNm",
    "실제 총 관객 수",
    "예측 총 관객 수",
    "예측 오차",
    "절대 오차",
    "오차율(%)"
]

display_columns = [
    col for col in display_columns
    if col in result_df.columns
]


st.subheader("🎯 테스트 영화별 예측 결과")

st.dataframe(
    result_df[
        display_columns
    ].sort_values("movieCd").style.format(
        {
            "실제 총 관객 수": "{:,.0f}",
            "예측 총 관객 수": "{:,.0f}",
            "예측 오차": "{:+,.0f}",
            "절대 오차": "{:,.0f}",
            "오차율(%)": "{:.1f}%"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 19. 1,000명 미만 예측
# =========================================================

predictions = main_result["predictions"]

under_1000 = predictions < 1000

under_1000_count = int(
    under_1000.sum()
)


st.subheader("⚠️ 1,000명 미만 예측 영화")

if under_1000_count > 0:

    st.warning(
        f"예측한 총 관객 수가 **1,000명보다 작은 영화는 "
        f"{under_1000_count}편**입니다."
    )

else:

    st.success(
        "예측한 총 관객 수가 1,000명보다 작은 영화는 없습니다."
    )


# =========================================================
# 20. 산점도
# =========================================================

st.subheader(
    "📈 실제 총 관객 수와 예측한 총 관객 수"
)


actual = main_result["y_test"].values.astype(float)
predicted = predictions.astype(float)


# 로그 스케일에서는 0을 사용할 수 없으므로 최소 1로 처리
actual_plot = np.maximum(
    actual,
    1
)

predicted_plot = np.maximum(
    predicted,
    1
)


# 1,000명 미만 예측은 그래프 바닥에 표시
predicted_floor = predicted_plot.copy()

floor_mask = predicted < 1000

predicted_floor[floor_mask] = 1


# 축 범위
all_values = np.concatenate(
    [
        actual_plot,
        predicted_plot
    ]
)

axis_min = max(
    1,
    float(np.min(all_values))
)

axis_max = max(
    10,
    float(np.max(all_values))
)


fig = go.Figure()


# ---------------------------------------------------------
# 실제값 = 예측값 대각선
# ---------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=[axis_min, axis_max],
        y=[axis_min, axis_max],
        mode="lines",
        name="실제값 = 예측값",
        line=dict(
            dash="dash"
        ),
        hoverinfo="skip"
    )
)


# ---------------------------------------------------------
# 일반 예측값
# ---------------------------------------------------------

normal_mask = ~floor_mask


normal_customdata = np.column_stack(
    [
        result_df.loc[
            normal_mask,
            "movieCd"
        ].astype(str).values,

        result_df.loc[
            normal_mask,
            "movieNm"
        ].astype(str).values,

        actual[normal_mask],

        predicted[normal_mask]
    ]
)


fig.add_trace(
    go.Scatter(
        x=actual_plot[normal_mask],
        y=predicted_floor[normal_mask],
        mode="markers",
        name="예측값",
        customdata=normal_customdata,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "영화코드: %{customdata[0]}<br>"
            "실제 총 관객: %{customdata[2]:,.0f}명<br>"
            "예측 총 관객: %{customdata[3]:,.0f}명"
            "<extra></extra>"
        )
    )
)


# ---------------------------------------------------------
# 1,000명 미만 예측값
# ---------------------------------------------------------

if under_1000_count > 0:

    floor_customdata = np.column_stack(
        [
            result_df.loc[
                floor_mask,
                "movieCd"
            ].astype(str).values,

            result_df.loc[
                floor_mask,
                "movieNm"
            ].astype(str).values,

            actual[floor_mask],

            predicted[floor_mask]
        ]
    )


    fig.add_trace(
        go.Scatter(
            x=actual_plot[floor_mask],
            y=predicted_floor[floor_mask],
            mode="markers",
            name="1,000명 미만 예측",
            customdata=floor_customdata,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "영화코드: %{customdata[0]}<br>"
                "실제 총 관객: %{customdata[2]:,.0f}명<br>"
                "예측 총 관객: %{customdata[3]:,.0f}명"
                "<br>⚠️ 1,000명 미만 예측"
                "<extra></extra>"
            )
        )
    )


# ---------------------------------------------------------
# 그래프 설정
# ---------------------------------------------------------

fig.update_layout(
    title="실제 총 관객 수 vs 예측한 총 관객 수",
    xaxis_title="실제 총 관객 수",
    yaxis_title="예측한 총 관객 수",
    height=650,
    hovermode="closest"
)


# X축 로그
fig.update_xaxes(
    type="log",
    range=[
        np.log10(axis_min),
        np.log10(axis_max)
    ]
)


# Y축 로그
fig.update_yaxes(
    type="log",
    range=[
        0,
        np.log10(axis_max)
    ]
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 21. 그래프 설명
# =========================================================

st.markdown(
    f"""
### 🔎 그래프에서 알 수 있는 점

- 대각선에 가까울수록 **실제 관객 수와 예측값이 비슷합니다.**
- 대각선보다 위에 있으면 **실제보다 많이 예측한 경우**입니다.
- 대각선보다 아래에 있으면 **실제보다 적게 예측한 경우**입니다.
- 영화별 관객 수의 차이가 매우 크기 때문에 **두 축을 로그 스케일**로 표시했습니다.
- 예측값이 1,000명 미만인 영화 **{under_1000_count}편**은 그래프의 바닥에 붙여 표시했습니다.
"""
)


# =========================================================
# 22. 학습 / 평가 방법 설명
# =========================================================

with st.expander("📌 학습과 평가 방법 자세히 보기"):

    st.write(
        "영화별 데이터를 영화코드(movieCd) 순으로 정렬한 뒤, "
        "10편마다 앞의 3편을 테스트용으로 떼어 놓고 "
        "나머지 7편을 학습용으로 사용했습니다."
    )

    st.write(
        f"• 전체 영화별 데이터: **{len(movies):,}편**"
    )

    st.write(
        f"• 학습에 사용한 영화: **{len(main_result['X_train']):,}편**"
    )

    st.write(
        f"• 평가한 영화: **{len(main_result['X_test']):,}편**"
    )

    st.write(
        f"• 기준 기간: **{period_text}**"
    )

    st.write(
        "• 평가에는 학습 과정에 사용하지 않은 테스트 영화를 사용했습니다."
    )

    st.write(
        "• 목표 변수는 영화의 최종 총 관객 수(total_audi)입니다."
    )
