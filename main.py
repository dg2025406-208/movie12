# =========================================================
# 변수 선택
# =========================================================

st.subheader("🔧 예측에 사용할 변수 선택")

target_col = "total_audi"

excluded_columns = {
    "movieCd",
    "movieNm",
    target_col
}

available_features = [
    c for c in movies.columns
    if c not in excluded_columns
]

# 기본 변수 3개
basic_features = [
    "first_scrn",
    "first_show",
    "days_in_top10"
]

# 실제 데이터에 존재하는 변수만 사용
basic_features = [
    c for c in basic_features
    if c in available_features
]

# 기본 3개 + 첫 주 관객
basic_plus_week_features = basic_features.copy()

if "first_week_audi" in available_features:
    basic_plus_week_features.append("first_week_audi")


selected_features = []

feature_cols = st.columns(3)

for i, feature in enumerate(available_features):

    with feature_cols[i % 3]:

        checked = st.checkbox(
            feature,
            value=feature in basic_features,
            key=f"feature_{feature}"
        )

        if checked:
            selected_features.append(feature)


if not selected_features:
    st.warning("최소 1개의 변수를 선택해주세요.")
    st.stop()


st.info(
    f"현재 선택된 변수: **{len(selected_features)}개**  \n"
    + ", ".join(selected_features)
)
