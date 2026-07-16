import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb

st.set_page_config(
    page_title = '泰坦尼克号生存预测',
    page_icon="",
    layout="wide",
    initial_sidebar_state = "expanded"
)

st.title("泰坦尼克号生存预测")
st.markdown("""
    <style>
    /* 主标题样式 */
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* 卡片样式 */
    .card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* 结果卡片 */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    
    .result-card h2 {
        color: white;
        margin: 0;
    }
    
    /* 存活/死亡标签 */
    .survived {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 0.5rem 2rem;
        border-radius: 30px;
        color: white;
        font-weight: bold;
        font-size: 1.5rem;
        display: inline-block;
    }
    
    .deceased {
        background: linear-gradient(135deg, #cb2d3e, #ef473a);
        padding: 0.5rem 2rem;
        border-radius: 30px;
        color: white;
        font-weight: bold;
        font-size: 1.5rem;
        display: inline-block;
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 侧边栏样式 */
    .sidebar-info {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* 统计数字样式 */
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('titanic_xgb_model.pkl')

model = load_model()
st.sidebar.success("模型加载成功")

st.sidebar.header("乘客信息")

pclass = st.sidebar.selectbox("舱位等级 （Pclass）", [1, 2, 3], format_func=lambda x: {1: "头等舱", 2:"二等舱", 3:"三等舱"}[x])
sex = st.sidebar.selectbox("性别（Sex）", ["男性","女性"])
age = st.sidebar.slider("年龄(Age)",0, 80, 25)
sibsp = st.sidebar.number_input("兄弟姐妹/配偶数(SibSp)", min_value=0, max_value=8, value=0, step=1)
parch = st.sidebar.number_input("父母/子女数(Parch)", min_value=0, max_value=6, value=0, step=1)
fare = st.sidebar.number_input("船票价格(Fare)", min_value=0.0, max_value=512.0, value=32.0, step=1.0)
embarked = st.sidebar.selectbox("登船港口（Embarked）", ["南安普顿（S）", "瑟堡(C)", "皇后镇(Q)"])

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

if age <=12:
    age_bin = "儿童"
elif age <= 18:
    age_bin = "青少年"
elif age <= 35:
    age_bin = "青年"
elif age <= 60:
    age_bin = "中年"
else:
    age_bin="老年"

if fare <= 8.0:
    fare_bin ="低"
elif fare <= 15.0:
    fare_bin = "中低"
elif fare <=31.0:
    fare_bin = "中高"
else:
    fare_bin = "高"

sex_map = {"男性":1, "女性":0}
embarked_map = {"南安普顿（S）":0, "瑟堡(C)":1, "皇后镇(Q)":2}
age_bin_map = {"儿童":0, "青少年":1, "青年":2, "中年":3, "老年":4}
fare_bin_map = {"低":0, "中低":1, "中高":2, "高":3}

if st.sidebar.button("预测生存概率"):

    input_data = pd.DataFrame({
        'pclass': [pclass],
        'sex': [sex_map[sex]],
        'age': [age],
        'fare': [fare],
        'sibsp': [sibsp],
        'parch': [parch],
        'FamilySize': [family_size],
        'IsAlone': [is_alone],
        'AgeBin': [age_bin_map[age_bin]],
        'FareBin': [fare_bin_map[fare_bin]],
        'embarked': [embarked_map[embarked]]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.markdown(f"""<div class="prediction-box survived">
             预测结果：生还<br>
        <span style="font-size:16px;">生存概率: {probability:.2%}</span>
    </div>
    """, unable_allow_html=True)
        else:
            st.markdown(f"""
    <div class="prediction-box perished">
             预测结果：未生还<br>
        <span style="font-size:16px;">生存概率: {probability:.2%}</span>
    </div>
    """, unsafe_allow_html=True)

    with col2:
        st.metric(label="生存概率", value=f"{probability:.2%}")
        


    st.subheader("预测解释（SHAP）瀑布图")

    try:

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        fig, ax = plt.subplots(figsize=(10, 5))

        feature_names = input_data.columns.tolist()
        values = shap_values[0]

        sorted_idx = np.argsort(np.abs(values))[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_balues = [values[i] for i in sorted_idx]

        colors =['red' if v < 0 else 'blue' for v in sorted_values]
        plt.barh(sorted_names, sorted_values, color=colors, alpha=0.7)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('SHAP 值 → 降低生存概率，正值 → 提高生存概率）')
        plt.title('每个特征对预测结果的贡献')
        plt.tight_layout()

        st.pyplot(fig)
        plt.clf()
       
    except Exception as e:
        st.warning(f"SHAP 图生成失败: {e}")

st.sidebar.markdown("--")
st.sidebar.subheader("快速体验")

col1, col2 = st.sidebar.columns(2)
if col1.button("女性 头等舱"):
    st.session_state.pclass = 1
    st.session_state.sex = "男性"
    st.session_state.age = 25
    st.session_state.fare = 8.0
    st.session_state.sibsp = 0
    st.session_state.parch = 0
    st.session_state.embarked = "南安普顿(S)"
    st.rerun()

if col2.button("男性 三等舱"):
    st.session_state.pclass = 3
    st.session_state.sex = "男性"
    st.session_state.age = 25
    st.session_state.fare = 8.0
    st.session_state.sibsp = 0
    st.session_state.parch = 0
    st.session_state.embarked = "南安普顿(S)"
    st.rerun()

st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.caption("基于 XGBoost 模型 | 数据来自 Kaggle Titanic")
