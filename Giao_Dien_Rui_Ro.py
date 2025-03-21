import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Dự Đoán Rủi Ro Tín Dụng", page_icon="💰", layout="centered")

# Load mô hình và bộ xử lý dữ liệu
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model.pkl")

@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor.pkl")

mo_hinh = load_model()
preprocessor = load_preprocessor()

# Giao diện tiêu đề
st.subheader("NCKH: P.Nam, H.Nam, P.Huy, T.Tiến, V.Vinh")
st.title("🔍 Ứng dụng Dự Đoán Rủi Ro Tín Dụng")
st.markdown("### **Nhập thông tin khách hàng để kiểm tra rủi ro tín dụng** 💳")

# Nhập dữ liệu khách hàng
st.subheader("📋 Nhập thông tin khách hàng:")
age = st.slider("📆 Tuổi", 18, 100, 30)
job = st.selectbox("👔 Loại công việc", ["Không có kỹ năng & không cư trú", "Không có kỹ năng & cư trú", "Có kỹ năng", "Rất có kỹ năng"])
job_mapping = {"Không có kỹ năng & không cư trú": 0, "Không có kỹ năng & cư trú": 1, "Có kỹ năng": 2, "Rất có kỹ năng": 3}
job = job_mapping[job]

credit_amount = st.number_input("💵 Khoản vay (DM)", min_value=500, max_value=50000, value=10000, step=100)

duration = st.slider("🕒 Thời hạn vay (tháng)", 6, 72, 24)

sex = st.radio("🚻 Giới tính", ["Nam", "Nữ"], horizontal=True)
sex = 0 if sex == "Nam" else 1

housing = st.selectbox("🏠 Hình thức nhà ở", ["Sở hữu", "Thuê", "Miễn phí"])
housing_mapping = {"Sở hữu": 0, "Thuê": 1, "Miễn phí": 2}
housing = housing_mapping[housing]

saving_accounts = st.selectbox("💰 Tài khoản tiết kiệm", ["Không có", "Ít", "Trung bình", "Khá nhiều", "Nhiều"])
saving_mapping = {"Không có": -1, "Ít": 0, "Trung bình": 1, "Khá nhiều": 2, "Nhiều": 3}
saving_accounts = saving_mapping[saving_accounts]

checking_account = st.number_input("🏦 Số dư tài khoản vãng lai (DM)", min_value=-1.0, max_value=10000.0, value=0.0, step=100.0)

purpose = st.selectbox("🎯 Mục đích vay", ["Mua ô tô", "Mua nội thất/trang thiết bị", "Mua radio/TV", "Mua thiết bị gia dụng", "Sửa chữa", "Giáo dục", "Kinh doanh", "Du lịch/Khác"])
purpose_mapping = {"Mua ô tô": 0, "Mua nội thất/trang thiết bị": 1, "Mua radio/TV": 2, "Mua thiết bị gia dụng": 3,
                   "Sửa chữa": 4, "Giáo dục": 5, "Kinh doanh": 6, "Du lịch/Khác": 7}
purpose = purpose_mapping[purpose]

# Chuyển đổi dữ liệu đầu vào thành DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "Job": job,
    "Credit amount": credit_amount,
    "Duration": duration,
    "Sex": sex,
    "Housing": housing,
    "Saving accounts": saving_accounts,
    "Checking account": checking_account,
    "Purpose": purpose
}])

# Dự đoán kết quả
if st.button("📌 Dự đoán ngay"):
    with st.spinner("⏳ Đang phân tích dữ liệu..."):
        # Tiền xử lý dữ liệu đầu vào
        input_transformed = preprocessor.transform(input_data)
        prediction = mo_hinh.predict_proba(input_transformed)[:, 1]  # Lấy xác suất mắc nợ xấu
        risk_score = prediction[0]

    # Hiển thị kết quả
    st.markdown("---")
    st.subheader("🔹 Kết quả Dự Đoán")
    if risk_score > 0.5:
        st.error(f"⚠️ **Nguy cơ tín dụng xấu: {risk_score:.2%}**")
    else:
        st.success(f"✅ **Khả năng hoàn trả tốt: {risk_score:.2%}**")

    # Biểu đồ trực quan
    st.markdown("#### 📊 Phân tích rủi ro")

    # Biểu đồ gauge
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        title={"text": "Nguy cơ tín dụng xấu (%)"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "red"},
               "steps": [{"range": [0, 50], "color": "grey"},
                         {"range": [50, 100], "color": "grey"}]}
    ))
    st.plotly_chart(fig1, use_container_width=True)

    # Biểu đồ cột
    fig2, ax = plt.subplots()
    ax.bar(["Tốt", "Rủi ro"], [1 - risk_score, risk_score], color=["green", "red"])
    ax.set_ylabel("Xác suất")
    ax.set_title("So sánh mức độ rủi ro tín dụng")
    st.pyplot(fig2)

    # Biểu đồ tròn
    labels = ["Hoàn trả tốt", "Nợ xấu"]
    values = [1 - risk_score, risk_score]
    fig3 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
    fig3.update_traces(marker=dict(colors=["green", "red"]))
    st.plotly_chart(fig3, use_container_width=True)
