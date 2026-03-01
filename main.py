import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# --- Thiết lập trang ---
st.set_page_config(page_title="Mô phỏng các thư viện AI", layout="wide")
st.title("Phân tích Dữ liệu và Trí tuệ Nhân tạo")
st.markdown("---")

# --- Menu Sidebar ---
st.sidebar.title("Chọn Thư viện Mô phỏng")
option = st.sidebar.radio(
    "Danh mục:",
    ("1. Scikit-learn (Học máy truyền thống)", 
     "2. TensorFlow (Mạng Nơ-ron / Deep Learning)", 
     "3. PyTorch (Nghiên cứu & Tối ưu hóa)")
)

# ==========================================
# VÍ DỤ 1: SCIKIT-LEARN
# ==========================================
if option == "1. Scikit-learn (Học máy truyền thống)":
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    st.header("1. Scikit-learn: Học máy truyền thống")
    st.write("**Mục đích:** Xử lý dữ liệu bảng (nhỏ/trung bình), rất dễ hiểu và phù hợp để giảng dạy.")

    # 1. Thu thập và chuẩn bị dữ liệu
    st.subheader("Bước a & b: Dữ liệu đầu vào")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Loại hoa (Target)'] = iris.target
    st.dataframe(df.head(5)) # Hiển thị 5 dòng đầu cho HS xem cấu trúc

    # 2 & 3. Huấn luyện mô hình
    st.subheader("Bước c: Xây dựng Cây quyết định (Decision Tree)")
    model = DecisionTreeClassifier()
    model.fit(iris.data, iris.target)
    st.success("Mô hình đã học xong từ bảng dữ liệu trên cực kỳ nhanh chóng!")

    # Trực quan dự đoán
    st.write("**Giáo viên demo dự đoán:** (Hãy thử kéo các thanh trượt)")
    col1, col2 = st.columns(2)
    with col1:
        sepal_l = st.slider("Chiều dài đài hoa", 4.0, 8.0, 5.0)
        sepal_w = st.slider("Chiều rộng đài hoa", 2.0, 4.5, 3.0)
    with col2:
        petal_l = st.slider("Chiều dài cánh hoa", 1.0, 7.0, 1.5)
        petal_w = st.slider("Chiều rộng cánh hoa", 0.1, 2.5, 0.2)

    pred = model.predict([[sepal_l, sepal_w, petal_l, petal_w]])
    st.info(f"👉 AI dự đoán đây là hoa: **{iris.target_names[pred[0]].upper()}**")

# ==========================================
# VÍ DỤ 2: TENSORFLOW
# ==========================================
elif option == "2. TensorFlow (Mạng Nơ-ron / Deep Learning)":
    # Sử dụng try-except để tránh lỗi cài đặt nặng trên Replit
    try:
        import tensorflow as tf

        st.header("2. TensorFlow: Deep Learning cho Ứng dụng công nghiệp")
        st.write("**Mục đích:** Xây dựng các lớp nơ-ron phức tạp để giải quyết bài toán phi tuyến tính.")

        # Tạo dữ liệu phi tuyến (Đường cong)
        X = np.linspace(-1, 1, 100)
        y = X**3 + np.random.normal(0, 0.05, 100) # Dữ liệu thực tế dạng chữ S

        # Cấu trúc mô hình
        st.code("""
        # Kiến trúc mô hình mạng nơ-ron sâu (Deep Neural Network)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'), # Lớp ẩn 1
            tf.keras.layers.Dense(16, activation='relu'), # Lớp ẩn 2
            tf.keras.layers.Dense(1)                      # Lớp đầu ra
        ])
        """, language='python')

        if st.button("Bắt đầu huấn luyện mô hình TensorFlow"):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            with st.spinner("TensorFlow đang tính toán qua nhiều lớp nơ-ron..."):
                model.fit(X, y, epochs=150, verbose=0) # Mô phỏng học sâu
                predictions = model.predict(X)

                fig, ax = plt.subplots()
                ax.scatter(X, y, label="Dữ liệu thực tế (Phức tạp)", color='gray', alpha=0.5)
                ax.plot(X, predictions, color='red', linewidth=3, label="Đường dự đoán của TensorFlow")
                ax.legend()
                st.pyplot(fig)
                st.success("Học sinh có thể thấy AI đã tự động uốn cong đường thẳng để khớp với dữ liệu khó!")

    except ImportError:
        st.error("Chưa cài đặt TensorFlow. Trong Replit, hãy chạy lệnh: pip install tensorflow-cpu")

# ==========================================
# VÍ DỤ 3: PYTORCH
# ==========================================
elif option == "3. PyTorch (Nghiên cứu & Tối ưu hóa)":
    try:
        import torch
        import torch.nn as nn

        st.header("3. PyTorch: Cú pháp linh hoạt trong Nghiên cứu")
        st.write("**Mục đích:** Cho phép lập trình viên can thiệp vào từng bước học (epoch) để xem bên trong AI diễn ra quá trình gì.")

        # Dữ liệu giả lập
        x_tensor = torch.randn(100, 1)
        y_tensor = x_tensor * 2.5 + 1.0 + torch.randn(100, 1) * 0.5

        model_pt = nn.Linear(1, 1) # Hàm tuyến tính đơn giản
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model_pt.parameters(), lr=0.1)

        st.write("Nhấn nút dưới đây để xem quá trình AI tự động sửa sai (giảm Loss) qua từng chu kỳ học:")
        if st.button("Mô phỏng quá trình huấn luyện từng bước (Epochs)"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            chart = st.line_chart([]) # Biểu đồ trống để vẽ live

            for epoch in range(50):
                optimizer.zero_grad()
                y_pred = model_pt(x_tensor)
                loss = criterion(y_pred, y_tensor)
                loss.backward()
                optimizer.step()

                # Cập nhật giao diện Streamlit trực tiếp
                chart.add_rows([loss.item()])
                progress_bar.progress((epoch + 1) * 2)
                status_text.text(f"Đang học chu kỳ {epoch+1}/50 | Mức độ sai số (Loss): {loss.item():.4f}")
                time.sleep(0.05) # Làm chậm lại để HS kịp quan sát

            st.success("Hoàn thành! Biểu đồ dốc xuống minh họa rõ nét việc AI đang 'học' từ lỗi sai của chính nó.")

    except ImportError:
        st.error("Chưa cài đặt PyTorch. Trong Replit, hãy chạy lệnh: pip install torch")
