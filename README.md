# 🏛️ Fintech Risk Command Center: Enterprise Fraud Detection Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)
![KEDGE](https://img.shields.io/badge/Academic-KEDGE%20Business%20School-navy.svg)

## 📌 Project Overview
In high-frequency fintech environments, "Standard Accuracy" is a liability. A model that is 99.8% accurate but misses the 0.2% of fraudulent transactions represents a catastrophic failure. 

This project implements a **Real-Time Risk Decisioning Engine** designed to solve the "Imbalance Trap" in banking data. Developed as part of my focus on AI implementation in Fintech, this dashboard provides a production-grade interface for risk analysts to monitor, audit, and explain automated decisions in real-time.

### 🖥️ Dashboard Preview
![Main Dashboard](image_1.png)

---

## 🚀 Key Features & Enterprise Value

### 1. High-Performance Fraud Mitigation
* **Engine:** Optimized **XGBoost** classifier utilizing `scale_pos_weight` to handle extreme class imbalance.
* **Business Benefit:** Maximizes "Recall" to catch rare fraud events, directly protecting the institution's bottom line.

### 2. Explainable AI (XAI) for Regulatory Compliance
* **Risk Reasoning:** Every blocked transaction is automatically tagged with a human-readable "Primary Risk Factor" (e.g., Velocity Spike, Merchant Risk, or Limit Breach).
* **Business Benefit:** Meets **GDPR/PSD2 requirements** for transparency in automated decision-making.

### 3. Real-Time Threat Intelligence
* **Dynamic Visuals:** Integrated **Time-Series Analysis** and **Pattern Distribution** charts track coordinated attack vectors as they occur.

![Threat Pattern Analysis](image_2.png)

### 4. Immutable Audit Trail
* **Complete Archive:** A persistent, session-wide log of every transaction with business-friendly "Yes/No" fraud flagging.

![Complete Transaction Archive](image_3.png)

---

## 🛠️ Technical Architecture

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Model** | XGBoost | High-precision classification of imbalanced transaction data. |
| **Frontend** | Streamlit | High-speed dashboarding with localized UI state updates via Fragments. |
| **Visualization** | Altair | Interactive time-series tracking and pattern distribution charts. |
| **Inference** | Vectorized Predictor | Achieving < 10ms latency for real-time payment processing. |

---

## 💻 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/vighnesh-shetty/fintech-fraud-sentinel.git](https://github.com/vighnesh-shetty/fintech-fraud-sentinel.git)
   cd fintech-fraud-sentinel
   ```
2. **Setup Environment:**
    ```bash
   pip install -r requirements.txt
   ```
3. **Launch the Engine:**
    ```bash
   streamlit run app.py
   ```
**🎓 About the Developer**
I am Vighnesh Shetty, currently pursuing an MSc in Data Analytics for Business at KEDGE Business School (Bordeaux, France). 
With a background in Computer Engineering, I specialize in bridging the gap between advanced machine learning (PyTorch, XGBoost) 
and commercial business value in the Fintech sector.
