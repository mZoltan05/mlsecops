import time

import pandas as pd
import psutil
import streamlit as st

st.set_page_config(page_title="System Monitor", layout="wide")
st.title("Real-Time System Monitor")

update_interval = st.sidebar.slider("Update interval (seconds)", 1, 10, 2)
cpu_threshold = st.sidebar.slider("CPU alert threshold (%)", 0, 100, 80)
mem_threshold = st.sidebar.slider("Memory alert threshold (%)", 0, 100, 80)

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Time", "CPU", "Memory"])

placeholder = st.empty()

# Live Updating Loop
while True:
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    now = time.strftime("%H:%M:%S")

    new_row = pd.DataFrame([[now, cpu, mem]], columns=["Time", "CPU", "Memory"])
    st.session_state.data = pd.concat([st.session_state.data, new_row]).tail(50)  # keep last 50 samples

    with placeholder.container():
        # Alerts
        if cpu > cpu_threshold:
            st.error(f"⚠️ High CPU usage: {cpu}%")
        elif mem > mem_threshold:
            st.warning(f"⚠️ High Memory usage: {mem}%")
        else:
            st.success("✅ System running normally")

        # Charts
        st.line_chart(st.session_state.data.set_index("Time")[["CPU", "Memory"]])

    time.sleep(update_interval)
