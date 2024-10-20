import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from predict import predict

PRODUCTS = ["\"Product 1\""]
DISTRIBUTORS = ["\"Distributor X\"", "\"Distributor Y\"", "\"Distributor Z\""]

def generate_random_df(product, distributor):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    quantity = np.random.randint(50, 500, len(dates)) * (PRODUCTS.index(product) + 1) * (DISTRIBUTORS.index(distributor) + 1)
    return pd.DataFrame({
        'time': dates,
        'quantity': quantity
    })

def generate_predictions(product, distributor):
    np.random.seed(42)
    preds = predict(distributor)
    today = datetime.now()
    one_year_later = today + datetime.timedelta(days=365)
    dates = pd.date_range(start=datetime.now, end=one_year_later, freq='D')
    return pd.DataFrame({
        'time': dates,
        'quantity': preds
    })

def aggregate_data(df, aggregation):
    if aggregation == "Weekly":
        aggregated_data = df.resample('W', on='time').sum()
        week_labels = [f'W{q+1}' for q in range(len(aggregated_data))]
        aggregated_data.index = week_labels
        label_interval = 2
    elif aggregation == "Monthly":
        aggregated_data = df.resample('ME', on='time').sum()
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        aggregated_data.index = month_names[:len(aggregated_data)]
        label_interval = 1
    elif aggregation == "Quarterly":
        aggregated_data = df.resample('QE', on='time').sum()
        x_labels = ['I', 'II', 'III', 'IV']
        aggregated_data.index = [f'Q{x_labels[q]}' for q in range(len(aggregated_data))]
        label_interval = 1
    return aggregated_data, label_interval

def plot_bar_chart(aggregated_data, label_interval, product, distributor):
    st.subheader(f"[Bar Chart] Demand Forecast for {product} from {distributor} for the Next Year")
    plt.figure(figsize=(10, 6))
    plt.bar(aggregated_data.index, aggregated_data['quantity'])
    plt.xticks(np.arange(0, len(aggregated_data), label_interval),
               aggregated_data.index[::label_interval], rotation=45, ha='right')
    plt.ylabel('Quantity')
    plt.title(f'Demand Forecast for {product} from {distributor}')
    st.pyplot(plt)
    plt.close()

def plot_line_chart(aggregated_data, label_interval, product, distributor):
    st.subheader(f"[Line Chart] Demand Forecast for {product} from {distributor} for the Next Year")
    plt.figure(figsize=(10, 6))
    plt.plot(aggregated_data.index, aggregated_data['quantity'], marker='o', linestyle='-')
    plt.xticks(np.arange(0, len(aggregated_data), label_interval),
               aggregated_data.index[::label_interval], rotation=45, ha='right')
    plt.ylabel('Quantity')
    plt.title(f'Demand Trend for {product} from {distributor}')
    st.pyplot(plt)
    plt.close()

def set_background_gradient():
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            background-attachment: fixed;
            background-size: cover;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    set_background_gradient()
    st.title(f"Demand Forecast module")
    selected_product = st.selectbox("Choose a product", PRODUCTS)
    selected_distributor = st.selectbox("Choose a distributor", DISTRIBUTORS)
    st.title(f"Demand Forecast for {selected_product} from {selected_distributor} for the Next Year")
    df = generate_predictions(selected_product, selected_distributor)
    aggregation = st.selectbox("Choose aggregation method", ["Weekly", "Monthly", "Quarterly"])
    aggregated_data, label_interval = aggregate_data(df, aggregation)
    plot_bar_chart(aggregated_data, label_interval, selected_product, selected_distributor)
    plot_line_chart(aggregated_data, label_interval, selected_product, selected_distributor)
    st.subheader("Download Prediction as JSON")
    json_data = aggregated_data.to_json(orient='records')
    st.download_button(
        label=f"Download {selected_product}'s prediction data from {selected_distributor} as JSON",
        data=json_data,
        file_name="prediction.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
