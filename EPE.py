import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# Set page config
st.set_page_config(page_title="Employee Portfolio Evaluation", layout="centered")

# Helper functions
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

def calculate_metrics(data):
    last_price = data['Close'].iloc[-1]
    min_price = data['Low'].min()
    max_price = data['High'].max()
    
    # Calcular rendimientos logarítmicos diarios
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Eliminar posibles valores NaN resultantes del cálculo
    log_returns = data['Log_Returns'].dropna()
    
    # Calcular el rendimiento medio diario logarítmico
    mean_daily_log_return = log_returns.mean()
    
    # Calcular la volatilidad diaria (desviación estándar de los rendimientos logarítmicos)
    daily_volatility = log_returns.std()
    
    # Número de días de trading en un año
    T = 252
    
    # Anualizar el rendimiento medio y la volatilidad
    annual_return = mean_daily_log_return * T * 100  # Convertir a porcentaje
    annual_volatility = daily_volatility * np.sqrt(T) * 100  # Convertir a porcentaje
    
    return last_price, min_price, max_price, annual_return, annual_volatility


def calculate_option_value(current_price, strike_price, number_of_options):
    return max(0, current_price - strike_price) * number_of_options

def export_to_csv():
    export_data = pd.DataFrame({
        'ticker': [st.session_state.ticker],
        'shares': [st.session_state.shares],
    })
    for op in st.session_state.option_positions:
        new_row = pd.DataFrame([{
            'ticker': '',
            'shares': '',
            'option_shares': op['shares'],
            'strike_price': op['strike_price'],
            'grant_date': op['grant_date']
        }])
        export_data = pd.concat([export_data, new_row], ignore_index=True)
    
    csv = export_data.to_csv(index=False)
    return csv

def import_from_csv(csv_contents):
    df = pd.read_csv(io.StringIO(csv_contents))
    
    # Calculate the number of option positions
    num_option_positions = len(df) - 1  # Subtract 1 for the header row
    
    # Extract ticker and shares
    ticker = df['ticker'].iloc[0]
    shares = int(df['shares'].iloc[0])
    
    # Extract option positions
    option_positions = []
    for _, row in df.iloc[1:].iterrows():
        option_positions.append({
            'shares': int(row['option_shares']),
            'strike_price': float(row['strike_price']),
            'grant_date': pd.to_datetime(row['grant_date']).date(),
            'is_vested': False,  # Add this line to set a default value for is_vested
            'position_value': 0  # Add this line to set a default value for position_value
        })
    
    return ticker, shares, num_option_positions, option_positions

def clear_imported_data():
    st.session_state.ticker = ""
    st.session_state.shares = 0
    st.session_state.num_option_positions = 1
    st.session_state.option_positions = []

with st.sidebar:
    st.title("User Guide / Guía de Uso")

    # Language selection
    language = st.selectbox("Select Language / Seleccione Idioma", ["English", "Español"])

    if language == "English":
        st.markdown("""
        **Welcome to Employee Portfolio Evaluation**

        This application allows you to evaluate and project the value of your employer's stock and option portfolio. Follow the steps below to utilize all functionalities:

        ### 1. Import/Export Portfolio
        - **Import from CSV:** Upload a CSV file with your portfolio details to automatically fill in the fields.
        - **Export and Download CSV:** Save your current portfolio to a CSV file for future use.
        - **Clear Data:** Reset all fields and entered data.

        ### 2. Enter Company Data
        - **Ticker Symbol:** Enter the company's stock ticker (e.g., `AAPL` for Apple Inc.).
        - **Time Window:** Select the period to analyze the stock's historical data.

        ### 3. Stock Summary
        - View key metrics like current price, minimum/maximum price, annual return, and volatility.
        - Review the stock price chart over the selected time.

        ### 4. Performance Comparison
        - **Comparison ETF/Stock:** Enter a ticker to compare performance (default is `SPY`).
        - Analyze normalized performance between your stock and the comparison asset.

        ### 5. Portfolio Details
        - **Stock Positions:**
            - Enter the number of shares you own.
            - The total value of your stock position is displayed.
        - **Option Positions:**
            - **Number of Option Positions:** Specify how many option grants you have.
            - For each option:
                - **Granted Quantity:** Number of shares granted.
                - **Exercise Price:** Price at which you can buy the shares.
                - **Grant Date:** Date when you received the options.
            - **Years until Full Vesting:** Time until all your options are vested.
            - The status (vested or unvested) and value of each option are calculated.

        ### 6. Portfolio Values
        - Review the current value of your stocks and options.
        - View the total value of your portfolio, both vested and total potential.

        ### 7. Sensitivity Analysis
        - Explore how changes in the stock price affect your portfolio's potential value.
        - The analysis covers adjustments from -100% to +200% of the current price.

        ### 8. Portfolio Projection
        - **Projection Years:** Select the time horizon for the projection (1-10 years).
        - **Simulation Parameters:**
            - torical return and volatility or input your own estimates.
            - **Show Simulated Paths:** Optionally, visualize the simulated price paths.
        - The Monte Carlo simulation estimates your portfolio's future value.
        - Review projected metrics and the projection chart.

        **Note:** This tool is informational and does not substitute professional financial advice. Always consult an expert before making investment decisions.

        **Privacy:** None of the information loaded or used in this app is saved or transmitted.
        """)
    else:
        st.markdown("""
        **Bienvenido a Employee Portfolio Evaluation**

        Esta aplicación le permite evaluar y proyectar el valor de su portafolio de acciones y opciones de su empleador. Siga los pasos a continuación para utilizar todas las funcionalidades:

        ### 1. Importar/Exportar Portafolio
        - **Importar desde CSV:** Cargue un archivo CSV con los detalles de su portafolio para rellenar automáticamente los campos.
        - **Exportar y Descargar CSV:** Guarde su portafolio actual en un archivo CSV para uso futuro.
        - **Limpiar Datos:** Restablezca todos los campos y datos ingresados.

        ### 2. Ingresar Datos de la Compañía
        - **Símbolo Bursátil:** Ingrese el ticker de la empresa (por ejemplo, `AAPL` para Apple Inc.).
        - **Ventana de Tiempo:** Seleccione el período para analizar los datos históricos de la acción.

        ### 3. Resumen de la Acción
        - Vea métricas clave como precio actual, precio mínimo/máximo, rendimiento anual y volatilidad.
        - Revise el gráfico del precio de la acción a lo largo del tiempo seleccionado.

        ### 4. Comparación de Rendimiento
        - **ETF/Acción de Comparación:** Ingrese un ticker para comparar el rendimiento (por defecto es `SPY`).
        - Analice el rendimiento normalizado entre su acción y el activo de comparación.

        ### 5. Detalles de su Portafolio
        - **Posiciones en Acciones:**
            - Ingrese el número de acciones que posee.
            - Se muestra el valor total de su posición en acciones.
        - **Posiciones en Opciones:**
            - **Número de Posiciones de Opción:** Especifique cuántas concesiones de opciones tiene.
            - Para cada opción:
                - **Cantidad Otorgada:** Número de acciones otorgadas.
                - **Precio de Ejercicio:** Precio al que puede comprar las acciones.
                - **Fecha de Concesión:** Fecha en que recibió las opciones.
            - **Años hasta Consolidación Completa:** Tiempo hasta que todas sus opciones estén consolidadas.
            - Se calcula el estado (consolidado o no) y el valor de cada opción.

        ### 6. Valores del Portafolio
        - Revise el valor actual de sus acciones y opciones.
        - Vea el valor total de su portafolio, tanto consolidado como potencial total.

        ### 7. Análisis de Sensibilidad
        - Explore cómo cambios en el precio de la acción afectan el valor potencial de su portafolio.
        - El análisis abarca ajustes desde -100% hasta +200% del precio actual.

        ### 8. Proyección del Portafolio
        - **Años de Proyección:** Seleccione el horizonte temporal para la proyección (1-10 años).
        - **Parámetros de Simulación:**
            - Use el rendimiento y volatilidad históricos o ingrese sus propias estimaciones.
            - **Mostrar Trayectorias Simuladas:** Opcionalmente, visualice las trayectorias simuladas del precio.
        - La simulación Monte Carlo estima el valor futuro de su portafolio.
        - Revise las métricas proyectadas y el gráfico de proyección.

        **Nota:** Esta herramienta es informativa y no sustituye el asesoramiento financiero profesional. Siempre consulte con un experto antes de tomar decisiones de inversión.

        **Privacidad:** Ninguna de la información cargada o utilizada en esta aplicación es guardada o transmitida.
        """)


# Initialize session state
if 'ticker' not in st.session_state:
    st.session_state.ticker = ""
if 'shares' not in st.session_state:
    st.session_state.shares = 0
if 'num_option_positions' not in st.session_state:
    st.session_state.num_option_positions = 1
if 'years_until_full_vesting' not in st.session_state:
    st.session_state.years_until_full_vesting = 3
if 'option_positions' not in st.session_state:
    st.session_state.option_positions = []
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Main app
st.title("Employee Portfolio Evaluation")

# Add CSV import/export buttons
csv_file = st.file_uploader("Import portfolio from CSV", type="csv", key=f"file_csv_{st.session_state.uploader_key}")
if csv_file is not None:
    clear_imported_data()  # Clear existing data before import
    st.session_state.ticker, st.session_state.shares, st.session_state.num_option_positions, st.session_state.option_positions = import_from_csv(csv_file.getvalue().decode())
    st.success("CSV imported successfully!")
    st.session_state.uploader_key += 1  # Reset the uploader by changing its key

if st.button("Clear Data"):
    clear_imported_data()
    st.success("Data cleared successfully!")

if st.button("Export and Download CSV"):
    if st.session_state.ticker:
        csv = export_to_csv()
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="portfolio_export.csv",
            mime="text/csv"
        )
    else:
        st.warning("Please enter a ticker symbol before exporting.")

# User input
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.session_state.ticker = st.text_input("Enter company ticker symbol", value=st.session_state.ticker)
with col2:
    time_window = st.selectbox("Select time window", 
                               options=["1m", "6m", "YTD", "1y", "3y", "5y", "10y", "25y", "Max"], 
                               index=4)

# Calculate date range
end_date = datetime.now()
if time_window == "1m":
    start_date = end_date - timedelta(days=30)
elif time_window == "6m":
    start_date = end_date - timedelta(days=180)
elif time_window == "YTD":
    start_date = datetime(end_date.year, 1, 1)
elif time_window == "1y":
    start_date = end_date - timedelta(days=365)
elif time_window == "3y":
    start_date = end_date - timedelta(days=3*365)
elif time_window == "5y":
    start_date = end_date - timedelta(days=5*365)
elif time_window == "10y":
    start_date = end_date - timedelta(days=10*365)
elif time_window == "25y":
    start_date = end_date - timedelta(days=25*365)
else:  # Max
    start_date = end_date - timedelta(days=200*365)

st.markdown("---")
# Fetch stock data

if st.session_state.ticker:
    # Initialize yfinance Ticker object
    stock = yf.Ticker(st.session_state.ticker)
    
    # Attempt to retrieve the company name
    try:
        company_name = stock.info.get('longName') or stock.info.get('shortName') or 'N/A'
    except Exception as e:
        company_name = 'N/A'
        st.warning(f"Could not retrieve company name for ticker '{st.session_state.ticker}'.")
    
    # Display the company name and ticker symbol as a header
    st.markdown(f"## {company_name} ({st.session_state.ticker.upper()})")
    
    # Fetch historical stock data
    data = fetch_stock_data(st.session_state.ticker, start_date, end_date)

    if data.empty:
        st.warning("No data available for the selected time window.")
    else:
        # Calculate metrics
        last_price, min_price, max_price, annual_return, volatility = calculate_metrics(data)

        st.subheader("Stock Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Last Price", f"${last_price:.2f}")
        col2.metric("Min Price", f"${min_price:.2f}")
        col3.metric("Max Price", f"${max_price:.2f}")
        col4.metric("Annual Return", f"{annual_return:.2f}%")
        col5.metric("Volatility", f"{volatility:.2f}%")

        # Plot Stock Price Over Time
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Stock Price'))
        fig1.update_layout(title=f"{st.session_state.ticker.upper()} Stock Price Over Time", xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig1)

        # 2. Normalized performance comparison
        comparison_etf = st.text_input("Select comparison ETF/Stock", value="SPY")
        etf_data = fetch_stock_data(comparison_etf, start_date, end_date)

        if not etf_data.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data.index, y=data['Close']/data['Close'].iloc[0], mode='lines', name=st.session_state.ticker.upper()))
            fig2.add_trace(go.Scatter(x=etf_data.index, y=etf_data['Close']/etf_data['Close'].iloc[0], mode='lines', name=comparison_etf.upper()))
            fig2.update_layout(title="Normalized Performance Comparison", xaxis_title="Date", yaxis_title="Normalized Price")
            st.plotly_chart(fig2)
        else:
            st.warning(f"No data available for comparison ticker '{comparison_etf}'.")

        # User portfolio inputs
        st.markdown("---")
        st.subheader("Your Employee Portfolio")
        with st.expander("Stock Positions", True):
            st.session_state.shares = st.number_input("Number of shares", min_value=0, value=int(st.session_state.shares))
            st.write(f"**Position Value:** ${st.session_state.shares * last_price:,.0f}")

        # Multiple option positions
        #st.subheader("Option Positions")
        with st.expander("Option Positions", True):
            st.session_state.num_option_positions = st.number_input("Number of option positions", min_value=1, max_value=10, value=int(st.session_state.num_option_positions))
            st.session_state.option_positions = st.session_state.option_positions[:st.session_state.num_option_positions]
            # Single 'Years until full vesting' input for all option positions
            st.session_state.years_until_full_vesting = st.number_input("Years until full vesting for all options", min_value=0, max_value=100, value=int(st.session_state.years_until_full_vesting))

            while len(st.session_state.option_positions) < st.session_state.num_option_positions:
                st.session_state.option_positions.append({
                    'shares': 0,
                    'strike_price': last_price,
                    'grant_date': datetime.now().date(),
                    'is_vested': False,
                    'position_value': 0
                })

            for i in range(st.session_state.num_option_positions):
                st.write(f"**Option Position {i+1}**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.session_state.option_positions[i]['shares'] = st.number_input(
                        f"Granted Quantity {i+1}", 
                        min_value=0, 
                        value=int(st.session_state.option_positions[i]['shares']), 
                        key=f"option_shares_{i}"
                    )
                with col2:
                    st.session_state.option_positions[i]['strike_price'] = st.number_input(
                        f"Exercise Price {i+1}", 
                        min_value=0.0, 
                        value=float(st.session_state.option_positions[i]['strike_price']), 
                        key=f"strike_price_{i}"
                    )
                with col3:
                    st.session_state.option_positions[i]['grant_date'] = st.date_input(
                        f"Grant Date {i+1}", 
                        value=st.session_state.option_positions[i]['grant_date'], 
                        key=f"grant_date_{i}"
                    )
                
                current_date = datetime.now().date()
                is_vested = (current_date - st.session_state.option_positions[i]['grant_date']).days / 365.25 >= st.session_state.years_until_full_vesting
                option_value = calculate_option_value(last_price, st.session_state.option_positions[i]['strike_price'], st.session_state.option_positions[i]['shares'])
                
                st.session_state.option_positions[i]['is_vested'] = is_vested
                st.session_state.option_positions[i]['position_value'] = option_value
                
                with col4:
                    status = "Vested" if is_vested else "Unvested"
                    color = "green" if is_vested else "red"
                    st.markdown(f"<span style='color:{color};'>**Status:** {status}</span>", unsafe_allow_html=True)
                    st.write(f"**Position Value:** ${option_value:,.0f}")

        # **Add Conditional Check Here**
        # Check if the user has entered at least one share or option position
        has_shares = st.session_state.shares > 0
        has_options = any(op['shares'] > 0 for op in st.session_state.option_positions)
        
        if has_shares or has_options:
            # Calculate portfolio value
            stock_value = st.session_state.shares * last_price
            vested_option_value = sum(op['position_value'] for op in st.session_state.option_positions if op.get('is_vested', False))
            total_option_value = sum(op['position_value'] for op in st.session_state.option_positions)

            total_value = stock_value + vested_option_value
            total_potential_value = stock_value + total_option_value

            # Display portfolio values with enhanced visualization
            st.markdown("### Portfolio Values")
            # First row: 3 metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                col1.metric("Current Stock Value", f"${stock_value:,.0f}")
            with col2:
                col2.metric("Vested Option Value", f"${vested_option_value:,.0f}")
            with col3:
                col3.metric("Total Option Value", f"${total_option_value:,.0f}")
            
            # Second row: 2 metrics
            col4, col5 = st.columns(2)
            with col4:
                col4.metric("Total Portfolio (Vested)", f"${total_value:,.0f}")
            with col5:
                col5.metric("Total Potential Portfolio", f"${total_potential_value:,.0f}")

            st.markdown("---")

            # Sensitivity Analysis of Potential Portfolio Value

            # Define the range of stock price adjustments (±100%, etc.)
            adjustments = np.array([-1.0, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])  # -100%, -75%, etc.
            adjustment_labels = [
                "-100%", "-75%", "-50%", "-25%", "-10%", "0%", 
                "+10%", "+25%", "+50%", "+75%", "+100%", "+150%", "+200%"
            ]

            # Calculate the new stock prices based on adjustments
            adjusted_prices = last_price * (1 + adjustments)

            # Recalculate the Potential Portfolio value for each adjusted price
            sensitivity_values = []
            for price in adjusted_prices:
                # Recalculate option values
                potential_option_value = sum([calculate_option_value(price, op['strike_price'], op['shares']) for op in st.session_state.option_positions])
                potential_portfolio = st.session_state.shares * price + potential_option_value
                sensitivity_values.append(potential_portfolio)

            # Create the sensitivity analysis figure
            fig_sensitivity = go.Figure()
            fig_sensitivity.add_trace(go.Scatter(
                x=adjustment_labels,
                y=sensitivity_values,
                mode='lines+markers',
                name='Potential Portfolio Value',
                line=dict(color='blue')
            ))

            # Update layout and fix the X-axis labels
            fig_sensitivity.update_layout(
                title='Sensitivity Analysis of Potential Portfolio Value',
                xaxis_title='Stock Price Adjustment',
                yaxis_title='Potential Portfolio Value ($)',
                xaxis=dict(
                    tickmode='array',
                    tickvals=adjustment_labels,  # Only show these specific points on the X-axis
                    ticktext=adjustment_labels,  # The labels for the X-axis
                ),
                yaxis=dict(tickformat=","),
                height=600
            )

            st.plotly_chart(fig_sensitivity)
            
            # Portfolio projection
            st.subheader("Portfolio Value Projection")
            projection_years = st.slider("Projection years", min_value=1, max_value=10, value=5)

            st.markdown("### Portfolio Projection")
            lower_percentile = 5
            upper_percentile = 95

            # Checkbox to use historical return and volatility or specify new values
            use_historical = st.checkbox("Use historical return and volatility", value=True)

            if use_historical:
                mu = annual_return / 100  # Convert percentage to decimal
                sigma = volatility / 100  # Ensure we use the annualized volatility
                st.text(f"Historical Return: ${annual_return:,.0f}%")
                st.text(f"Historical Volatility: ${volatility:,.0f}%")
            else:
                mu = st.number_input("Expected annual return (%)", value=10.0)
                sigma = st.number_input("Expected annual volatility (%)", value=15.0)
                mu = mu / 100  # Convert to decimal
                sigma = sigma / 100

            # Option to show simulated stock price paths
            show_simulation_paths = st.checkbox("Show all simulated stock price paths", value=False)

            # Simulation parameters
            num_simulations = 10000

            # Define the time step (dt)
            dt = 1/252  # Daily time steps (assuming 252 trading days per year)

            # Calculate the total number of time steps
            steps_per_year = int(1 / dt)  # Should be 252
            total_steps = steps_per_year * projection_years  # Total steps

            # Extract portfolio details
            shares = st.session_state.shares
            option_shares = np.array([op['shares'] for op in st.session_state.option_positions])
            strike_prices = np.array([op['strike_price'] for op in st.session_state.option_positions])

            # Handle cases with no shares or no options
            if shares == 0 and len(option_shares) == 0:
                st.warning("Your portfolio has no shares or option positions to project.")
            else:
                # Perform Monte Carlo simulation
                with st.spinner("Running Monte Carlo simulation..."):
                    np.random.seed(42)  # For reproducibility

                    # Generate random standard normal variables
                    Z = np.random.standard_normal((num_simulations, total_steps))

                    # Calculate the increments using GBM formula
                    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

                    # Initialize log_S array to include initial price
                    log_S = np.zeros((num_simulations, total_steps + 1))
                    log_S[:, 0] = np.log(last_price)

                    # Compute cumulative sum of increments
                    log_S[:, 1:] = log_S[:, 0, np.newaxis] + np.cumsum(increments, axis=1)

                    # Calculate stock price paths
                    stock_price_paths = np.exp(log_S)  # Shape: (num_simulations, total_steps + 1)

                    # If the user selected to show the simulated paths
                    if show_simulation_paths:
                        # Create the figure
                        fig_simulation_paths = go.Figure()
                        
                        # Create a time array in years
                        time_array = np.linspace(0, projection_years, total_steps + 1)
                        
                        # Define the number of paths to plot to avoid overloading the graph
                        num_paths_to_plot = st.slider("Number of simulation paths to display", min_value=10, max_value=500, value=50, step=10)
                        
                        # Add the paths to the figure
                        for i in range(num_paths_to_plot):
                            fig_simulation_paths.add_trace(go.Scatter(
                                x=time_array,
                                y=stock_price_paths[i],
                                mode='lines',
                                line=dict(width=1),
                                opacity=0.5,
                                showlegend=False
                            ))
                        
                        # Update the layout of the plot
                        fig_simulation_paths.update_layout(
                            title='Simulated Stock Price Paths',
                            xaxis_title='Years',
                            yaxis_title='Stock Price ($)',
                            template='plotly_white',
                            height=600
                        )
                        
                        # Display the plot in Streamlit
                        st.plotly_chart(fig_simulation_paths, use_container_width=True)

                    # Continue with the calculation of stock_price_paths_yearly and other code
                    # Define year indices (including time zero)
                    year_indices = np.arange(0, total_steps + 1, steps_per_year)  # Indices at which we sample for each year

                    # Get stock prices at the end of each year (including initial price)
                    stock_price_paths_yearly = stock_price_paths[:, year_indices]  # Shape: (num_simulations, projection_years + 1)

                    # Calculate stock value
                    stock_values = shares * stock_price_paths_yearly  # Shape: (num_simulations, projection_years + 1)

                    # Calculate option values
                    if len(option_shares) > 0:
                        # Expand dimensions for broadcasting
                        stock_price_expanded = stock_price_paths_yearly[:, :, np.newaxis]  # Shape: (num_simulations, projection_years + 1, 1)
                        strike_prices_expanded = strike_prices[np.newaxis, np.newaxis, :]  # Shape: (1, 1, num_options)
                        option_values = np.maximum(stock_price_expanded - strike_prices_expanded, 0) * option_shares  # Shape: (num_simulations, projection_years + 1, num_options)
                        total_option_values = np.sum(option_values, axis=2)  # Shape: (num_simulations, projection_years + 1)
                    else:
                        total_option_values = np.zeros((num_simulations, projection_years + 1))  # Shape: (num_simulations, projection_years + 1)

                    # Calculate total portfolio value
                    total_portfolio_values = stock_values + total_option_values  # Shape: (num_simulations, projection_years + 1)

                    # Calculate median and confidence intervals
                    median_portfolio = np.median(total_portfolio_values, axis=0)
                    lower_portfolio = np.percentile(total_portfolio_values, lower_percentile, axis=0)
                    upper_portfolio = np.percentile(total_portfolio_values, upper_percentile, axis=0)


                    # Identify the simulation indices for median, lower, and upper portfolio values at the final year
                    median_final = median_portfolio[-1]
                    lower_final = lower_portfolio[-1]
                    upper_final = upper_portfolio[-1]

                    # Find the indices of simulations closest to the median, lower, and upper portfolio values
                    median_index = np.abs(total_portfolio_values[:, -1] - median_final).argmin()
                    lower_index = np.abs(total_portfolio_values[:, -1] - lower_final).argmin()
                    upper_index = np.abs(total_portfolio_values[:, -1] - upper_final).argmin()

                    # Extract the corresponding stock values for these simulations
                    median_stock_value = stock_values[median_index, -1]
                    lower_stock_value = stock_values[lower_index, -1]
                    upper_stock_value = stock_values[upper_index, -1]

                # Prepare data for plotting
                years = np.arange(0, projection_years + 1)
                fig_projection = go.Figure()

                # Add median portfolio path
                fig_projection.add_trace(go.Scatter(
                    x=years,
                    y=median_portfolio,
                    mode='lines',
                    name='Median Portfolio Value',
                    line=dict(color='blue', width=2)
                ))

                # Add lower and upper confidence interval paths
                fig_projection.add_trace(go.Scatter(
                    x=years,
                    y=lower_portfolio,
                    mode='lines',
                    name=f'{lower_percentile}th Percentile',
                    line=dict(color='red', width=1, dash='dash')
                ))

                fig_projection.add_trace(go.Scatter(
                    x=years,
                    y=upper_portfolio,
                    mode='lines',
                    name=f'{upper_percentile}th Percentile',
                    line=dict(color='green', width=1, dash='dash')
                ))

                # Update layout
                fig_projection.update_layout(
                    title='Portfolio Value Projection',
                    xaxis_title='Years',
                    yaxis_title='Portfolio Value ($)',
                    legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0)'),
                    template='plotly_white',
                    height=600
                )

                st.plotly_chart(fig_projection, use_container_width=True)

                # Display summary statistics with both portfolio and stock values
                st.markdown("### Projection Summary")
                col_sum1, col_sum2, col_sum3 = st.columns(3)

                with col_sum1:
                    st.metric("Median Portfolio Value", f"${median_final:,.0f}")
                    if shares > 0:
                        median_stock_per_share = median_stock_value / shares if shares != 0 else 0
                        st.markdown(f"**Median Stock Value:** ${median_stock_per_share:,.2f}")
                    else:
                        st.markdown("**Median Stock Value:** N/A")

                with col_sum2:
                    st.metric(f"{lower_percentile}th Percentile Value", f"${lower_final:,.0f}")
                    if shares > 0:
                        lower_stock_per_share = lower_stock_value / shares if shares != 0 else 0
                        st.markdown(f"**Lower Percentile Stock Value:** ${lower_stock_per_share:,.2f}")
                    else:
                        st.markdown("**Lower Percentile Stock Value:** N/A")

                with col_sum3:
                    st.metric(f"{upper_percentile}th Percentile Value", f"${upper_final:,.0f}")
                    if shares > 0:
                        upper_stock_per_share = upper_stock_value / shares if shares != 0 else 0
                        st.markdown(f"**Upper Percentile Stock Value:** ${upper_stock_per_share:,.2f}")
                    else:
                        st.markdown("**Upper Percentile Stock Value:** N/A")
        else:
            st.warning("Please enter at least one share or option position to proceed.")
