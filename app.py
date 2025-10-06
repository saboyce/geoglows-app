#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 19:15:47 2025

@author: sboyce
"""

# Streamlit Web App for GeoGlows: Forecast, Historical & Flow Duration Curve
# pip install streamlit geoglows pandas plotly scipy numpy

import streamlit as st
import geoglows.data as gdata
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np  # Added for arange in FDC
from datetime import datetime
from scipy.interpolate import interp1d
from io import StringIO

# Hardcoded rivers table with real GeoGlows IDs
rivers_df = pd.DataFrame({
    'country': ['Trinidad and Tobago', 'Jamaica', 'Jamaica', 'Jamaica', 'Belize', 'Belize', 'Belize', 'Belize'],
    'location': ['Coroni_River@Piarco', 'Black_River@Crane_Road', 'Rio_Minho@BrokenBank', 'Rio_Cobre@Bogwalk', 'South_Stann_Creek@Highway', 'Belize_River@Belize_City', 'New_River@Corozal', 'Rio_Hondo@Chetumal'],
    'id': [610171862, 780056741, 780006059, 780062115, 770244113, 770335606, 770332458, 770299173]
})

# App title
st.title("GeoGlows Streamflow Viewer: Forecast, Historical & Flow Duration Curve")
st.write("Select a river via filters or enter ID manually. Enhanced interactivity: toggles, hover details.")

# Sidebar for inputs
st.sidebar.header("River Selection")
use_manual = st.sidebar.checkbox("Use Manual River ID", value=False)
if use_manual:
    river_id = st.sidebar.number_input("River ID", value=760021611, step=1, format="%d")
else:
    # Ordered filter: Country -> Location -> ID
    countries = sorted(rivers_df['country'].unique())
    selected_country = st.sidebar.selectbox("Select Country", options=countries)
    filtered_df = rivers_df[rivers_df['country'] == selected_country]
    locations = sorted(filtered_df['location'].unique())
    selected_location = st.sidebar.selectbox("Select Location", options=locations)
    selected_river = filtered_df[filtered_df['location'] == selected_location]
    if not selected_river.empty:
        river_id = selected_river['id'].iloc[0]
        st.sidebar.info(f"Selected River ID: {river_id}")
    else:
        river_id = 760021611  # Fallback
        st.sidebar.warning("No river selected, using default ID.")

data_type = st.sidebar.selectbox("Data Type", ["Forecast", "Historical", "Flow Duration Curve"])
if data_type == "Forecast":
    show_uncertainty_global = st.sidebar.checkbox("Show Uncertainty Bounds", value=True)
if data_type == "Historical":
    resolution = st.sidebar.selectbox("Resolution", ["hourly", "daily", "monthly", "yearly"])
    start_date = st.sidebar.date_input("Start Date", value=datetime(2000, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime(2025, 10, 6))  # Updated to current date
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = min(end_date, datetime.now().date()).strftime("%Y-%m-%d")  # Clamp to now
else:  # Flow Duration Curve
    pass  # No additional sidebar options

# Fetch button
if st.button("Fetch Data"):
    with st.spinner("Fetching data..."):
        rp_df = None
        max_flow = None
        original_df = None
        df = None
        x_col = None
        y_col = None
        title = None
        rp_overlay = False
        forecast_df = None
        forecast_y_col = None
        forecast_show_uncertainty = False
        forecast_max = None
        
        if data_type == "Forecast":
            # Fetch forecast for plain plot
            try:
                forecast_raw = gdata.forecast(river_id=river_id, data_source='rest')
                if isinstance(forecast_raw, str):
                    forecast_df = pd.read_csv(StringIO(forecast_raw), parse_dates=['date'])
                else:
                    forecast_df = forecast_raw
                forecast_df = forecast_df.reset_index(drop=False)
                if 'date' not in forecast_df.columns:
                    date_col_name = forecast_df.columns[0]
                    forecast_df = forecast_df.rename(columns={date_col_name: 'date'})
                possible_y_cols = ['flow_median', 'flow']
                forecast_y_col = next((col for col in possible_y_cols if col in forecast_df.columns), None)
                if forecast_y_col is None:
                    st.error("No suitable flow column found in forecast. Available columns: " + str(forecast_df.columns.tolist()))
                    st.stop()
                forecast_show_uncertainty = 'flow_uncertainty_upper' in forecast_df.columns
                forecast_max = forecast_df[forecast_y_col].max()
                st.info(f"Forecast columns: {forecast_df.columns.tolist()}")  # Debug
            except Exception as e:
                st.error(f"Forecast API fetch failed: {e}")
                st.stop()
            
            # For second plot: with RP
            df = forecast_df.copy()
            y_col = forecast_y_col
            x_col = 'date'
            title = f"Ensemble Median Forecast with Return Period Thresholds (River ID: {river_id})"
            rp_overlay = True
            max_flow = forecast_max
        elif data_type == "Historical":
            kwargs = {'resolution': resolution}
            if 'start_str' in locals():
                kwargs['start'] = start_str
                kwargs['end'] = end_str
            try:
                df_raw = gdata.retrospective(river_id=river_id, **kwargs)
                if isinstance(df_raw, str):
                    df = pd.read_csv(StringIO(df_raw), index_col=0, parse_dates=True)
                else:
                    df = df_raw
                df = df.reset_index(drop=False)
                date_col_name = df.columns[0]
                df = df.rename(columns={date_col_name: 'date'})
                df = df.rename(columns={river_id: 'flow'})
                y_col = 'flow'
                x_col = 'date'
                title = f"Historical Simulation with Return Period Thresholds (River ID: {river_id}, {resolution})"
                rp_overlay = True
                max_flow = df[y_col].max()
                original_df = df.copy()
            except Exception as e:
                st.error(f"Historical API fetch failed: {e}")
                st.stop()
        else:  # Flow Duration Curve
            try:
                df_raw = gdata.retrospective(river_id=river_id, resolution='daily')
                if isinstance(df_raw, str):
                    original_df = pd.read_csv(StringIO(df_raw), index_col=0, parse_dates=True)
                else:
                    original_df = df_raw
                original_df = original_df.reset_index(drop=False)
                date_col_name = original_df.columns[0]
                original_df = original_df.rename(columns={date_col_name: 'date'})
                original_df = original_df.rename(columns={river_id: 'flow'})
                flows_sorted = original_df['flow'].sort_values(ascending=False).reset_index(drop=True)
                n = len(flows_sorted)
                exceedance = (np.arange(1, n+1) / n) * 100
                df = pd.DataFrame({'exceedance': exceedance, 'flow': flows_sorted})
                y_col = 'flow'
                x_col = 'exceedance'
                title = f"Flow Duration Curve with Return Period Thresholds (River ID: {river_id}, Daily)"
                rp_overlay = True
                max_flow = original_df['flow'].max()
            except Exception as e:
                st.error(f"FDC API fetch failed: {e}")
                st.stop()
    
    # Conditional: Show forecast only if data_type == "Forecast"
    if data_type == "Forecast" and not forecast_df.empty:
        # First Plot: Forecast without RP thresholds, enhanced interactivity (no slider)
        st.subheader("Latest Streamflow Forecast (No Thresholds)")
        
        forecast_fig = px.line(forecast_df, x='date', y=forecast_y_col, 
                               title=f"Ensemble Median Forecast (River ID: {river_id})",
                               labels={forecast_y_col: 'Flow (m³/s)', 'date': 'Date/Time'})
        forecast_fig.update_traces(line_color='blue', hovertemplate=f'<b>Date:</b> %{{x}}<br><b>Flow:</b> %{{y:.2f}} m³/s<extra></extra>')
        
        if show_uncertainty_global and forecast_show_uncertainty:
            forecast_fig.add_traces([
                go.Scatter(x=forecast_df['date'], y=forecast_df['flow_uncertainty_upper'], 
                          fill=None, mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'), showlegend=False, name='Upper'),
                go.Scatter(x=forecast_df['date'], y=forecast_df['flow_uncertainty_lower'], 
                          fill='tonexty', fillcolor='rgba(0,100,80,0.2)', 
                          line=dict(width=0, color='rgba(0,0,0,0)'), name='Uncertainty', showlegend=True)
            ])
        if show_uncertainty_global and not forecast_show_uncertainty:
            st.info("No uncertainty data available.")
        
        # Add buttons for reset zoom
        forecast_fig.update_layout(
            hovermode='x unified'  # Enhanced hover
        )
        
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Forecast Stats
        st.subheader("Forecast Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Flow", f"{forecast_df[forecast_y_col].mean():.2f} m³/s")
        with col2:
            st.metric("Max Flow", f"{forecast_max:.2f} m³/s")
        with col3:
            st.metric("Min Flow", f"{forecast_df[forecast_y_col].min():.2f} m³/s")
        with col4:
            st.metric("Data Points", f"{len(forecast_df)}")
        
        # Download Forecast CSV
        forecast_csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast CSV",
            data=forecast_csv,
            file_name=f'forecast_river_{river_id}.csv',
            mime='text/csv'
        )
    
    # Always show the selected data_type plot
    if df is not None and not df.empty:
        # Fetch RP for overlays and metrics
        try:
            rp_df_raw = gdata.return_periods(river_id=river_id)
            if isinstance(rp_df_raw, str):
                rp_df_raw = pd.read_csv(StringIO(rp_df_raw))
            if not rp_df_raw.empty:
                rp_df = rp_df_raw.reset_index(names=['return_period'])
                rp_df = rp_df.rename(columns={river_id: 'flow'})
            else:
                st.warning("No return period data available for overlays.")
                rp_df = pd.DataFrame()
        except Exception as e:
            st.warning(f"Could not fetch return periods: {e}")
            rp_df = pd.DataFrame()
        
        # Debug: Show columns
        st.info(f"Data columns: {list(df.columns)}")
        
        # Display raw data - remove uncertainty columns if Forecast and toggle off
        display_df = df.copy()
        if data_type == "Forecast" and not show_uncertainty_global and 'flow_uncertainty_upper' in display_df.columns:
            display_df = display_df.drop(columns=['flow_uncertainty_upper', 'flow_uncertainty_lower'])
        
        # Download CSV - use full df for download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f'{data_type.lower().replace(" ", "_")}_river_{river_id}.csv',
            mime='text/csv'
        )
        
        # Plot with enhanced interactivity (no slider)
        st.subheader("Analysis Plot")
        if x_col == 'exceedance':
            fig = px.line(df, x=x_col, y=y_col, log_y=False, title=title,
                          labels={x_col: 'Exceedance Probability (%)', y_col: 'Flow (m³/s)'})
            fig.update_traces(line_color='blue', hovertemplate=f'<b>Exceedance:</b> %{{x:.2f}}%<br><b>Flow:</b> %{{y:.2f}} m³/s<extra></extra>')
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title,
                          labels={y_col: 'Flow (m³/s)', x_col: 'Date/Time'})
            fig.update_traces(line_color='blue', hovertemplate=f'<b>Date:</b> %{{x}}<br><b>Flow:</b> %{{y:.2f}} m³/s<extra></extra>')
        
        # Add uncertainty if applicable and toggled on
        if data_type == "Forecast" and show_uncertainty_global and x_col == 'date':
            fig.add_traces([
                go.Scatter(x=df['date'], y=df['flow_uncertainty_upper'], 
                          fill=None, mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'), showlegend=False, name='Upper'),
                go.Scatter(x=df['date'], y=df['flow_uncertainty_lower'], 
                          fill='tonexty', fillcolor='rgba(0,100,80,0.2)', 
                          line=dict(width=0, color='rgba(0,0,0,0)'), name='Uncertainty', showlegend=True)
            ])
        
        # Overlay return period thresholds as shaded regions (for plot if rp_overlay)
        if rp_overlay and not rp_df.empty:
            key_rps = [2, 5, 10, 25, 50, 100]  # Key return periods
            colors = ['rgba(255, 165, 0, 0.3)', 'rgba(255, 215, 0, 0.3)', 'rgba(255, 0, 0, 0.3)', 'rgba(255, 0, 255, 0.3)', 'rgba(128, 0, 128, 0.3)', 'rgba(165, 42, 42, 0.3)']
            rp_flows = {rp: rp_df[rp_df['return_period'] == rp]['flow'].values[0] for rp in key_rps if rp in rp_df['return_period'].values}
            sorted_rps = sorted(rp_flows.keys())
            sorted_flows = [rp_flows[rp] for rp in sorted_rps]
            
            # Get full x-range for spanning width
            if x_col == 'date':
                x_min, x_max = df['date'].min(), df['date'].max()
            else:  # exceedance
                x_min, x_max = df[x_col].min(), df[x_col].max()
            x_span = [x_min, x_max]
            
            # Add shaded regions between thresholds
            for i in range(len(sorted_flows) - 1):
                lower_flow = sorted_flows[i]
                upper_flow = sorted_flows[i + 1]
                band_name = f"{sorted_rps[i]}-{sorted_rps[i+1]} yr Flood Zone"
                fig.add_traces([
                    go.Scatter(
                        x=x_span,
                        y=[upper_flow, upper_flow],
                        mode='lines',
                        line=dict(width=0, color='rgba(0,0,0,0)'),
                        showlegend=False,
                        name='Upper'
                    ),
                    go.Scatter(
                        x=x_span,
                        y=[lower_flow, lower_flow],
                        fill='tonexty',
                        fillcolor=colors[i],
                        line=dict(width=0, color='rgba(0,0,0,0)'),
                        mode='lines',
                        name=band_name,
                        showlegend=True
                    )
                ])
            
            # Add the highest band (above 100-yr) - label on right
            highest_flow = sorted_flows[-1]
            fig.add_hline(y=highest_flow, line_dash="dash", line_color="brown")
            fig.add_annotation(
                x=1, y=highest_flow, xref="paper", yref="y",
                text=f">100-yr: {highest_flow:.0f} m³/s",
                showarrow=False, xanchor="right",
                bgcolor="brown", bordercolor="brown", borderwidth=1
            )
            
            # For historical or FDC: Add max flow estimation - label on left
            if data_type in ["Historical", "Flow Duration Curve"] and max_flow is not None:
                interp_func = interp1d(rp_df['flow'], rp_df['return_period'], bounds_error=False, fill_value=(1, 1000))
                est_rp = interp_func(max_flow)
                fig.add_hline(y=max_flow, line_dash="dot", line_color="green")
                if x_col == 'date':
                    fig.add_annotation(
                        x=0, y=max_flow, xref="paper", yref="y",
                        text=f"Max: {max_flow:.0f} m³/s (~{est_rp:.1f}-yr)",
                        showarrow=False, xanchor="left",
                        bgcolor="green", bordercolor="darkgreen", borderwidth=1
                    )
                else:  # For FDC, add annotation at curve position for max
                    n_days = len(original_df)
                    exceedance_for_max = 100 / n_days
                    fig.add_annotation(
                        x=exceedance_for_max, y=max_flow,
                        text=f"Observed Max<br>{max_flow:.0f} m³/s<br>(~{est_rp:.1f}-yr RP)",
                        showarrow=True, arrowhead=2, ax=20, ay=-30,
                        bgcolor="green", bordercolor="darkgreen", borderwidth=1
                    )
                st.info(f"Max flow: {max_flow:.2f} m³/s (estimated ~{est_rp:.1f}-year return period)")
        
        # Add low flow threshold (Q10) for Historical daily/monthly/yearly
        if data_type == "Historical" and resolution in ['daily', 'monthly', 'yearly']:
            low_flow = df[y_col].quantile(0.10)
            fig.add_hline(y=low_flow, line_dash="dash", line_color="blue")
            fig.add_annotation(
                x=0.5, y=low_flow, xref="paper", yref="y",
                text=f"Q10 (Lowest 10%): {low_flow:.0f} m³/s",
                showarrow=False, xanchor="center",
                bgcolor="lightblue", bordercolor="blue", borderwidth=1
            )
            st.info(f"Q10 low flow threshold: {low_flow:.2f} m³/s (exceeded 90% of the time)")
        
        # Add reset button for plot if time-series
        if x_col == 'date':
            fig.update_layout(
                hovermode='x unified'  # Enhanced hover
            )
        else:
            fig.update_layout(hovermode='closest')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats summary for plot
        st.subheader("Summary Statistics")
        if x_col == 'exceedance':  # FDC indices
            q95 = original_df['flow'].quantile(0.05)  # Low flow, exceeded 95% time
            q50 = original_df['flow'].quantile(0.50)
            q05 = original_df['flow'].quantile(0.95)  # High flow, exceeded 5% time
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Q95 (Low Flow)", f"{q95:.2f} m³/s")
            with col2:
                st.metric("Q50 (Median)", f"{q50:.2f} m³/s")
            with col3:
                st.metric("Q05 (High Flow)", f"{q05:.2f} m³/s")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Flow", f"{df[y_col].mean():.2f} m³/s")
            with col2:
                st.metric("Max Flow", f"{max_flow:.2f} m³/s")
            with col3:
                st.metric("Min Flow", f"{df[y_col].min():.2f} m³/s")
            with col4:
                st.metric("Data Points", f"{len(df)}")
        
        # Return Period Thresholds (always show)
        if not rp_df.empty:
            st.subheader("Key Return Period Thresholds")
            # Updated to show more RPs in a table for better display
            rp_metrics = rp_df[rp_df['return_period'].isin([2, 5, 10, 25, 50, 100])][['return_period', 'flow']].sort_values('return_period')
            rp_metrics.columns = ['Return Period (years)', 'Threshold Flow (m³/s)']
            st.dataframe(rp_metrics)  # Allows sorting
            
            # Download RP CSV
            rp_csv = rp_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Return Periods CSV",
                data=rp_csv,
                file_name=f'return_periods_river_{river_id}.csv',
                mime='text/csv'
            )
    else:
        st.error("No data retrieved. Check the river ID and try again.")

# Footer
st.write("Data from GeoGlows REST API. Updated daily with 15-day forecasts in 3-hour intervals.")