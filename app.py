#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 10:36:33 2025

@author: sboyce
"""

# Streamlit Web App for GeoGlows: Forecast, Historical & Flow Duration Curve
# pip install streamlit geoglows pandas plotly scipy numpy

import streamlit as st
import geoglows.data as gdata
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
from io import StringIO

# Hardcoded rivers table with real GeoGlows IDs
rivers_df = pd.DataFrame({
    'country': ['Trinidad and Tobago', 'Jamaica', 'Jamaica', 'Jamaica', 'Belize', 'Belize', 'Belize', 'Belize'],
    'location': ['Coroni_River@Piarco', 'Black_River@Crane_Road', 'Rio_Minho@BrokenBank', 'Rio_Cobre@Bogwalk', 'South_Stann_Creek@Highway', 'Belize_River@Belize_City', 'New_River@Corozal', 'Rio_Hondo@Chetumal'],
    'id': [610171862, 780056741, 780006059, 780062115, 770244113, 770335606, 770332458, 770299173]
})

# Function to calculate stage from discharge using rating curve: stage = a * (discharge ** b) + c
def calculate_stage(discharge, a, b, c):
    try:
        # Handle non-negative discharge to avoid invalid power operations
        discharge = np.maximum(discharge, 0)
        stage = a * (discharge ** b) + c
        return stage
    except Exception as e:
        st.error(f"Error calculating stage: {e}")
        return np.nan

# App title
st.title("GeoGlows Streamflow Viewer: Forecast, Historical & Flow Duration Curve")
st.write("Select a river via filters or enter ID manually. View discharge and stage with rating curve.")

# Sidebar for inputs
st.sidebar.header("River Selection")
use_manual = st.sidebar.checkbox("Use Manual River ID", value=False)
if use_manual:
    river_id = st.sidebar.number_input("River ID", value=760021611, step=1, format="%d")
else:
    countries = sorted(rivers_df['country'].unique())
    selected_country = st.sidebar.selectbox("Select Country", options=countries)
    filtered_df = rivers_df[rivers_df['country'] == selected_country]
    locations = sorted(filtered_df['location'].unique())
    selected_location = st.sidebar.selectbox("Select Location", options=locations)
    selected_river = filtered_df[filtered_df['location'] == selected_location]
    if not selected_river.empty:
        river_id = selected_river['id'].iloc[0]
        river_name = selected_location
        st.sidebar.info(f"Selected River: {river_name} (ID: {river_id})")
    else:
        river_id = 760021611
        river_name = "Mississippi River"
        st.sidebar.warning("No river selected, using default ID.")

# Rating curve coefficients
st.sidebar.header("Rating Curve (Stage = a * Discharge^b + c)")
use_rating_curve = st.sidebar.checkbox("Apply Rating Curve", value=False)
if use_rating_curve:
    a_coeff = st.sidebar.number_input("Coefficient a", value=0.01, format="%.4f")
    b_coeff = st.sidebar.number_input("Exponent b", value=0.5, format="%.4f")
    c_coeff = st.sidebar.number_input("Constant c (m)", value=0.0, format="%.4f")

data_type = st.sidebar.selectbox("Data Type", ["Forecast", "Historical", "Flow Duration Curve"])
if data_type == "Forecast":
    show_uncertainty_global = st.sidebar.checkbox("Show Uncertainty Bounds", value=True)
if data_type == "Historical":
    resolution = st.sidebar.selectbox("Resolution", ["hourly", "daily", "monthly", "yearly"])
    start_date = st.sidebar.date_input("Start Date", value=datetime(2000, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime(2025, 10, 3))
    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

# Fetch button
if st.button("Fetch Data"):
    with st.spinner("Fetching data..."):
        rp_df = None
        max_flow = None
        max_stage = None
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
        
        # Fetch and process data
        if data_type == "Forecast":
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
            if use_rating_curve:
                forecast_df['stage'] = calculate_stage(forecast_df[forecast_y_col], a_coeff, b_coeff, c_coeff)
                forecast_max_stage = forecast_df['stage'].max()
            
            df = forecast_df.copy()
            y_col = forecast_y_col
            x_col = 'date'
            title = f"Ensemble Median Forecast with Return Period Thresholds (River ID: {river_id})"
            rp_overlay = True
            max_flow = forecast_max
            max_stage = forecast_max_stage if use_rating_curve else None
        elif data_type == "Historical":
            kwargs = {'resolution': resolution}
            if 'start_str' in locals():
                kwargs['start'] = start_str
                kwargs['end'] = end_str
            df_raw = gdata.retrospective(river_id=river_id, **kwargs)
            if isinstance(df_raw, str):
                df = pd.read_csv(StringIO(df_raw), index_col=0, parse_dates=True)
            else:
                df = df_raw
            df = df.reset_index(drop=False)
            date_col_name = df.columns[0]
            df = df.rename(columns={date_col_name: 'date'})
            df = df.rename(columns={river_id: 'flow'})
            if use_rating_curve:
                df['stage'] = calculate_stage(df['flow'], a_coeff, b_coeff, c_coeff)
                max_stage = df['stage'].max()
            y_col = 'flow'
            x_col = 'date'
            title = f"Historical Simulation with Return Period Thresholds (River ID: {river_id}, {resolution})"
            rp_overlay = True
            max_flow = df[y_col].max()
            original_df = df.copy()
        else:  # Flow Duration Curve
            df_raw = gdata.retrospective(river_id=river_id, resolution='daily')
            if isinstance(df_raw, str):
                original_df = pd.read_csv(StringIO(df_raw), index_col=0, parse_dates=True)
            else:
                original_df = df_raw
            original_df = original_df.reset_index(drop=False)
            date_col_name = original_df.columns[0]
            original_df = original_df.rename(columns={date_col_name: 'date'})
            original_df = original_df.rename(columns={river_id: 'flow'})
            if use_rating_curve:
                original_df['stage'] = calculate_stage(original_df['flow'], a_coeff, b_coeff, c_coeff)
                max_stage = original_df['stage'].max()
            flows_sorted = original_df['flow'].sort_values(ascending=False).reset_index(drop=True)
            stages_sorted = original_df['stage'].sort_values(ascending=False).reset_index(drop=True) if use_rating_curve else None
            n = len(flows_sorted)
            exceedance = (np.arange(1, n+1) / n) * 100
            df = pd.DataFrame({'exceedance': exceedance, 'flow': flows_sorted})
            if use_rating_curve:
                df['stage'] = stages_sorted
            y_col = 'flow'
            x_col = 'exceedance'
            title = f"Flow Duration Curve with Return Period Thresholds (River ID: {river_id}, Daily)"
            rp_overlay = True
            max_flow = original_df['flow'].max()
    
    # Conditional: Show forecast only if data_type == "Forecast"
    if data_type == "Forecast" and not forecast_df.empty:
        st.subheader("Latest Streamflow Forecast (No Thresholds)")
        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(
            x=forecast_df['date'], y=forecast_df[forecast_y_col],
            mode='lines', name='Flow (m³/s)', line=dict(color='blue'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Flow:</b> %{y:.2f} m³/s<extra></extra>'
        ))
        if use_rating_curve:
            forecast_fig.add_trace(go.Scatter(
                x=forecast_df['date'], y=forecast_df['stage'],
                mode='lines', name='Stage (m)', line=dict(color='orange'),
                yaxis='y2',
                hovertemplate='<b>Date:</b> %{x}<br><b>Stage:</b> %{y:.2f} m<extra></extra>'
            ))
        if show_uncertainty_global and forecast_show_uncertainty:
            forecast_fig.add_traces([
                go.Scatter(x=forecast_df['date'], y=forecast_df['flow_uncertainty_upper'],
                          fill=None, mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'), showlegend=False, name='Upper'),
                go.Scatter(x=forecast_df['date'], y=forecast_df['flow_uncertainty_lower'],
                          fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                          line=dict(width=0, color='rgba(0,0,0,0)'), name='Uncertainty', showlegend=True)
            ])
        
        forecast_fig.update_layout(
            title=f"Ensemble Median Forecast (River ID: {river_id})",
            xaxis=dict(title='Date/Time'),
            yaxis=dict(title='Flow (m³/s)', side='left'),
            yaxis2=dict(title='Stage (m)', overlaying='y', side='right', showgrid=False) if use_rating_curve else None,
            hovermode='x unified'
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
        if use_rating_curve:
            st.subheader("Stage Summary Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Stage", f"{forecast_df['stage'].mean():.2f} m")
            with col2:
                st.metric("Max Stage", f"{forecast_max_stage:.2f} m")
            with col3:
                st.metric("Min Stage", f"{forecast_df['stage'].min():.2f} m")
        
        forecast_csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast CSV",
            data=forecast_csv,
            file_name=f'forecast_river_{river_id}.csv',
            mime='text/csv'
        )
    
    # Main plot for selected data type
    if df is not None and not df.empty:
        # Fetch return periods
        try:
            rp_df_raw = gdata.return_periods(river_id=river_id)
            if isinstance(rp_df_raw, str):
                rp_df_raw = pd.read_csv(StringIO(rp_df_raw))
            if not rp_df_raw.empty:
                rp_df = rp_df_raw.reset_index(names=['return_period'])
                rp_df = rp_df.rename(columns={river_id: 'flow'})
                if use_rating_curve:
                    rp_df['stage'] = calculate_stage(rp_df['flow'], a_coeff, b_coeff, c_coeff)
        except Exception as e:
            st.warning(f"Could not fetch return periods: {e}")
            rp_df = pd.DataFrame()
        
        # Display data table
        st.subheader("Data Table")
        display_df = df.copy()
        if data_type == "Forecast" and not show_uncertainty_global and 'flow_uncertainty_upper' in display_df.columns:
            display_df = display_df.drop(columns=['flow_uncertainty_upper', 'flow_uncertainty_lower'])
        st.dataframe(display_df.head(100))
        
        # Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f'{data_type.lower().replace(" ", "_")}_river_{river_id}.csv',
            mime='text/csv'
        )
        
        # Main plot with dual axis
        st.subheader("Analysis Plot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[y_col],
            mode='lines', name='Flow (m³/s)', line=dict(color='blue'),
            hovertemplate=f'<b>{x_col.capitalize()}:</b> %{{x}}<br><b>Flow:</b> %{{y:.2f}} m³/s<extra></extra>'
        ))
        if use_rating_curve:
            fig.add_trace(go.Scatter(
                x=df[x_col], y=df['stage'],
                mode='lines', name='Stage (m)', line=dict(color='orange'),
                yaxis='y2',
                hovertemplate=f'<b>{x_col.capitalize()}:</b> %{{x}}<br><b>Stage:</b> %{{y:.2f}} m<extra></extra>'
            ))
        
        # Add uncertainty for forecast
        if data_type == "Forecast" and show_uncertainty_global and x_col == 'date':
            fig.add_traces([
                go.Scatter(x=df['date'], y=df['flow_uncertainty_upper'],
                          fill=None, mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'), showlegend=False, name='Upper'),
                go.Scatter(x=df['date'], y=df['flow_uncertainty_lower'],
                          fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                          line=dict(width=0, color='rgba(0,0,0,0)'), name='Uncertainty', showlegend=True)
            ])
        
        # Overlay return period thresholds
        if rp_overlay and not rp_df.empty:
            key_rps = [2, 5, 10, 25, 50, 100]
            colors = ['rgba(255, 165, 0, 0.3)', 'rgba(255, 215, 0, 0.3)', 'rgba(255, 0, 0, 0.3)', 'rgba(255, 0, 255, 0.3)', 'rgba(128, 0, 128, 0.3)', 'rgba(165, 42, 42, 0.3)']
            rp_flows = {rp: rp_df[rp_df['return_period'] == rp]['flow'].values[0] for rp in key_rps if rp in rp_df['return_period'].values}
            sorted_rps = sorted(rp_flows.keys())
            sorted_flows = [rp_flows[rp] for rp in sorted_rps]
            
            x_min, x_max = df[x_col].min(), df[x_col].max()
            x_span = [x_min, x_max]
            
            for i in range(len(sorted_flows) - 1):
                lower_flow = sorted_flows[i]
                upper_flow = sorted_flows[i + 1]
                band_name = f"{sorted_rps[i]}-{sorted_rps[i+1]} yr Flood Zone"
                fig.add_traces([
                    go.Scatter(
                        x=x_span, y=[upper_flow, upper_flow],
                        mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
                        showlegend=False, name='Upper'
                    ),
                    go.Scatter(
                        x=x_span, y=[lower_flow, lower_flow],
                        fill='tonexty', fillcolor=colors[i],
                        line=dict(width=0, color='rgba(0,0,0,0)'),
                        mode='lines', name=band_name, showlegend=True
                    )
                ])
            
            highest_flow = sorted_flows[-1]
            fig.add_hline(y=highest_flow, line_dash="dash", line_color="brown",
                          annotation_text=f">100-yr: {highest_flow:.0f} m³/s")
            
            # Add stage thresholds if rating curve is used
            if use_rating_curve:
                rp_stages = {rp: rp_df[rp_df['return_period'] == rp]['stage'].values[0] for rp in sorted_rps if rp in rp_df['return_period'].values}
                sorted_stages = [rp_stages[rp] for rp in sorted_rps]
                highest_stage = sorted_stages[-1]
                fig.add_hline(y=highest_stage, line_dash="dash", line_color="purple",
                              annotation_text=f">100-yr: {highest_stage:.2f} m", yref='y2')
        
        # Update layout for dual axis
        fig.update_layout(
            title=title,
            xaxis=dict(title='Exceedance Probability (%)' if x_col == 'exceedance' else 'Date/Time'),
            yaxis=dict(title='Flow (m³/s)', side='left', type='log' if x_col == 'exceedance' else 'linear'),
            yaxis2=dict(title='Stage (m)', overlaying='y', side='right', showgrid=False) if use_rating_curve else None,
            hovermode='x unified' if x_col == 'date' else 'closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Max flow and stage with return period estimation
        if data_type in ["Historical", "Flow Duration Curve"] and max_flow is not None:
            if not rp_df.empty:
                interp_func = interp1d(np.log(rp_df['flow']), rp_df['return_period'], bounds_error=False, fill_value=(1, 1000))
                est_rp = interp_func(np.log(max_flow))
                if x_col == 'date':
                    fig.add_hline(y=max_flow, line_dash="dot", line_color="green",
                                  annotation_text=f"Max: {max_flow:.0f} m³/s (~{est_rp:.1f}-yr)")
                    if use_rating_curve:
                        fig.add_hline(y=max_stage, line_dash="dot", line_color="darkorange",
                                      annotation_text=f"Max: {max_stage:.2f} m (~{est_rp:.1f}-yr)", yref='y2')
                else:
                    exceedance_max = 100 / 365.25
                    fig.add_vline(x=exceedance_max, line_dash="dot", line_color="green",
                                  annotation_text=f"Max: {max_flow:.0f} m³/s (~{est_rp:.1f}-yr at {exceedance_max:.1f}%)")
                    if use_rating_curve:
                        fig.add_vline(x=exceedance_max, line_dash="dot", line_color="darkorange",
                                      annotation_text=f"Max: {max_stage:.2f} m (~{est_rp:.1f}-yr)", xref='x')
                st.info(f"Max flow: {max_flow:.2f} m³/s (estimated ~{est_rp:.1f}-year return period)")
                if use_rating_curve:
                    st.info(f"Max stage: {max_stage:.2f} m (estimated ~{est_rp:.1f}-year return period)")
        
        # Summary statistics
        st.subheader("Summary Statistics")
        if x_col == 'exceedance':
            q95 = original_df['flow'].quantile(0.05)
            q50 = original_df['flow'].quantile(0.50)
            q05 = original_df['flow'].quantile(0.95)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Q95 (Low Flow)", f"{q95:.2f} m³/s")
            with col2:
                st.metric("Q50 (Median)", f"{q50:.2f} m³/s")
            with col3:
                st.metric("Q05 (High Flow)", f"{q05:.2f} m³/s")
            if use_rating_curve:
                q95_stage = original_df['stage'].quantile(0.05)
                q50_stage = original_df['stage'].quantile(0.50)
                q05_stage = original_df['stage'].quantile(0.95)
                st.subheader("Stage Percentiles")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Q95 Stage", f"{q95_stage:.2f} m")
                with col2:
                    st.metric("Q50 Stage", f"{q50_stage:.2f} m")
                with col3:
                    st.metric("Q05 Stage", f"{q05_stage:.2f} m")
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
            if use_rating_curve:
                st.subheader("Stage Summary Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Stage", f"{df['stage'].mean():.2f} m")
                with col2:
                    st.metric("Max Stage", f"{max_stage:.2f} m")
                with col3:
                    st.metric("Min Stage", f"{df['stage'].min():.2f} m")
        
        # Return period thresholds
        if not rp_df.empty:
            st.subheader("Key Return Period Thresholds")
            rp_metrics = rp_df[rp_df['return_period'].isin([2, 5, 10, 25, 50, 100])][['return_period', 'flow']]
            if use_rating_curve:
                rp_metrics['stage'] = rp_df[rp_df['return_period'].isin([2, 5, 10, 25, 50, 100])]['stage']
            rp_metrics.columns = ['Return Period (years)', 'Threshold Flow (m³/s)', 'Threshold Stage (m)'] if use_rating_curve else ['Return Period (years)', 'Threshold Flow (m³/s)']
            st.table(rp_metrics)
            
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
