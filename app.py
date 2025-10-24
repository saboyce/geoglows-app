#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoGlows Viewer – Forecast, Historical, FDC
Now with:
  • Rating curve (stage)
  • Plot toggle: Flow only / Stage only / Both
  • Uncertainty shown for Flow-only and Stage-only
  • Axes hidden when not in use
"""

import streamlit as st
import geoglows.data as gdata
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
from io import StringIO

# ------------------------------------------------------------
# River catalogue
rivers_df = pd.DataFrame({
    'country': ['Trinidad and Tobago', 'Jamaica', 'Jamaica', 'Jamaica', 'Belize', 'Belize', 'Belize', 'Belize', 'Haiti'],
    'location': ['Coroni_River@Piarco', 'Black_River@Crane_Road', 'Rio_Minho@BrokenBank', 'Rio_Cobre@Bogwalk', 'South_Stann_Creek@Highway', 'Belize_River@Belize_City', 'New_River@Corozal', 'Rio_Hondo@Chetumal', 'Ravine_du_Sud@Les_Cayes'],
    'id': [610171862, 780056741, 780006059, 780062115, 770244113, 770335606, 770332458, 770299173, 780071323]
})

# ------------------------------------------------------------
# Rating-curve helper
def calculate_stage(discharge, a, b, c):
    """stage = a * discharge^b + c   (discharge >= 0)"""
    try:
        discharge = np.maximum(discharge, 0)
        return a * (discharge ** b) + c
    except Exception as e:
        st.error(f"Stage calculation error: {e}")
        return np.nan

# ------------------------------------------------------------
# UI
st.title("GeoGlows Streamflow Viewer")
st.write("Select river, data type, and optional rating curve. "
         "Toggle **Flow**, **Stage**, or **Both** on plots.")

# ---------- Sidebar ----------
st.sidebar.header("River Selection")
use_manual = st.sidebar.checkbox("Use Manual River ID", value=False)

if use_manual:
    river_id = st.sidebar.number_input("River ID", value=760021611, step=1, format="%d")
    river_name = "Custom River"
else:
    countries = sorted(rivers_df['country'].unique())
    sel_country = st.sidebar.selectbox("Country", options=countries)
    locs = sorted(rivers_df[rivers_df['country'] == sel_country]['location'].unique())
    sel_loc = st.sidebar.selectbox("Location", options=locs)
    river_row = rivers_df[(rivers_df['country'] == sel_country) &
                         (rivers_df['location'] == sel_loc)].iloc[0]
    river_id = river_row['id']
    river_name = river_row['location']
    st.sidebar.info(f"**{river_name}** (ID: {river_id})")

# ---------- Rating curve ----------
st.sidebar.header("Rating Curve (stage = a·Q^b + c)")
use_rating_curve = st.sidebar.checkbox("Apply Rating Curve", value=False)
if use_rating_curve:
    a_coeff = st.sidebar.number_input("a (scale)", value=0.01, format="%.6f")
    b_coeff = st.sidebar.number_input("b (exponent)", value=0.5, format="%.6f")
    c_coeff = st.sidebar.number_input("c (offset, m)", value=0.0, format="%.6f")

    plot_view = st.sidebar.radio(
        "Plot view",
        options=["Both", "Flow only", "Stage only"],
        index=0,
        horizontal=True
    )
else:
    plot_view = "Flow only"

# ---------- Data type ----------
data_type = st.sidebar.selectbox("Data Type", ["Forecast", "Historical", "Flow Duration Curve"])

if data_type == "Forecast":
    show_uncertainty_global = st.sidebar.checkbox("Show Uncertainty Bounds", value=True)

if data_type == "Historical":
    resolution = st.sidebar.selectbox("Resolution", ["hourly", "daily", "monthly", "yearly"])
    start_date = st.sidebar.date_input("Start Date", value=datetime(2000, 1, 1))
    end_date   = st.sidebar.date_input("End Date",   value=datetime(2025, 10, 3))
    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

# ------------------------------------------------------------
# Fetch data
if st.button("Fetch Data"):
    with st.spinner("Fetching data…"):
        rp_df = pd.DataFrame()
        max_flow = max_stage = None
        original_df = df = None
        x_col = y_col = None
        title = None
        rp_overlay = False
        forecast_df = None
        forecast_y_col = None
        forecast_show_uncertainty = False

        # ----- Forecast -------------------------------------------------
        if data_type == "Forecast":
            raw = gdata.forecast(river_id=river_id, data_source='rest')
            if isinstance(raw, str):
                forecast_df = pd.read_csv(StringIO(raw), parse_dates=['date'])
            else:
                forecast_df = raw.copy()
            forecast_df = forecast_df.reset_index(drop=False)
            if 'date' not in forecast_df.columns:
                forecast_df = forecast_df.rename(columns={forecast_df.columns[0]: 'date'})
            for col in ['flow_median', 'flow']:
                if col in forecast_df.columns:
                    forecast_y_col = col
                    break
            else:
                st.error("No flow column found.")
                st.stop()
            forecast_show_uncertainty = 'flow_uncertainty_upper' in forecast_df.columns
            max_flow = forecast_df[forecast_y_col].max()

            if use_rating_curve:
                forecast_df['stage'] = calculate_stage(forecast_df[forecast_y_col],
                                                       a_coeff, b_coeff, c_coeff)
                max_stage = forecast_df['stage'].max()
                # Convert flow uncertainty → stage uncertainty
                if forecast_show_uncertainty:
                    forecast_df['stage_upper'] = calculate_stage(
                        forecast_df['flow_uncertainty_upper'], a_coeff, b_coeff, c_coeff)
                    forecast_df['stage_lower'] = calculate_stage(
                        forecast_df['flow_uncertainty_lower'], a_coeff, b_coeff, c_coeff)

            df = forecast_df.copy()
            y_col = forecast_y_col
            x_col = 'date'
            title = f"Ensemble Median Forecast – {river_name}"
            rp_overlay = True

        # ----- Historical -----------------------------------------------
        elif data_type == "Historical":
            kwargs = {'resolution': resolution}
            if 'start_str' in locals():
                kwargs['start'] = start_str
                kwargs['end']   = end_str
            raw = gdata.retrospective(river_id=river_id, **kwargs)
            if isinstance(raw, str):
                df = pd.read_csv(StringIO(raw), index_col=0, parse_dates=True)
            else:
                df = raw.copy()
            df = df.reset_index(drop=False)
            df = df.rename(columns={df.columns[0]: 'date', river_id: 'flow'})
            if use_rating_curve:
                df['stage'] = calculate_stage(df['flow'], a_coeff, b_coeff, c_coeff)
                max_stage = df['stage'].max()
            y_col = 'flow'
            x_col = 'date'
            title = f"Historical ({resolution}) – {river_name}"
            rp_overlay = True
            max_flow = df['flow'].max()
            original_df = df.copy()

        # ----- Flow Duration Curve ---------------------------------------
        else:  # FDC
            raw = gdata.retrospective(river_id=river_id, resolution='daily')
            if isinstance(raw, str):
                original_df = pd.read_csv(StringIO(raw), index_col=0, parse_dates=True)
            else:
                original_df = raw.copy()
            original_df = original_df.reset_index(drop=False)
            original_df = original_df.rename(columns={original_df.columns[0]: 'date',
                                                     river_id: 'flow'})
            if use_rating_curve:
                original_df['stage'] = calculate_stage(original_df['flow'],
                                                       a_coeff, b_coeff, c_coeff)
                max_stage = original_df['stage'].max()

            flows = original_df['flow'].sort_values(ascending=False).reset_index(drop=True)
            exceed = (np.arange(1, len(flows)+1) / len(flows)) * 100
            df = pd.DataFrame({'exceedance': exceed, 'flow': flows})
            if use_rating_curve:
                stages = original_df['stage'].sort_values(ascending=False).reset_index(drop=True)
                df['stage'] = stages
            y_col = 'flow'
            x_col = 'exceedance'
            title = f"Flow Duration Curve – {river_name}"
            rp_overlay = True
            max_flow = original_df['flow'].max()

        # ----- Return periods --------------------------------------------
        try:
            rp_raw = gdata.return_periods(river_id=river_id)
            if isinstance(rp_raw, str):
                rp_raw = pd.read_csv(StringIO(rp_raw))
            if not rp_raw.empty:
                rp_df = rp_raw.reset_index(names=['return_period'])
                rp_df = rp_df.rename(columns={river_id: 'flow'})
                if use_rating_curve:
                    rp_df['stage'] = calculate_stage(rp_df['flow'],
                                                     a_coeff, b_coeff, c_coeff)
        except Exception as e:
            st.warning(f"Return periods unavailable: {e}")

        # ------------------------------------------------------------------
        # 1. Forecast-only preview (no RP thresholds)
        # ------------------------------------------------------------------
        if data_type == "Forecast" and forecast_df is not None:
            st.subheader("Latest Forecast (no thresholds)")

            fig_fc = go.Figure()

            # === Flow trace + uncertainty ===
            if plot_view in ["Both", "Flow only"]:
                fig_fc.add_trace(go.Scatter(
                    x=forecast_df['date'], y=forecast_df[forecast_y_col],
                    mode='lines', name='Flow (m³/s)', line=dict(color='royalblue'),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Flow:</b> %{y:.2f} m³/s<extra></extra>'
                ))
                if show_uncertainty_global and forecast_show_uncertainty:
                    fig_fc.add_traces([
                        go.Scatter(x=forecast_df['date'], y=forecast_df['flow_uncertainty_upper'],
                                   fill=None, mode='lines', line=dict(width=0), showlegend=False),
                        go.Scatter(x=forecast_df['date'], y=forecast_df['flow_uncertainty_lower'],
                                   fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                                   line=dict(width=0), name='Flow Uncertainty')
                    ])

            # === Stage trace + uncertainty ===
            if use_rating_curve and plot_view in ["Both", "Stage only"]:
                fig_fc.add_trace(go.Scatter(
                    x=forecast_df['date'], y=forecast_df['stage'],
                    mode='lines', name='Stage (m)', line=dict(color='orange'),
                    yaxis='y2',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Stage:</b> %{y:.2f} m<extra></extra>'
                ))
                if show_uncertainty_global and forecast_show_uncertainty:
                    fig_fc.add_traces([
                        go.Scatter(x=forecast_df['date'], y=forecast_df['stage_upper'],
                                   fill=None, mode='lines', line=dict(width=0), showlegend=False, yaxis='y2'),
                        go.Scatter(x=forecast_df['date'], y=forecast_df['stage_lower'],
                                   fill='tonexty', fillcolor='rgba(255,165,0,0.2)',
                                   line=dict(width=0), name='Stage Uncertainty', yaxis='y2')
                    ])

            # Layout – hide axes not in use
            yaxis_cfg = dict(title='Flow (m³/s)')
            yaxis2_cfg = dict(title='Stage (m)', overlaying='y', side='right', showgrid=False)

            if plot_view == "Stage only":
                yaxis_cfg = dict(visible=False)
            elif plot_view == "Flow only":
                yaxis2_cfg = None

            fig_fc.update_layout(
                title=f"Forecast – {river_name}",
                xaxis=dict(title='Date/Time'),
                yaxis=yaxis_cfg,
                yaxis2=yaxis2_cfg,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Stats
            st.subheader("Forecast Summary")
            cols = st.columns(4)
            cols[0].metric("Mean Flow", f"{forecast_df[forecast_y_col].mean():.2f} m³/s")
            cols[1].metric("Max Flow",  f"{max_flow:.2f} m³/s")
            cols[2].metric("Min Flow",  f"{forecast_df[forecast_y_col].min():.2f} m³/s")
            cols[3].metric("Points",    f"{len(forecast_df)}")
            if use_rating_curve:
                st.subheader("Stage Summary")
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean Stage", f"{forecast_df['stage'].mean():.2f} m")
                c2.metric("Max Stage",  f"{max_stage:.2f} m")
                c3.metric("Min Stage",  f"{forecast_df['stage'].min():.2f} m")

            csv_fc = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Forecast CSV",
                               data=csv_fc,
                               file_name=f'forecast_{river_id}.csv',
                               mime='text/csv')

        # ------------------------------------------------------------------
        # 2. Main analysis plot (Historical / FDC / Forecast with RP)
        # ------------------------------------------------------------------
        if df is not None and not df.empty:
            st.subheader("Data Table")
            display = df.copy()
            if data_type == "Forecast" and not show_uncertainty_global:
                display = display.drop(columns=['flow_uncertainty_upper',
                                                'flow_uncertainty_lower',
                                                'stage_upper', 'stage_lower'], errors='ignore')
            st.dataframe(display.head(100))

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV",
                               data=csv,
                               file_name=f'{data_type.lower().replace(" ", "_")}_{river_id}.csv',
                               mime='text/csv')

            # ------------------- Plot -------------------
            st.subheader("Analysis Plot")
            fig = go.Figure()

            # === Flow trace + uncertainty ===
            if plot_view in ["Both", "Flow only"]:
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df[y_col],
                    mode='lines', name='Flow (m³/s)', line=dict(color='royalblue'),
                    hovertemplate=f'<b>{x_col.capitalize()}:</b> %{{x}}<br><b>Flow:</b> %{{y:.2f}} m³/s<extra></extra>'
                ))
                if data_type == "Forecast" and show_uncertainty_global and forecast_show_uncertainty:
                    fig.add_traces([
                        go.Scatter(x=df['date'], y=df['flow_uncertainty_upper'],
                                   fill=None, mode='lines', line=dict(width=0), showlegend=False),
                        go.Scatter(x=df['date'], y=df['flow_uncertainty_lower'],
                                   fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                                   line=dict(width=0), name='Flow Uncertainty')
                    ])

            # === Stage trace + uncertainty ===
            if use_rating_curve and plot_view in ["Both", "Stage only"]:
                fig.add_trace(go.Scatter(
                    x=df[x_col], y=df['stage'],
                    mode='lines', name='Stage (m)', line=dict(color='orange'),
                    yaxis='y2',
                    hovertemplate=f'<b>{x_col.capitalize()}:</b> %{{x}}<br><b>Stage:</b> %{{y:.2f}} m<extra></extra>'
                ))
                if data_type == "Forecast" and show_uncertainty_global and forecast_show_uncertainty:
                    fig.add_traces([
                        go.Scatter(x=df['date'], y=df['stage_upper'],
                                   fill=None, mode='lines', line=dict(width=0), showlegend=False, yaxis='y2'),
                        go.Scatter(x=df['date'], y=df['stage_lower'],
                                   fill='tonexty', fillcolor='rgba(255,165,0,0.2)',
                                   line=dict(width=0), name='Stage Uncertainty', yaxis='y2')
                    ])

            # === Return-period shading ===
            if rp_overlay and not rp_df.empty:
                key_rps = [2, 5, 10, 25, 50, 100]
                colors = ['rgba(255,165,0,0.3)', 'rgba(255,215,0,0.3)', 'rgba(255,0,0,0.3)',
                          'rgba(255,0,255,0.3)', 'rgba(128,0,128,0.3)', 'rgba(165,42,42,0.3)']
                rp_flow = {rp: rp_df[rp_df['return_period']==rp]['flow'].iloc[0]
                           for rp in key_rps if rp in rp_df['return_period'].values}
                sorted_rp = sorted(rp_flow.keys())
                sorted_fl = [rp_flow[rp] for rp in sorted_rp]

                x_min, x_max = df[x_col].min(), df[x_col].max()
                x_span = [x_min, x_max]

                for i in range(len(sorted_fl)-1):
                    low, high = sorted_fl[i], sorted_fl[i+1]
                    name = f"{sorted_rp[i]}-{sorted_rp[i+1]} yr"
                    fig.add_traces([
                        go.Scatter(x=x_span, y=[high, high], mode='lines', line=dict(width=0), showlegend=False),
                        go.Scatter(x=x_span, y=[low, low], fill='tonexty', fillcolor=colors[i],
                                   line=dict(width=0), name=name, showlegend=True)
                    ])

                # >100-yr flow line
                if plot_view in ["Both", "Flow only"]:
                    fig.add_hline(y=sorted_fl[-1], line_dash='dash', line_color='brown',
                                  annotation_text=f">100-yr: {sorted_fl[-1]:.0f} m³/s")

                # >100-yr stage line
                if use_rating_curve and plot_view in ["Both", "Stage only"]:
                    rp_stage = {rp: rp_df[rp_df['return_period']==rp]['stage'].iloc[0]
                                for rp in key_rps if rp in rp_df['return_period'].values}
                    highest_stage = rp_stage[sorted_rp[-1]]
                    fig.add_hline(y=highest_stage, line_dash='dash', line_color='purple',
                                  yref='y2', annotation_text=f">100-yr: {highest_stage:.2f} m")

            # === Max annotation ===
            if data_type in ["Historical", "Flow Duration Curve"] and max_flow is not None:
                if not rp_df.empty:
                    interp = interp1d(np.log(rp_df['flow']), rp_df['return_period'],
                                      bounds_error=False, fill_value=(1, 1000))
                    est_rp = interp(np.log(max_flow))
                    if x_col == 'date':
                        if plot_view in ["Both", "Flow only"]:
                            fig.add_hline(y=max_flow, line_dash='dot', line_color='green',
                                          annotation_text=f"Max: {max_flow:.0f} m³/s (~{est_rp:.1f}-yr)")
                        if use_rating_curve and plot_view in ["Both", "Stage only"]:
                            fig.add_hline(y=max_stage, line_dash='dot', line_color='darkorange',
                                          yref='y2', annotation_text=f"Max: {max_stage:.2f} m (~{est_rp:.1f}-yr)")
                    else:
                        exc = 100 / 365.25
                        if plot_view in ["Both", "Flow only"]:
                            fig.add_vline(x=exc, line_dash='dot', line_color='green',
                                          annotation_text=f"Max: {max_flow:.0f} m³/s (~{est_rp:.1f}-yr)")
                        if use_rating_curve and plot_view in ["Both", "Stage only"]:
                            fig.add_vline(x=exc, line_dash='dot', line_color='darkorange',
                                          annotation_text=f"Max: {max_stage:.2f} m (~{est_rp:.1f}-yr)")

                    st.info(f"Max flow: **{max_flow:.2f}** m³/s (~{est_rp:.1f}-yr RP)")
                    if use_rating_curve and plot_view != "Flow only":
                        st.info(f"Max stage: **{max_stage:.2f}** m (~{est_rp:.1f}-yr RP)")

            # === Layout – hide unused axes ===
            yaxis_cfg = dict(title='Flow (m³/s)', type='log' if x_col == 'exceedance' else 'linear')
            yaxis2_cfg = dict(title='Stage (m)', overlaying='y', side='right', showgrid=False)

            if plot_view == "Stage only":
                yaxis_cfg = dict(visible=False)
            elif plot_view == "Flow only":
                yaxis2_cfg = None

            fig.update_layout(
                title=title,
                xaxis=dict(title='Exceedance Probability (%)' if x_col == 'exceedance' else 'Date/Time'),
                yaxis=yaxis_cfg,
                yaxis2=yaxis2_cfg,
                hovermode='x unified' if x_col == 'date' else 'closest',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ------------------- Stats -------------------
            st.subheader("Summary Statistics")
            if x_col == 'exceedance':
                q95 = original_df['flow'].quantile(0.05)
                q50 = original_df['flow'].quantile(0.50)
                q05 = original_df['flow'].quantile(0.95)
                c1, c2, c3 = st.columns(3)
                c1.metric("Q95 (low)", f"{q95:.2f} m³/s")
                c2.metric("Q50 (median)", f"{q50:.2f} m³/s")
                c3.metric("Q05 (high)", f"{q05:.2f} m³/s")
                if use_rating_curve and plot_view != "Flow only":
                    s95 = original_df['stage'].quantile(0.05)
                    s50 = original_df['stage'].quantile(0.50)
                    s05 = original_df['stage'].quantile(0.95)
                    st.subheader("Stage Percentiles")
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Q95", f"{s95:.2f} m")
                    d2.metric("Q50", f"{s50:.2f} m")
                    d3.metric("Q05", f"{s05:.2f} m")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean Flow", f"{df[y_col].mean():.2f} m³/s")
                c2.metric("Max Flow",  f"{max_flow:.2f} m³/s")
                c3.metric("Min Flow",  f"{df[y_col].min():.2f} m³/s")
                c4.metric("Points",    f"{len(df)}")
                if use_rating_curve and plot_view != "Flow only":
                    st.subheader("Stage Statistics")
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Mean Stage", f"{df['stage'].mean():.2f} m")
                    d2.metric("Max Stage",  f"{max_stage:.2f} m")
                    d3.metric("Min Stage",  f"{df['stage'].min():.2f} m")

            # ------------------- Return periods -------------------
            if not rp_df.empty:
                st.subheader("Key Return-Period Thresholds")
                tbl = rp_df[rp_df['return_period'].isin([2,5,10,25,50,100])][['return_period','flow']]
                if use_rating_curve:
                    tbl['stage'] = rp_df[rp_df['return_period'].isin([2,5,10,25,50,100])]['stage']
                tbl.columns = ['Return Period (yr)', 'Flow (m³/s)', 'Stage (m)'] if use_rating_curve else ['Return Period (yr)', 'Flow (m³/s)']
                st.table(tbl)

                rp_csv = rp_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Return-Periods CSV",
                                   data=rp_csv,
                                   file_name=f'return_periods_{river_id}.csv',
                                   mime='text/csv')
        else:
            st.error("No data retrieved – check river ID / network.")
