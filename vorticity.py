import streamlit as st
import xarray as xr
import numpy as np
import metpy
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_folium import st_folium
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from io import BytesIO
from metpy.units import units
from metpy.calc import vorticity, divergence
import time

# --- Load and parse data ---
@st.cache_data(show_spinner="Loading ERA5 data...")
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            ds = xr.open_dataset(tmp_path, engine="netcdf4")
            os.unlink(tmp_path)  # Clean up temp file after loading
        else:
            ds = xr.open_dataset("era_5.nc", engine="netcdf4")
        ds = ds.metpy.parse_cf()

        # Validate required variables
        required_vars = ['u', 'v']
        missing_vars = [var for var in required_vars if var not in ds.data_vars]
        if missing_vars:
            st.error(f"Missing required variables: {missing_vars}")
            st.stop()

        # Validate coordinates
        required_coords = ['latitude', 'longitude', 'pressure_level']
        missing_coords = [coord for coord in required_coords if coord not in ds.coords]
        if missing_coords:
            st.error(f"Missing required coordinates: {missing_coords}")
            st.stop()

        return ds
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

st.set_page_config(layout="wide")
st.title("🌬️ ERA5 Wind Data Explorer")

# --- File uploader ---
uploaded_file = st.sidebar.file_uploader("Upload NetCDF File", type=["nc"])
ds = load_data(uploaded_file)

# --- Time coordinate detection ---
time_coord = next((c for c in ds.coords if 'time' in c.lower()), None)
if time_coord is None:
    st.error("No time coordinate found in the dataset.")
    st.stop()

lat = ds.latitude
lon = ds.longitude
lons, lats = np.meshgrid(lon, lat)

# --- Sidebar controls ---
st.sidebar.title("🛠️ Controls")
time_values = ds[time_coord].values
selected_level = st.sidebar.selectbox("Pressure Level (hPa)", ds.pressure_level.values, help="Select the atmospheric pressure level for analysis")
variables_to_plot = st.sidebar.multiselect("Variables to Overlay", ["Vorticity", "Divergence", "Wind Vectors", "Wind Speed"], default=["Wind Vectors"], help="Choose which meteorological variables to display")
map_type = st.sidebar.radio("Map Type", ["Static (Matplotlib)", "Dynamic (Plotly)", "Interactive (Folium)"], help="Select visualization type: Static for detailed plots, Dynamic for interactive zooming, Interactive for web-based maps")

# --- Animation Controls ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎬 Animation")
play_animation = st.sidebar.checkbox("▶️ Auto Play Animation", help="Automatically cycle through time steps")
frame_rate = st.sidebar.slider("Frame Rate (sec)", 0.1, 2.0, 0.5, 0.1, help="Control animation speed (lower = faster)")

# --- UI Containers ---
plot_col, map_col = st.columns([3, 2])

# --- Animation Handler ---
if play_animation:
    if 'time_index' not in st.session_state:
        st.session_state['time_index'] = 0
    if st.sidebar.button("▶️ Start Animation"):
        st.session_state['animating'] = True
    if st.sidebar.button("⏸️ Pause Animation"):
        st.session_state['animating'] = False
    if st.session_state.get('animating', False):
        st.session_state['time_index'] = (st.session_state['time_index'] + 1) % len(time_values)
        time.sleep(frame_rate)
        st.rerun()
else:
    st.session_state['time_index'] = st.sidebar.slider("Time Step", 0, len(time_values) - 1, st.session_state.get('time_index', 0))

selected_time = time_values[st.session_state['time_index']]

# --- Extract u/v components ---
u = ds['u'].sel({time_coord: selected_time, "pressure_level": selected_level}).metpy.quantify()
v = ds['v'].sel({time_coord: selected_time, "pressure_level": selected_level}).metpy.quantify()

# --- Derived calculations ---
vort = vorticity(u, v).metpy.dequantify()
div = divergence(u, v).metpy.dequantify()
wind_speed = np.sqrt(u ** 2 + v ** 2).metpy.dequantify()

# --- Plot function ---
def generate_static_plot():
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    if "Vorticity" in variables_to_plot:
        vort_plot = ax.contourf(lon, lat, vort, levels=21, cmap="RdBu_r", transform=ccrs.PlateCarree())
        plt.colorbar(vort_plot, ax=ax, shrink=0.8, label="Vorticity (1/s)")
    if "Divergence" in variables_to_plot:
        div_plot = ax.contourf(lon, lat, div, levels=21, cmap="PiYG", transform=ccrs.PlateCarree())
        plt.colorbar(div_plot, ax=ax, shrink=0.8, label="Divergence (1/s)")
    if "Wind Speed" in variables_to_plot:
        speed_plot = ax.contourf(lon, lat, wind_speed, levels=21, cmap="plasma", transform=ccrs.PlateCarree())
        plt.colorbar(speed_plot, ax=ax, shrink=0.8, label="Wind Speed (m/s)")
    if "Wind Vectors" in variables_to_plot:
        stride = 3
        ax.quiver(lons[::stride, ::stride], lats[::stride, ::stride],
                  u.values[::stride, ::stride], v.values[::stride, ::stride],
                  transform=ccrs.PlateCarree(), scale=700, color='black', width=0.003)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.set_title(f"ERA5 at {selected_level} hPa | {np.datetime_as_string(selected_time, 'h')}", fontsize=14, fontweight='bold')
    return fig

# --- Dynamic Plotly Plot ---
def generate_dynamic_plot():
    fig = px.imshow(wind_speed, x=lon, y=lat, origin='lower',
                    color_continuous_scale='plasma',
                    title=f"Wind Speed at {selected_level} hPa | {np.datetime_as_string(selected_time, 'h')}")
    return fig

# --- Folium Map ---
def generate_folium_map():
    try:
        import folium
        from folium.plugins import HeatMap
        m = folium.Map(location=[float(lat.mean().values), float(lon.mean().values)], zoom_start=2)
        # Add wind speed as heatmap
        data_flat = wind_speed.values.flatten()
        lat_flat = lat.values.flatten()
        lon_flat = lon.values.flatten()
        # Convert numpy types to Python types for JSON serialization
        heat_data = [[float(row[0]), float(row[1]), float(row[2])] for row in zip(lat_flat, lon_flat, data_flat)]
        HeatMap(heat_data).add_to(m)
        return m
    except Exception as e:
        st.error(f"Error generating Folium map: {e}")
        return None

# --- Main plot render ---
with plot_col:
    if map_type == "Static (Matplotlib)":
        fig = generate_static_plot()
        st.pyplot(fig)
    elif map_type == "Dynamic (Plotly)":
        st.plotly_chart(generate_dynamic_plot(), use_container_width=True)
    elif map_type == "Interactive (Folium)":
        fol_map = generate_folium_map()
        if fol_map is None:
            st.error("Folium is not installed in this environment. Install it with 'pip install folium streamlit-folium' to use interactive maps.")
        else:
            st_folium(fol_map, width=700)

# --- Download Buttons ---
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Downloads")

buffer = BytesIO()
if map_type == "Static (Matplotlib)":
    fig.savefig(buffer, format="png", dpi=300, bbox_inches='tight')
    st.sidebar.download_button("📥 Download PNG", buffer.getvalue(), file_name=f"era5_{selected_level}hPa_{np.datetime_as_string(selected_time, 'D')}.png", mime="image/png")
elif map_type == "Dynamic (Plotly)":
    fig = generate_dynamic_plot()
    buffer = fig.to_image(format="png")
    st.sidebar.download_button("📥 Download PNG", buffer, file_name=f"era5_plotly_{selected_level}hPa_{np.datetime_as_string(selected_time, 'D')}.png", mime="image/png")

# --- CSV Export ---
data_flat = wind_speed.values.flatten()
lat_flat = lat.values.flatten()
lon_flat = lon.values.flatten()
csv_data = "lat,lon,wind_speed\n" + "\n".join(f"{la},{lo},{val}" for la, lo, val in zip(lat_flat, lon_flat, data_flat))
st.sidebar.download_button("📥 Download CSV", csv_data, file_name=f"era5_data_{selected_level}hPa_{np.datetime_as_string(selected_time, 'D')}.csv", mime="text/csv")

# Additional data exports
if "Vorticity" in variables_to_plot:
    vort_flat = vort.values.flatten()
    vort_csv = "lat,lon,vorticity\n" + "\n".join(f"{la},{lo},{val}" for la, lo, val in zip(lat_flat, lon_flat, vort_flat))
    st.sidebar.download_button("📥 Download Vorticity CSV", vort_csv, file_name=f"era5_vorticity_{selected_level}hPa_{np.datetime_as_string(selected_time, 'D')}.csv", mime="text/csv")

if "Divergence" in variables_to_plot:
    div_flat = div.values.flatten()
    div_csv = "lat,lon,divergence\n" + "\n".join(f"{la},{lo},{val}" for la, lo, val in zip(lat_flat, lon_flat, div_flat))
    st.sidebar.download_button("📥 Download Divergence CSV", div_csv, file_name=f"era5_divergence_{selected_level}hPa_{np.datetime_as_string(selected_time, 'D')}.csv", mime="text/csv")
