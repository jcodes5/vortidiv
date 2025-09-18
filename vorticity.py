import streamlit as st
import xarray as xr
import numpy as np
import metpy
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from streamlit_folium import st_folium
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from io import BytesIO
from metpy.units import units
from metpy.calc import vorticity, divergence
import time

# --- Load and parse data ---
@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        ds = xr.open_dataset(uploaded_file, engine="netcdf4")
    else:
        ds = xr.open_dataset("era_5.nc", engine="netcdf4")
    ds = ds.metpy.parse_cf()
    return ds

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
selected_level = st.sidebar.selectbox("Pressure Level (hPa)", ds.pressure_level.values)
variables_to_plot = st.sidebar.multiselect("Variables to Overlay", ["Vorticity", "Divergence", "Wind Vectors", "Wind Speed"], default=["Wind Vectors"])
map_type = st.sidebar.radio("Map Type", ["Static (Matplotlib)", "Dynamic (Plotly)", "Interactive (Folium)"])

# --- Animation Controls ---
st.sidebar.markdown("---")
play_animation = st.sidebar.checkbox("▶️ Auto Play Animation")
frame_rate = st.sidebar.slider("Frame Rate (sec)", 0.1, 2.0, 0.5, 0.1)

# --- UI Containers ---
plot_col, map_col = st.columns([3, 2])

# --- Animation Handler ---
if play_animation:
    for i in range(len(time_values)):
        st.session_state['time_index'] = i
        time.sleep(frame_rate)
else:
    st.session_state['time_index'] = st.sidebar.slider("Time Step", 0, len(time_values) - 1, 0)

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
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    if "Vorticity" in variables_to_plot:
        ax.contourf(lon, lat, vort, levels=21, cmap="RdBu_r", transform=ccrs.PlateCarree())
    if "Divergence" in variables_to_plot:
        ax.contourf(lon, lat, div, levels=21, cmap="PiYG", transform=ccrs.PlateCarree())
    if "Wind Speed" in variables_to_plot:
        ax.contourf(lon, lat, wind_speed, levels=21, cmap="plasma", transform=ccrs.PlateCarree())
    if "Wind Vectors" in variables_to_plot:
        stride = 3
        ax.quiver(lons[::stride, ::stride], lats[::stride, ::stride],
                  u.values[::stride, ::stride], v.values[::stride, ::stride],
                  transform=ccrs.PlateCarree(), scale=700)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.set_title(f"ERA5 at {selected_level} hPa | {np.datetime_as_string(selected_time, 'h')}", fontsize=12)
    return fig

# --- Dynamic Plotly Plot ---
def generate_dynamic_plot():
    fig = px.imshow(wind_speed, x=lon, y=lat, origin='lower',
                    color_continuous_scale='plasma',
                    title=f"Wind Speed at {selected_level} hPa | {np.datetime_as_string(selected_time, 'h')}")
    return fig

# --- Folium Map ---
def generate_folium_map():
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=3, tiles="Stamen Terrain")
    return m

# --- Main plot render ---
with plot_col:
    if map_type == "Static (Matplotlib)":
        fig = generate_static_plot()
        st.pyplot(fig)
    elif map_type == "Dynamic (Plotly)":
        st.plotly_chart(generate_dynamic_plot(), use_container_width=True)
    elif map_type == "Interactive (Folium)":
        st_folium(generate_folium_map(), width=700)

# --- Download Buttons ---
buffer = BytesIO()
if map_type == "Static (Matplotlib)":
    fig.savefig(buffer, format="png")
    st.download_button("📥 Download Image", buffer.getvalue(), file_name="plot.png", mime="image/png")

# --- CSV Export ---
data_flat = wind_speed.values.flatten()
lat_flat = lat.values.flatten()
lon_flat = lon.values.flatten()
csv_data = "lat,lon,value\n" + "\n".join(f"{la},{lo},{val}" for la, lo, val in zip(lat_flat, lon_flat, data_flat))
st.download_button("📥 Download Data CSV", csv_data, file_name="data.csv", mime="text/csv")
