from datetime import datetime
import matplotlib.pyplot as plt
from metpy.plots import SkewT,Hodograph
from metpy.units import pandas_dataframe_to_unit_arrays, units
import numpy as np
from siphon.simplewebservice.wyoming import WyomingUpperAir
import xarray as xr
import pandas as pd
import metpy.calc as mpcalc
from metpy.calc import dewpoint_from_relative_humidity
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Leer el archivo grib2
ds = xr.open_dataset('cdas1_2022011700_.t00z.pgrbh00.grib2', engine='cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel': 'isobaricInhPa'}})

# Extraer las variables necesarias para realizar el calculo del sondeo Skew-T
p_sondeo = ds.coords['isobaricInhPa'].values * units.hPa # Presión en hPa (37 niveles)
t_sondeo = (ds.t.sel(latitude=-35.0, longitude=304.0).values - 273.15) * units.degC# Temperatura en superficie en Montevideo (37 niveles)
u_sondeo = (ds.u.sel(latitude=-35.0, longitude=304.0).values) 
v_sondeo = (ds.v.sel(latitude=-35.0, longitude=304.0).values) 
r_sondeo = np.float64(ds.r.sel(latitude=-35.0, longitude=304.0).values) * units.percent # Humedad relativa en superficie en Montevideo (37 niveles)

p_sondeo = p_sondeo[:30]
t_sondeo = t_sondeo[:30]
u_sondeo = u_sondeo[:30]
v_sondeo = v_sondeo[:30]
r_sondeo = r_sondeo[:30]
fecha = np.repeat( ds.valid_time.dt.strftime('%Y-%m-%d').values,30)

# calcular el punto de rocío a partir de la humedad relativa y la temperatura con metpy
tp_sondeo = dewpoint_from_relative_humidity(t_sondeo, r_sondeo ) #* units.degC # Temperatura de punto de rocío en superficie en Montevideo (37 niveles)

# Calcular la velocidad del viento a partir de las componentes u y v con metpy
velocidad_sondeo = mpcalc.wind_speed(ds.u.sel(latitude=-35.0, longitude=304.0), ds.v.sel(latitude=-35.0, longitude=304.0)) # Velocidad del viento en superficie en Montevideo (37 niveles)
velocidad_sondeo = velocidad_sondeo.values[:30] * units('m/s')

# Calcular la dirección del viento a partir de las componentes u y v con metpy
direccion_sondeo = mpcalc.wind_direction(ds.u.sel(latitude=-35.0, longitude=304.0), ds.v.sel(latitude=-35.0, longitude=304.0)) #* units('radians') # Dirección del viento en superficie en Montevideo (37 niveles)
direccion_sondeo = direccion_sondeo.values[:30] * units('radians')

# Crea un diccionario con los datos
dato = {
    'Presion (hPa)': p_sondeo,
    'Temperatura (°C)': t_sondeo.magnitude,  # Se extrae el valor numérico de la temperatura
    'Punto de Rocio (°C)': tp_sondeo.magnitude,  # Se extrae el valor numérico del punto de rocío
    'Direccion (°radianes)': direccion_sondeo.magnitude,  # Se extrae el valor numérico de la dirección del viento
    'Velocidad (m/s)': velocidad_sondeo.magnitude,  # Se extrae el valor numérico de la velocidad del viento    
    'Velocidad U (m/s)': u_sondeo,
    'Velocidad V (m/s)': v_sondeo,
    'Humedad Relativa (%)': r_sondeo.magnitude,  # Se extrae el valor numérico de la humedad relativa    
    'fecha': fecha
    
}

# Crea el DataFrame
df = pd.DataFrame(dato)

# Calcular el LCL
lcl_presion, lcl_temperatura = mpcalc.lcl(p_sondeo[0], t_sondeo[0], tp_sondeo[0])

# Calcular el perfil de la parcela.
parcel_prof = mpcalc.parcel_profile(p_sondeo, t_sondeo[0], tp_sondeo[0]).to('degC')

# Calcular parámetros de índice de sondeo comunes
ctotals = mpcalc.cross_totals(p_sondeo, t_sondeo, tp_sondeo)
kindex = mpcalc.k_index(p_sondeo, t_sondeo, tp_sondeo)
showalter = mpcalc.showalter_index(p_sondeo, t_sondeo, tp_sondeo)
total_totals = mpcalc.total_totals_index(p_sondeo, t_sondeo, tp_sondeo)
vert_totals = mpcalc.vertical_totals(p_sondeo, t_sondeo)

# Calcule los valores LI, CAPE, CIN correspondientes para una parcela de superficie
lift_index = mpcalc.lifted_index(p_sondeo, t_sondeo, parcel_prof)
cape, cin = mpcalc.cape_cin(p_sondeo, t_sondeo, tp_sondeo, parcel_prof)

# Determine el LCL, LFC y EL para nuestra parcela de superficie
# lcl_presion, lcl_temperatura = mpcalc.lcl(p_sondeo[0], t_sondeo[0], tp_sondeo[0])
lfcp, _ = mpcalc.lfc(p_sondeo, t_sondeo, tp_sondeo)
el_pressure, _ = mpcalc.el(p_sondeo, t_sondeo, tp_sondeo, parcel_prof)

# Calcule los componentes de corte a granel y luego la magnitud
ubshr, vbshr = mpcalc.bulk_shear(p, u, v, depth=6 * units.km)
bshear = (mpcalc.wind_speed(ubshr, vbshr)).to('m/s')

# Create a new figure
fig = plt.figure(figsize=(16, 14))
skew = SkewT(fig, rotation=45)

# Trazar los datos usando funciones de trazado normales, en este caso usando
# escala logarítmica en Y, según lo dictado por el gráfico meteorológico típico
skew.plot(p_sondeo, t_sondeo, 'r', linewidth=2)
skew.plot(p_sondeo, tp_sondeo, 'g', linewidth=2)
skew.plot_barbs(p_sondeo, u_sondeo, v_sondeo)
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-40, 40)

# Trazar la temperatura LCL como un punto negro
skew.plot(lcl_presion, lcl_temperatura, 'ko', markerfacecolor='black')

# Trace el perfil de la parcela como una línea negra
skew.plot(p_sondeo, parcel_prof, 'k', linewidth=2)

# Zonas de sombra de CAPE y CIN
skew.shade_cin(p_sondeo, t_sondeo, parcel_prof, tp_sondeo)
skew.shade_cape(p_sondeo, t_sondeo, parcel_prof)

# Plot a zero degree isotherm
skew.ax.axvline(0, color='g', linestyle='--', linewidth=3)

# Agregue las líneas especiales relevantes
skew.plot_dry_adiabats(t0=np.arange(243, 533, 10) * units.K,
                       alpha=0.25, color='orangered')
# skew.plot_moist_adiabats(t0=np.arange(243, 400, 5) * units.K,
#                          alpha=0.25, color='tab:green')

# skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines(linestyle='dotted', color='tab:blue')

# Add some descriptive titles
plt.title('Sondeo: {} '.format('Aeropuerto International Carrasco'), loc='left')
plt.title(' Tiempo: {}'.format(dato['fecha'][0] + ds.valid_time.dt.strftime(' %HUTC').values ), loc='center')

# Mostrar los índices de sondeo
plt.text(0.94, 0.89, 'Índices de Sondeo', transform=fig.transFigure,
         ha='center', va='center', fontsize=12, color='blue')
plt.text(0.94, 0.87, f'Cape: {np.round(cape,1)}', transform=fig.transFigure,
         ha='center', va='center', fontsize=12, color='blue')
plt.text(0.94, 0.85, f'CIN: {np.round(cin,1)}', transform=fig.transFigure,
         ha='center', va='center', fontsize=12, color='blue')
plt.text(0.95, 0.83, f'Lift_Index: {np.round(lift_index,1)}', transform=fig.transFigure,
         ha='center', va='center', fontsize=12, color='blue')
plt.text(0.94, 0.81, f'KIndex: {np.round(kindex,1)}', transform=fig.transFigure,
         ha='center', va='center', fontsize=12, color='blue')
plt.text(0.94, 0.79, f'Bulk_Shear: {np.round(bshear,1)}', transform=fig.transFigure,
         ha='center', va='center', fontsize=12, color='blue')

# Show the plot
plt.show()
