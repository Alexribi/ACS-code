from sgp4.api import Satrec, jday
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pygeomag import GeoMag #Valida de 2026 até 2030

def field_calc(s, t, start_date, T, dt):
    satellite = Satrec.twoline2rv(s, t)
    delta_t = dt
    num_points = int(T / dt)

    times, east_vals, north_vals, down_vals = [], [], [], []
    lats, lons, alts = [], [], []

    # Inicializa o calculador moderno (já vem com WMM2025 nativo)
    geo_mag = GeoMag()

    for i in range(num_points):
        current_time = start_date + datetime.timedelta(seconds=i * delta_t)
        jd, fr = jday(
            current_time.year, current_time.month, current_time.day,
            current_time.hour, current_time.minute,
            current_time.second + current_time.microsecond / 1e6
        )

        e, r, v = satellite.sgp4(jd, fr)
        if e == 0:
            x, y, z = r
            norm_r = np.linalg.norm(r)
            lat = np.degrees(np.arcsin(z / norm_r))
            lon = np.degrees(np.arctan2(y, x))
            alt_km = norm_r - 6371 # Raio médio da Terra

            # pygeomag exige o tempo em Ano Decimal (ex: 2026.15)
            day_of_year = current_time.timetuple().tm_yday
            decimal_year = current_time.year + (day_of_year / 365.25)

            # Calcula o campo magnético com precisão de ponta
            B = geo_mag.calculate(glat=lat, glon=lon, alt=alt_km, time=decimal_year)

            # pygeomag retorna: x (Norte), y (Leste), z (Para baixo) em nanoTeslas
            north_vals.append(B.x * 1e-9) #Tesla
            east_vals.append(B.y * 1e-9) #Tesla
            down_vals.append(B.z * 1e-9) #Tesla

            times.append(current_time)
            lats.append(lat)
            lons.append(lon)
            alts.append(alt_km)
        else:
            print(f"Erro SGP4 em {current_time}: código {e}")

    return times, east_vals, north_vals, down_vals, lats, lons, alts





