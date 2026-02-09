from sgp4.api import Satrec, jday
import numpy as np
import matplotlib.pyplot as plt
import datetime
import geomag

def field_calc(s, t, num_points, start_date):
    satellite = Satrec.twoline2rv(s, t)
    delta_t = 5400 / num_points

    times, east_vals, north_vals, down_vals = [], [], [], []
    lats, lons, alts = [], [], []

    g = geomag.geomag.GeoMag()

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
            alt_km = norm_r - 6371

            B = g.GeoMag(lat, lon, alt_km)

            north_vals.append(B.bx * 1e-5) #Gauss
            east_vals.append(B.by * 1e-5) #Gauss
            down_vals.append(B.bz * 1e-5) #Gauss

            times.append(current_time)
            lats.append(lat)
            lons.append(lon)
            alts.append(alt_km)
        else:
            print(f"Erro SGP4 em {current_time}: código {e}")

    return times, east_vals, north_vals, down_vals, lats, lons, alts

