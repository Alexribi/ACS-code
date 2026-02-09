import numpy as np
import matplotlib.pyplot as plt
from quaternion import simulate_attitude
from fieldcalc import field_calc
import datetime
from scipy.interpolate import interp1d
import transformations




###################################################### 1. Simulação do Campo Magnético e Atitude do Satélite ######################################################

# TLE do satélite
s = '1 99999U 26092A   26092.50000000  .00000000  00000-0  00000-0 0  9990'
t = '2 99999 040.0000 174.9269 0000000 025.2232 002.0151 15.1096    01'

# Parâmetros da simulação
num_points = 5400  # quantidade de pontos na órbita
start_date = datetime.datetime(2026, 10, 9, 12, 0, 0)

# ===== 1. Simulação do Campo Magnético =====
times, east, north, down, lats, lons, alts = field_calc(s, t, num_points, start_date)

# Plot dos componentes originais (opcional)
plt.figure(figsize=(12, 6))
plt.plot(times, east, label='East (G)', marker='o')
plt.plot(times, north, label='North (G)', marker='s')
plt.plot(times, down, label='Down (G)', marker='^')
plt.title('Componentes do Campo Magnético (WMM2020)')
plt.xlabel('Tempo')
plt.ylabel('Intensidade (Gauss)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


############################################################################################################################################





########################################################### 2. Conversão para Sistema ECI ###########################################################
# Calcular JD para cada tempo
jds = []
for t in times:
    jd, fr = transformations.jday(t.year, t.month, t.day, 
                                 t.hour, t.minute, t.second + t.microsecond/1e6)
    jds.append(jd + fr)

# Converter campo magnético de NED para ECI
mag_eci = []
mag_H = []
for i in range(len(times)):
    # Monta vetor NED [North, East, Down]
    v_ned = np.array([north[i], east[i], down[i]])
    
    # Converte NED → ECEF → ECI
    v_ecef = transformations.ned_to_ecef(lats[i], lons[i], v_ned)
    v_eci = transformations.ecef_to_eci(v_ecef, jds[i])
    mag_eci.append(v_eci)
    
mag_eci = np.array(mag_eci)


############################################################################################################################################




################################################################# 3. Processamento Conjunto #########################################################


# Perídodo de simulação e passo de tempo
time_hist = []
dt = 0.01                             # Passo de tempo [s]
T = 2500                             # Duração total [s] 
steps = int(T/dt)                # Número de passos de tempo

for i in range(steps):
    t = (i * dt)
    time_hist.append(t)
time_sim = np.array(time_hist)


# Converter tempos orbitais para segundos
t0 = start_date
time_sec = [(t - t0).total_seconds() for t in times]


# Repetir os dados de campo magnético N vezes para cobrir toda simulação
num_repetitions = int(T / time_sec[-1]) + 1  # Número de períodos orbitais
time_sec_extended = []
mag_eci_extended = []


for rep in range(num_repetitions):
    offset = rep * time_sec[-1]  # Deslocamento de tempo para cada repetição
    time_sec_extended.extend([t + offset for t in time_sec])
    mag_eci_extended.extend(mag_eci)


time_sec_extended = np.array(time_sec_extended)
mag_eci_extended = np.array(mag_eci_extended)


# Interpolar campo magnético ECI para os tempos da simulação de atitude
interp_x = interp1d(time_sec_extended, mag_eci_extended[:, 0], kind='linear')
interp_y = interp1d(time_sec_extended, mag_eci_extended[:, 1], kind='linear')
interp_z = interp1d(time_sec_extended, mag_eci_extended[:, 2], kind='linear')


# Obter valores interpolados
mag_x = interp_x(time_sim)
mag_y = interp_y(time_sim)
mag_z = interp_z(time_sim)
#############################################################################################################################################




################################################################### 4. Simulação de Atitude ###########################################################

# Configurações e condições de contorno do satélite

I = [0.00182, 0.00185, 0.00220]    # Momentos de inércia [kg*m²]
omega0 = [np.pi/30, np.pi/60, np.pi/45]                    # Velocidade angular inicial [rad/s]
q0 = [1, 0, 0, 0] 	      	          # Quaternion inicial


# Executa simulação de atitude
omega_hist, quat_hist, B_body, H_body = simulate_attitude(mag_x, mag_y, mag_z, I, omega0, q0, dt, steps)



# Plot dos quaternions (opcional)
plt.figure(figsize=(10, 6))
plt.plot(time_sim, quat_hist[:, 0], label='q0', color='blue')
plt.plot(time_sim, quat_hist[:, 1], label='q1', color='green')
plt.plot(time_sim, quat_hist[:, 2], label='q2', color='red')
plt.plot(time_sim, quat_hist[:, 3], label='q3', color='purple')

plt.xlabel('Tempo (s)')
plt.ylabel('Quaternions')
plt.title('Evolução da Atitude (Quaternion)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time_sim, omega_hist[:, 0], label='ωx', color='blue')
plt.plot(time_sim, omega_hist[:, 1], label='ωy', color='green')
plt.plot(time_sim, omega_hist[:, 2], label='ωz', color='red')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade Angular (rad/s)')
plt.title('Evolução da Atitude (Velocidade Angular)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

##############################################################################################################################################




#################################################### 5. Visualização dos Resultados #########################################################
plt.figure(figsize=(14, 10))

# Gráfico do campo magnético em ECI
plt.subplot(2, 1, 1)
plt.plot(time_sim, mag_x, 'r--', label='ECI X')
plt.plot(time_sim, mag_y, 'g--', label='ECI Y')
plt.plot(time_sim, mag_z, 'b--', label='ECI Z')
plt.title('Campo Magnético no Sistema Inercial (ECI)')
plt.ylabel('Intensidade (Gauss)')
plt.legend()
plt.grid(True)

# Gráfico do campo magnético em sistema de corpo
plt.subplot(2, 1, 2)
plt.plot(time_sim, B_body[:, 0], 'r-', label='Body X')
plt.plot(time_sim, B_body[:, 1], 'g-', label='Body Y')
plt.plot(time_sim, B_body[:, 2], 'b-', label='Body Z')
plt.title('Campo Magnético no Sistema de Corpo do Satélite')
plt.xlabel('Tempo (s)')
plt.ylabel('Intensidade (Gauss)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Gráfico 3D da trajetória do campo magnético no sistema de corpo
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(B_body[:, 0], B_body[:, 1], B_body[:, 2])
ax.set_xlabel('Eixo X (Gauss)')
ax.set_ylabel('Eixo Y (Gauss)')
ax.set_zlabel('Eixo Z (Gauss)')
ax.set_title('Trajetória do Campo Magnético no Sistema de Corpo')
plt.tight_layout()
plt.show()

# ===== 6. Análise Complementar =====
# Calcular magnitude total do campo magnético
mag_total = np.linalg.norm(B_body, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(time_hist, mag_total, 'm-', linewidth=2)
plt.title('Magnitude Total do Campo Magnético')
plt.xlabel('Tempo (s)')
plt.ylabel('Intensidade (Gauss)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcular variações por eixo
variation_x = np.max(B_body[:, 0]) - np.min(B_body[:, 0])
variation_y = np.max(B_body[:, 1]) - np.min(B_body[:, 1])
variation_z = np.max(B_body[:, 2]) - np.min(B_body[:, 2])

print("\nAnálise do Campo Magnético no Sistema de Corpo:")
print(f"Variação no eixo X: {variation_x:.6f} Gauss")
print(f"Variação no eixo Y: {variation_y:.6f} Gauss")
print(f"Variação no eixo Z: {variation_z:.6f} Gauss")
print(f"Média da magnitude: {np.mean(mag_total):.6f} Gauss")
print(f"Variação total: {np.max(mag_total) - np.min(mag_total):.6f} Gauss")


#############################################################################################################################################