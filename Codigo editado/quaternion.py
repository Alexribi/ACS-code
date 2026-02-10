import numpy as np
import Mag_in_body as mib
from ACS import Magbody
 
def simulate_attitude(mag_x, mag_y, mag_z, I, omega0, q0, dt, steps):
    """
    Simula a evolução da atitude de um corpo rígido usando quaternions com integração RK4.
 
    Parâmetros:
    - I: array-like, [Ixx, Iyy, Izz] (kg*m²)
    - omega0: array-like, [ωx, ωy, ωz] inicial (rad/s)
    - q0: array-like, quaternion inicial [q0, q1, q2, q3]
    - M: array-like, torques aplicados [Mx, My, Mz]
    - dt: float, passo de tempo (s)
    - T: float, duração total (s)
 
    Retorna:
    - time_hist: array com os instantes de tempo
    - omega_hist: histórico das velocidades angulares
    - quat_hist: histórico dos quaternions
    """
 
 
 
    Ixx, Iyy, Izz = I
    omega = np.array(omega0, dtype=float)
    q = np.array(q0, dtype=float)
 
    omega_hist = []
    quat_hist = []
    B_body = []
    H_body = []
 
 
    # Funções auxiliares para cálculo das derivadas
    def domega_dt(omega_vec, M):
        ωz, ωy, ωx = omega_vec
        dx = (M[0] - (ωy * ωz * (Izz - Iyy))) / Ixx
        dy = (M[1] - (ωz * ωx * (Ixx - Izz))) / Iyy
        dz = (M[2] - (ωx * ωy * (Iyy - Ixx))) / Izz
        return np.array([dx, dy, dz])
   
    def dq_dt(omega_vec, q_vec):
        ωx, ωy, ωz = omega_vec
        Omega_prime = np.array([
            [ 0,    ωz,  -ωy,  ωx],
            [-ωz,   0,   ωx,  ωy],
            [ ωy, -ωx,   0,   ωz],
            [-ωx, -ωy, -ωz,   0 ]
        ])
        return 0.5 * Omega_prime @ q_vec
 
 
 
 
    # Inicializa o sistema de controle ACS
    acs = Magbody()
    acs.add_hysteresis_rod(coercivity=1.59, remanence=0.35,  saturation_field=0.73, volume=5*(10**-9), direction=np.array([1,0,0], dtype=np.float64)) #coercividade em Ampere/meter, remanência em Tesla, campo de saturação em Tesla, volume em m^3 e direção em unidade vetorial
    acs.add_hysteresis_rod(coercivity=1.59, remanence=0.35,  saturation_field=0.73, volume=7*(10**-8), direction=np.array([0,1,0], dtype=np.float64)) #coercividade em Ampere/meter, remanência em Tesla, campo de saturação em Tesla, volume em m^3 e direção em unidade vetorial
    acs.add_hysteresis_rod(coercivity=1.59, remanence=0.35,  saturation_field=0.73, volume=7*(10**-8), direction=np.array([0,0,1], dtype=np.float64)) #coercividade em Ampere/meter, remanência em Tesla, campo de saturação em Tesla, volume em m^3 e direção em unidade vetorial
    acs.add_permanent_magnet(remanence=1.28, volume=(0.7), direction=np.array([1,0,0], dtype=np.float64)) #remanência em Tesla, volume em m^3 e direção em unidade vetorial
 
 
 
    for i in range(steps):
 
        # Armazenar históricos de omega e quaternion
        omega_hist.append(omega.copy())
        quat_hist.append(q.copy())
 
 
        # Obter campo magnético no sistema de corpo
        b_body, h_body = mib.mag_in_body(mag_x[i], mag_y[i], mag_z[i], q)
 
        B_body.append(b_body.copy())
        H_body.append(h_body.copy())
 
 
        # Calcular derivada temporal do campo magnético no sistema de corpo
        if i > 0:
            dH = mib.dH_dt(H_body[i], H_body[i-1], dt)
        else:
            dH = np.zeros(3)
       
 
        # Atualizar estado do ACS
        m = acs.magnetic_moment(dH, h_body)
 
 
        # Calcular torque magnético
        M = acs.torque(b_body, m)
 
       
        # Integração RK4 para omega
        k1_omega = domega_dt(omega, M)
        k2_omega = domega_dt(omega + 0.5 * dt * k1_omega, M)
        k3_omega = domega_dt(omega + 0.5 * dt * k2_omega, M)
        k4_omega = domega_dt(omega + dt * k3_omega, M)
        omega += (dt / 6.0) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
 
 
        # Integração RK4 para quaternion
        k1_q = dq_dt(omega, q)
        k2_q = dq_dt(omega, q + 0.5 * dt * k1_q)
        k3_q = dq_dt(omega, q + 0.5 * dt * k2_q)
        k4_q = dq_dt(omega, q + dt * k3_q)
        q += (dt / 6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
           
       
        # Normalização do quaternion
        q /= np.linalg.norm(q)
 
 
    return np.array(omega_hist), np.array(quat_hist), np.array(B_body), np.array(H_body)
