import numpy as np
import Mag_in_body as mib
from ACS import Magbody
from tqdm import tqdm

 
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
    state = np.concatenate((np.array(q0, dtype=float), np.array(omega0, dtype=float)))



    # Inicializa o sistema de controle ACS
    acs = Magbody()
    acs.add_hysteresis_rod(coercivity=1.59, remanence=0.35,  saturation_field=0.73, volume=5*(10**-9), direction=np.array([1,0,0], dtype=np.float64)) #coercividade em Ampere/meter, remanência em Tesla, campo de saturação em Tesla, volume em m^3 e direção em unidade vetorial
    acs.add_hysteresis_rod(coercivity=1.59, remanence=0.35,  saturation_field=0.73, volume=7*(10**-8), direction=np.array([0,1,0], dtype=np.float64)) #coercividade em Ampere/meter, remanência em Tesla, campo de saturação em Tesla, volume em m^3 e direção em unidade vetorial
    acs.add_hysteresis_rod(coercivity=1.59, remanence=0.35,  saturation_field=0.73, volume=7*(10**-8), direction=np.array([0,0,1], dtype=np.float64)) #coercividade em Ampere/meter, remanência em Tesla, campo de saturação em Tesla, volume em m^3 e direção em unidade vetorial
    acs.add_permanent_magnet(remanence=1.28, volume=(0.7*(4*np.pi*(10**-7))/1.28), direction=np.array([1,0,0], dtype=np.float64)) #remanência em Tesla, volume em m^3 e direção em unidade vetorial
 
 
    omega_hist = []
    quat_hist = []
    B_body_hist = []
    H_body_hist = []
 

    """
    # Funções auxiliares para cálculo das derivadas
    def domega_dt(omega_vec, M):
        ωx, ωy, ωz = omega_vec
        dx = (M[0] - (ωy * ωz * (Izz - Iyy))) / Ixx
        dy = (M[1] - (ωz * ωx * (Ixx - Izz))) / Iyy
        dz = (M[2] - (ωx * ωy * (Iyy - Ixx))) / Izz
        return np.array([dx, dy, dz])
   
    def dq_dt(omega_vec, q_vec):
        ωx, ωy, ωz = omega_vec
        Omega_prime = np.array([
            [ 0,    -ωx,  -ωy,  -ωz],
            [ ωx,   0,   ωz,  -ωy],
            [ ωy,  -ωz,   0,   ωx],
            [ ωz,   ωy,  -ωx,   0 ]
        ])
        return 0.5 * Omega_prime @ q_vec
    """   
 
 

    # --- FUNÇÃO DE DERIVADA ACOPLADA ---
    def state_derivative(current_state, B_ECI_step):
        q_curr = current_state[0:4]
        w_curr = current_state[4:7]
        q_curr = q_curr / np.linalg.norm(q_curr)

        # 1. Rotacionar campo para o corpo
        b_body, h_body = mib.mag_in_body(B_ECI_step[0], B_ECI_step[1], B_ECI_step[2], q_curr)

        # 2. Derivada cinemática do campo magnético devido à rotação do satélite (Essencial!)
        dH_dt_body = np.cross(h_body, w_curr)

        # 3. Torque Magnético (Lê a histerese para prever o RK4, mas NÃO salva na memória: update_state=False)
        m_total = acs.magnetic_moment(dH_dt_body, h_body, dt, update_state=False)
        M = acs.torque(b_body, m_total)

        # 4. Derivadas Cinemáticas (Quatérnio)
        wx, wy, wz = w_curr
        Omega_prime = np.array([
            [ 0,    -wx,  -wy,  -wz],
            [ wx,   0,   wz,  -wy],
            [ wy,  -wz,   0,   wx],
            [ wz,   wy,  -wx,   0 ]
        ])
        dq = 0.5 * Omega_prime @ q_curr

        # 5. Derivadas Dinâmicas (Equações de Euler)
        dwx = (M[0] - (wy * wz * (Izz - Iyy))) / Ixx
        dwy = (M[1] - (wz * wx * (Ixx - Izz))) / Iyy
        dwz = (M[2] - (wx * wy * (Iyy - Ixx))) / Izz
        dw = np.array([dwx, dwy, dwz])

        return np.concatenate((dq, dw))
 
    

 
    # --- LOOP PRINCIPAL DE SIMULAÇÃO ---
    for i in tqdm(range(steps), desc="Simulando atitude"):
 

        # Vetor do campo inercial deste passo de tempo exato
        b_inercial = np.array([mag_x[i], mag_y[i], mag_z[i]])

        # Extrai variáveis limpas para os históricos
        q_now = state[0:4] / np.linalg.norm(state[0:4])
        w_now = state[4:7]
        
        omega_hist.append(w_now.copy())
        quat_hist.append(q_now.copy())

        # Usa o estado validado para calcular o B_body principal do passo
        b_body_now, h_body_now = mib.mag_in_body(b_inercial[0], b_inercial[1], b_inercial[2], q_now)
        B_body_hist.append(b_body_now.copy())
        H_body_hist.append(h_body_now.copy())

        # --- INTEGRAÇÃO RK4 UNIFICADA (Sem injeção de energia) ---
        k1 = state_derivative(state, b_inercial)
        k2 = state_derivative(state + 0.5 * dt * k1, b_inercial)
        k3 = state_derivative(state + 0.5 * dt * k2, b_inercial)
        k4 = state_derivative(state + dt * k3, b_inercial)

        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Garante a normalização final do quatérnio na memória principal
        state[0:4] = state[0:4] / np.linalg.norm(state[0:4])

        # --- O FREIO REAL: Atualização da Memória da Histerese ---
        # Somente no final do loop o sistema retém a indução dissipada.
        dH_dt_now = np.cross(h_body_now, w_now)
        acs.magnetic_moment(dH_dt_now, h_body_now, dt, update_state=True)

    return np.array(omega_hist), np.array(quat_hist), np.array(B_body_hist), np.array(H_body_hist)





    #Primeira versão com RK4 desacoplado
    """
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
        m = acs.magnetic_moment(dH, h_body, dt)
 
 
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
    """
 
    #return np.array(omega_hist), np.array(quat_hist), np.array(B_body_hist), np.array(H_body_hist)


