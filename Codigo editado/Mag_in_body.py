import numpy as np
import transformations

 
 
 
#Utilizar os valores dos vetores mag_x, mag_y, mag_z usando [i]
def mag_in_body(mag_x, mag_y, mag_z, q):

       # Obter matriz de rotação do quaternion
       R = transformations.quat_to_rot_matrix(q)
        
        # Vetor campo magnético em ECI
       B_eci = np.array([mag_x, mag_y, mag_z])
       H_eci = B_eci / (4 * np.pi * 1e-7)  # Convertendo B (Tesla) para H (A/m)
        
        # Rotacionar para sistema de corpo
       B_body = R @ B_eci
       H_body = R @ H_eci

       return B_body, H_body


#Utilizar a lista H_body usando [i]
def dH_dt(H_body_now, H_body_last, dt):
    return (H_body_now - H_body_last) / dt
