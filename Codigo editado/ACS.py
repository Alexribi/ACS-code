import numpy as np
from scipy.spatial.transform import Rotation





class Magbody:    #inicia a classe Magbody
    def __init__(self): #inicializa a classe Magbody
        self.hysteresis_rods = []
        self.permanent_magnets = []

    class HysteresisRod: #inicia a classe HysteresisRod
        def __init__(self, coercivity, remanence, saturation_field, volume, direction):
            self.hysteresis_rods = []  # Inicializa a lista de hastes de histerese

           
            self.Hc = coercivity #Ampere/meter
            self.Br = remanence #Tesla
            self.Bs = saturation_field #Tesla
            self.B = 0 #Tesla
            self.k = (1/self.Hc)*np.tan(0.5*np.pi*self.Br/self.Bs)
            self.volume = volume #m^3
            self.direction = direction #unit vector
            self.space_direction = direction #unit vector
            
            self.magnetic_field_time_derivative = np.zeros(3) #Ampere/meter/second

            # Constantes empíricas do Algoritmo de Flatley & Henretty (Pág 41 TCC)
            self.q0 = 0.085
            self.p = 4.75
            

            
        
        # atualiza o vetor campo magnético do bastão de histerese
        def update_magnetic_field(self, dH, h_body, dt, update_state=False): 

            # Projeção escalar do campo e da derivada no eixo da barra
            H_proj = h_body @ self.direction
            dH_dt_proj = dH @ self.direction




            # self.B já é Tesla, usamos diretamente com a trava de segurança
            B_safe = np.clip(self.B, -self.Bs * 0.999, self.Bs * 0.999)

            # Usamos o self.state para achar o HL
            self.state = np.tan(0.5 * np.pi * B_safe / self.Bs)
            HL = (self.state / self.k) - self.Hc

            # Algoritmo de Flatley
            cos_term = np.cos(0.5 * np.pi * B_safe / self.Bs)
            numerator = 2 * self.Bs * np.tan(0.5 * np.pi * self.Br / self.Bs)
            denominator = np.pi * self.Hc
            B_prime = (numerator / denominator) * (cos_term ** 2)


            if dH_dt_proj > 0:
                Gamma = (H_proj - HL) / (2 * self.Hc)
            else:
                Gamma = 1.0 - ((H_proj - HL) / (2 * self.Hc))


            Gamma = np.clip(Gamma, 0.0, 1.0)
            q_val = self.q0 + (1 - self.q0) * (Gamma ** self.p)

            dB_dH = q_val * B_prime
            dB_dt = dB_dH * dH_dt_proj



            # Calcula o B instantâneo deste micro-passo do RK4
            B_new = B_safe + (dB_dt * dt)
            self.B_new = np.clip(B_new, -self.Bs, self.Bs)

            # Apenas salva na memória física se for o fim do passo do RK4
            if update_state:
                self.B = self.B_new
                # Atualiza o state final para a próxima iteração
                self.state = np.tan(0.5 * np.pi * self.B / self.Bs)


            # RETORNA o valor para uso no cálculo do momento magnético
            return B_new


            '''
            if (dH_dt_proj) > 0:
                self.B = (self.Bs*(2/np.pi)*np.arctan(self.k*(H_proj-self.Hc))) #Tesla     
            else:
                self.B = (self.Bs*(2/np.pi)*np.arctan(self.k*(H_proj+self.Hc))) #Tesla
            
            '''
            
            

            


    class PermanentMagnet: #inicia a classe PermanentMagnet
        def __init__(self, remanence, volume, direction): #inicializa a classe PermanentMagnet
            self.Br = remanence #Tesla
            self.volume = volume #m^3
            self.direction = direction #unit vector
            self.space_direction = direction #unit vector
            self.B = self.volume*self.Br #Tesla







    #Adiciona bastões de histerese à lista de bastões de histerese
    def add_hysteresis_rod(self, coercivity, remanence, saturation_field, volume, direction): #adiciona um bastão de histerese à lista de bastões de histerese
        rod = self.HysteresisRod(coercivity, remanence, saturation_field, volume, direction)
        self.hysteresis_rods.append(rod)

    #Adiciona ímãs permanentes à lista de ímãs permanentes
    def add_permanent_magnet(self, remanence, volume, direction): #adiciona um ímã permanente à lista de ímãs permanentes
        pm = self.PermanentMagnet(remanence, volume, direction)
        self.permanent_magnets.append(pm)





    def magnetic_moment(self, dH, h_body, dt, update_state=False):
        total_magnetic_moment = np.zeros(3)
        mu0 = 4 * np.pi * (10**-7)
        
        for rod in self.hysteresis_rods:
            if (dH @ rod.direction) != 0:
                B_new = rod.update_magnetic_field(dH, h_body, dt, update_state)
            else:
                B_new = rod.B  # Se a derivada for zero, mantemos o valor atual de B
            total_magnetic_moment += B_new * rod.direction * rod.volume/mu0  #A*m^2
           

        for pm in self.permanent_magnets:
            total_magnetic_moment += pm.B * pm.direction/mu0 #A*m^2
        return total_magnetic_moment
    

    #Utilizar a lista de B_body usando [i]
    def torque(self, B_body, m): 
        return np.cross(m, B_body)


