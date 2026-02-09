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
            self.volume = volume #m^3
            self.direction = direction #unit vector
            self.space_direction = direction #unit vector
            self.k = (1/self.Hc)*np.tan(0.5*np.pi*self.Br/self.Bs)
            
            self.magnetic_field_time_derivative = np.zeros(3) #Ampere/meter/second
            

            
        
        # atualiza o vetor campo magnético do bastão de histerese
        def update_magnetic_field(self, dH, h_body): 
            
            if (dH @ self.direction) > 0:
                self.B = (self.Bs*(2/np.pi)*np.arctan(self.k*((h_body @ self.direction)-self.Hc)))*self.volume #Tesla     
            else:
                self.B = (self.Bs*(2/np.pi)*np.arctan(self.k*((h_body @ self.direction)+self.Hc)))*self.volume #Tesla
            self.state = np.tan(0.5*np.pi*self.B/self.Bs)





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





    def magnetic_moment(self, dH, h_body):
        total_magnetic_moment = np.zeros(3)

        for rod in self.hysteresis_rods:
            if (dH @ rod.direction) == 0:
                rod.B = rod.B
            else:
                rod.update_magnetic_field(dH, h_body)
            total_magnetic_moment += rod.B * rod.direction

        for pm in self.permanent_magnets:
            total_magnetic_moment += pm.B * pm.direction
        return total_magnetic_moment
    

    #Utilizar a lista de B_body usando [i]
    def torque(self, B_body, m): 
        return np.cross(m, B_body)
