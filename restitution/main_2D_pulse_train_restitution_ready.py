from numpy import exp
from numpy import log
import numpy as np
from numpy import append
import time
import matplotlib.pyplot as plt
from numba import njit
from mpi4py import MPI

###############################################################################
'''
    INITIALIZE PARALLELIZATION PARAMETERS
'''
###############################################################################

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
size = comm.Get_size()
print('MPI size:',size)

##### INITIALIZE DOMAIN PARAMETERS
dx = 0.022
dy = dx
xspace = 384 
yspace = 8      #THIS IS FOR RESTITUTION

mypoints = int(xspace/size)

rightProc = myrank+1
leftProc = myrank-1

print('Welcome to the 2D simulation. This is process ',myrank)


####################################################################################
'''
    INITIALIZE CONSTANTS
'''
####################################################################################
#Time constants
Dt=0.0 #ms
# Universal Constants
R = 8.3143 #Gas ctant. (J/K/mol)
T = 310 #Temp. (K)
F = 96.4867 #Faraday ctant. (C/mmol)
#Cell properties
D=0.003036 #cm^2/ms                                                                       DIFFUSION COEFFICIENT
D = 0.0018
D = 0.0014
D = 0.00126
Cm = 100
Vol_cell = 20100 #Cell Volume (mum^3)
Vol_i = 13668 #Intracellular Volume (mum^3)
V_up = 1109.52 #SR uptake compartment volume (mum^3)
V_rel = 96.48 #SR release compartment volume (mum^3)
#Concentrations (ion_concentration_inside/outside)
K_con_o = 5.4 #Extracellular K+ concentration (mM)
Na_con_o = 140 #Extracellular Na+ concentration (mM)
Cl_con_o = 132 #mM
Ca_con_o = 1.8 #Extracellular Ca2+ concentration (mM)
Ca_con_up_max = 15 #Maximal Ca concentration in uptake compartment (mM)
Cmdn_con_max = 0.05 #Total calmodulin concentration in myoplasm (mM)
Trpn_con_max = 0.07 #Total troponin concentration in myoplasm (mM)
Csqn_con_max = 10 #Total calsequestrin concentration in SR release compartment (mM)
#Conductances
g_Na = 13.989973151947163 #Maximal I_Na conductance (nS/pF)
g_K1 = 0.08217804978026715 #Maximal I_K1 conductance (nS/pF)
g_to2 = 0.15730848000000008 #Maximal I_to2 conductance (nS/pF)
g_Kur_amplitude = 0.4553855613824757
g_Kr = 0.017297279999999998 #Maximal I_Kr conductance (nS/pF)
g_Ks = 0.0594 #Maximal I_Ks conductance (nS/pF)
g_Ca_L = 0.06574044338878168 #Maximal I_Ca_L conductance (nS/pF)
g_b_Ca = 0.00113 #Maximal I_b_Ca conductance (nS/pF)
g_b_Na = 0.000674 #Maximal I_b_Na conductance (nS/pF)
#Maximal Currents
I_NaK_max = 0.94935256416 #Maximal I_NaK (pA/pF)
I_NaCa_max = 2304.0 #Maximal I_NaCa (pA/pF) ##########ATTENTION UNITS (Compare table with page H308)
I_p_Ca_max = 0.275 #Maximal I_p_Ca (pA/pF)
I_up_max = 0.005 #Maximal I_up_max mM/ms
#Scaling factors
K_Q_10 = 3 #Temperature scaling factor for I_Kur and I_to2 kinetics
gamma = 0.35 #Voltage dependence parameter for I_NaCa
#Saturation constants
K_m_Na_i = 10 #Na_con_i half-saturation constant for I_NaK (mM)
K_m_K_o = 1.5 #K_con_o half-saturation constant for I_NaK (mM)
K_m_Na = 87.5 #Na_con_o half-saturation constant for I_NaCa (mM)
K_m_Ca = 1.38 #Ca_con_o half-saturation constant for I_NaCa (mM)
k_sat = 0.1 #Saturation factor for I_NaCa
k_rel = 30 #Maximal release rate for I_rel (30 ms^-1)
K_up = 0.00092 #Ca_con_i half-saturation constant for I_up (mM)
K_m_Cmdn = 0.00238 #Ca_con_i half-saturation constant for calmodulin (mM)
K_m_Trpn = 0.0005 #Ca_con_i half-saturation constant for troponin (mM)
K_m_Csqn = 0.8 #Ca_con_rel half-saturation constant for I_up (mM)

####################################################################################
'''
    BUFFER CONCENTRATIONS
'''
####################################################################################

@njit(fastmath=True)
def Ca_con_Cmdn_func(Cmdn_con_max,Ca_con_i,K_m_Cmdn):
    Ca_con_Cmdn = Cmdn_con_max * Ca_con_i/(Ca_con_i+K_m_Cmdn)
    return Ca_con_Cmdn
@njit(fastmath=True)
def Ca_con_Trpn_func(Trpn_con_max,Ca_con_i,K_m_Trpn):
    Ca_con_Trpn = Trpn_con_max * Ca_con_i/(Ca_con_i+K_m_Trpn)
    return Ca_con_Trpn
@njit(fastmath=True)
def Ca_con_Csqn_func(Csqn_con_max,Ca_con_rel,K_m_Csqn):
    Ca_con_Csqn = Csqn_con_max * Ca_con_rel/(Ca_con_rel+K_m_Csqn)
    return Ca_con_Csqn

@njit(fastmath=True)
def get_concentrations(Cmdn_con_max,Ca_con_i,K_m_Cmdn,
                       Trpn_con_max,K_m_Trpn,
                       Csqn_con_max,Ca_con_rel,K_m_Csqn):
    Ca_con_Cmdn = Ca_con_Cmdn_func(Cmdn_con_max,Ca_con_i,K_m_Cmdn)
    Ca_con_Trpn = Ca_con_Trpn_func(Trpn_con_max,Ca_con_i,K_m_Trpn)
    Ca_con_Csqn = Ca_con_Csqn_func(Csqn_con_max,Ca_con_rel,K_m_Csqn)
    return Ca_con_Cmdn,Ca_con_Trpn,Ca_con_Csqn

####################################################################################
'''
    DIFFERENTIAL EQUATIONS FOR: Na, K, Ca
'''
####################################################################################

@njit(fastmath=True)
def dNa_con_i(I_NaK,I_NaCa,I_b_Na,I_Na,F,Vol_i):
    dNa = (-3*I_NaK-3*I_NaCa-I_b_Na-I_Na)/(F*Vol_i)
    return dNa *Cm ### *Cm for dymensional analysis
@njit(fastmath=True)
def dK_con_i(I_NaK,I_K1,I_to2,I_Kur,I_Kr,I_Ks,I_b_K,F,Vol_i):
    dK = (2*I_NaK-I_K1-I_to2-I_Kur-I_Kr-I_Ks-I_b_K)/(F*Vol_i)
    return dK *Cm ### *Cm for dimensional analysis
@njit(fastmath=True)
def dCa_con_i(B1,B2):
    dCa = B1/B2
    return dCa
@njit(fastmath=True)
def B1(I_NaCa,I_p_Ca,I_Ca_L,I_b_Ca,V_up,I_up_leak,I_up,I_rel,V_rel,F,Vol_i):
    B1 = (2*I_NaCa-I_p_Ca-I_Ca_L-I_b_Ca)/(2*F*Vol_i)*Cm + (V_up*(I_up_leak-I_up)+I_rel*V_rel)/Vol_i
    return B1  #*Cm for dimensional analysis, ONLY IN THE FIRST TERM
@njit(fastmath=True)
def B2(Trpn_con_max,K_m_Trpn,Cmdn_con_max,K_m_Cmdn,Ca_con_i):
    B2 = 1 + (Trpn_con_max*K_m_Trpn)/(Ca_con_i+K_m_Trpn)**2 + (Cmdn_con_max*K_m_Cmdn)/(Ca_con_i+K_m_Cmdn)**2
    return B2 #No need to *Cm because units are already in mM/ms
@njit(fastmath=True)
def dCa_con_up(I_up,I_up_leak,I_tr,V_rel,V_up):
    dCa = I_up-I_up_leak-I_tr*V_rel/V_up
    return dCa
@njit(fastmath=True)
def dCa_con_rel(I_tr,I_rel,Csqn_con_max,K_m_Csqn, Ca_con_rel):
    dCa = (I_tr-I_rel)*(1+(Csqn_con_max*K_m_Csqn)/(Ca_con_rel+K_m_Csqn)**2)**(-1)
    return dCa

@njit(fastmath=True)
def update_concentrations(dt,Na_con_i,K_con_i,Ca_con_i,Ca_con_up,Ca_con_rel,
                          I_NaK,I_NaCa,I_b_Na,I_Na,F,Vol_i,
                          I_K1,I_to2,I_Kur,I_Kr,I_Ks,I_b_K,
                          I_p_Ca,I_Ca_L,I_b_Ca,V_up,I_up_leak,I_up,I_rel,V_rel,
                          Trpn_con_max,K_m_Trpn,Cmdn_con_max,K_m_Cmdn,
                          I_tr,
                          Csqn_con_max,K_m_Csqn,
                          Cl_con_i,Cm):
    Na_con_i = Na_con_i + dNa_con_i(I_NaK,I_NaCa,I_b_Na,I_Na,F,Vol_i)*dt
    K_con_i = K_con_i + dK_con_i(I_NaK,I_K1,I_to2,I_Kur,I_Kr,I_Ks,I_b_K,F,Vol_i)*dt
    Ca_con_i = Ca_con_i + dCa_con_i(
        B1(I_NaCa,I_p_Ca,I_Ca_L,I_b_Ca,V_up,I_up_leak,I_up,I_rel,V_rel,F,Vol_i),
        B2(Trpn_con_max,K_m_Trpn,Cmdn_con_max,K_m_Cmdn,Ca_con_i))*dt
    Ca_con_up = Ca_con_up + dCa_con_up(I_up,I_up_leak,I_tr,V_rel,V_up)*dt
    Ca_con_rel = Ca_con_rel + dCa_con_rel(I_tr,I_rel,Csqn_con_max,K_m_Csqn,
                                          Ca_con_rel)*dt
    Cl_con_i = Cl_con_i+I_to2/(F*Vol_i)*Cm*dt
    return Na_con_i,K_con_i,Ca_con_i,Ca_con_up,Ca_con_rel,Cl_con_i

####################################################################################
'''
    POTENTIALS
'''
####################################################################################
z_Na = 1
@njit(fastmath=True)
def E_Na_func(R,T,z_Na,F,Na_con_o,Na_con_i):
    E_Na = R*T/(z_Na*F) * np.log(Na_con_o/Na_con_i)
    return E_Na
z_K = 1
@njit(fastmath=True)
def E_K_func(R,T,z_K,F,K_con_o,K_con_i):
    E_K = R*T/(z_K*F) * np.log(K_con_o/K_con_i)
    return E_K
z_Ca = 2
@njit(fastmath=True)
def E_Ca_func(R,T,z_Ca,F,Ca_con_o,Ca_con_i):
    E_Ca = R*T/(z_Ca*F) * np.log(Ca_con_o/Ca_con_i)
    return E_Ca

z_Cl = -1
@njit(fastmath=True)
def E_Cl_func(R,T,z_Cl,F,Cl_con_o,Cl_con_i):
    E_Cl = R*T/(z_Cl*F) * np.log(Cl_con_o/Cl_con_i)
    return E_Cl

@njit(fastmath=True)
def get_rest_potentials(R,T,z_Na,F,Na_con_o,Na_con_i,
                        z_K,K_con_o,K_con_i,
                        z_Ca,Ca_con_o,Ca_con_i,
                        z_Cl,Cl_con_o,Cl_con_i):
    E_Na = E_Na_func(R,T,z_Na,F,Na_con_o,Na_con_i)
    E_K = E_K_func(R,T,z_K,F,K_con_o,K_con_i)
    E_Ca = E_Ca_func(R,T,z_Ca,F,Ca_con_o,Ca_con_i)
    E_Cl = R*T/(z_Cl*F) * np.log(Cl_con_o/Cl_con_i)
    return E_Na, E_K, E_Ca, E_Cl

@njit(fastmath=True)
def dV_func(I_ion, I_st):
    dV = -(I_ion+I_st) #NO DIVISION BY Cm; ALREADY INCLUDED IN CURRENTS
    return dV

@njit(fastmath=True)
def update_potential(dt,V,I_ion,I_st,Cm):
    V = V + dV_func(I_ion,I_st)*dt
    return V

####################################################################################
'''
    CURRENTS
''' 
####################################################################################

########################
##### I_Na
########################
@njit(fastmath=True)
def I_Na_func(m, h, j, V, E_Na):
    global g_Na
    return g_Na*m**3*h*j*(V-E_Na)
@njit(fastmath=True)
def m(V,t,dt,m_0):
    alfa_m = (0.32*(V+47.13)/(1-np.exp(-0.1*(V+47.13))))
    beta_m = 0.08*np.exp(-V/11)
    m_inf = 1/(1+exp(-(V+43)/7.7))
    tau = 1/(alfa_m+beta_m)*1.7
    m = m_inf-(m_inf-m_0)*np.exp(-(dt)/tau)
    return m
@njit(fastmath=True)
def h(V,t,dt,h_0):
    bool_a = np.where(V<-40,1,0)
    bool_b = 1-bool_a
    alfa_h = 0.135*np.exp(-(V+80)/6.8)*bool_a+0*bool_b

    bool_a = np.where(V<-40,1,0)
    bool_b = 1-bool_a
    beta_h = 3.56*np.exp(0.079*V)+3.1*10**5*np.exp(0.35*V)*bool_a+((0.13*(1+np.exp(-(V+10.66)/11.1)))**(-1))*bool_b
    
    h_inf = 1/(1+exp(+(V+66.5)/4.4))
    tau = 1/(alfa_h+beta_h)*2
    h = h_inf-(h_inf-h_0)*np.exp(-(dt)/tau)
    return h
@njit(fastmath=True)
def j(V,t,dt,j_0):

    bool_a = np.where(V<-40,1,0)
    bool_b = 1-bool_a
    alfa_j = ((-127140*np.exp(0.2444*V)-3.474*10**(-5)*np.exp(-0.04391*V))*(V+37.78)/(1+np.exp(0.311*(V+79.23))))*bool_a+0*bool_b

    bool_a = np.where(V<-40,1,0)
    bool_b = 1-bool_a
    beta_j = (0.1212*np.exp(-0.01052*V)/(1+np.exp(-0.1378*(V+40.14))))*bool_a+bool_b*(0.3*np.exp(-2.535*10**(-7)*V)/(1+np.exp(-0.1*(V+32))))
 
    j_inf = 1/(1+exp(+(V+66.5)/4.1))
    tau = 1/(alfa_j+beta_j)*2
    j = j_inf-(j_inf-j_0)*np.exp(-(dt)/tau)
    return j

########################
##### I_K1
########################
@njit(fastmath=True)
def I_K1_func(V,E_K):
    global g_K1
    return g_K1*(V-E_K-5)/(1+np.exp(0.07*0.9*(V+70)))

########################
##### I_to2
########################
@njit(fastmath=True)
def I_to2_func(V,E_Cl,q_Ca):
    global g_to2
    
    I_to2 = q_Ca*(V-E_Cl)
    
    return g_to2*I_to2
@njit(fastmath=True)
def q_Ca(V,t,F_n_val,q_Ca_0):
    q_Ca_inf = 1-1/(1+((F_n_val)/(1.1e-10))**3)
    tau_qCa = 2
    q_Ca_val = q_Ca_inf+(q_Ca_0-q_Ca_inf)*np.exp(-dt/tau_qCa)
    return q_Ca_val


########################
##### I_Kur
########################

@njit(fastmath=True)
def g_Kur(V):
    return g_Kur_amplitude*(0.005+0.05/(1+np.exp(-(V-15)/(13))))
@njit(fastmath=True)
def I_Kur_func(g_Kur,u_a,u_if,u_is,V,E_K):
    return g_Kur*u_a**3*(0.25*u_if+0.75*u_is)*(V-E_K)
@njit(fastmath=True)
def u_a(V,t,K_Q_10,dt,u_a_0):
    alfa_ua = 0.65*(np.exp(-(V+10)/8.5)+np.exp(-(V-30)/59))**(-1)
    beta_ua = 0.65*(2.5+np.exp((V+82)/17))**(-1)
    u_a_inf = (1+np.exp(-(V+30.3)/9.6))**(-1)
    tau = 1/K_Q_10/(alfa_ua+beta_ua)
    u_a  = u_a_inf-(u_a_inf-u_a_0)*np.exp(-(dt)/tau)
    return u_a
@njit(fastmath=True)
def u_if(V,t,K_Q_10,dt,u_if_0):
    u_if_inf = (1+np.exp((V+17.358)/5.849))**-1
    tau = 400+1068*np.exp(-(V/50)**2)
    u_if = u_if_inf-(u_if_inf-u_if_0)*np.exp(-(dt)/tau)
    return u_if
@njit(fastmath=True)
def u_is(V,t,K_Q_10,dt,u_is_0):
    u_is_inf = (1+np.exp((V+17.358)/5.849))**-1
    tau = 2000+60000*np.exp(-((V+39.3)/30)**2)
    u_is = u_is_inf-(u_is_inf-u_is_0)*np.exp(-(dt)/tau)
    return u_is

########################
##### I_Kr
########################
@njit(fastmath=True)
def I_Kr_func(g_Kr,x_r,V,E_K):
    I_Kr = g_Kr*x_r*(V-E_K)/(1+np.exp((V-79.482516)/8.221714))
    return I_Kr
@njit(fastmath=True)
def x_r(V,t,dt,x_r_0):
    alfa_xr = (0.0003*(V+14.1)/(1-np.exp(-(V+14.1)/5)))
    beta_xr = (7.3898e-5*(V-3.3328)/(np.exp((V-3.3328)/5.1237)-1))
    x_r_inf = (1+np.exp(-(V-4.445095)/9.33047))**(-1)
    tau = 1/(alfa_xr+beta_xr)
    x_r = x_r_inf-(x_r_inf-x_r_0)*np.exp(-(dt)/tau)
    return x_r

########################
##### I_Ks
########################
@njit(fastmath=True)
def I_Ks_func(g_Ks,x_s,V,E_K):
    I_Ks = g_Ks*x_s**2*(V-E_K)
    return I_Ks
@njit(fastmath=True)
def x_s(V,t,dt,x_s_0):
    alfa_xs = 4e-5*(V-18.80816)/(1-np.exp(-(V-18.80816)/17))
    beta_xs = 3.5e-5*(V-18.80816)/(np.exp((V-18.80816)/9)-1)
    tau = 1/(alfa_xs+beta_xs)*1/2 #The 1/2 used in the model but not the fitting!!!
    x_s_inf = (1+np.exp(-(V-18.80816)/12.6475))**(-1/2) #1/2 used in the model, not the fit
    x_s = x_s_inf-(x_s_inf-x_s_0)*np.exp(-(dt)/tau)
    return x_s

########################
##### I_Ca_L
########################
@njit(fastmath=True)
def I_Ca_L_func(g_Ca_L, d, f, f_Ca, V):
    I_Ca_L = g_Ca_L*d*f*f_Ca*(V-65)
    return I_Ca_L
@njit(fastmath=True)
def d(V,t,dt,d_0):
    d_inf = 1/(1+np.exp(-(V+5)/8))
    tau = (1-np.exp(-(V+5)/6.24))/(0.035*(V+5)*(1+np.exp(-(V+5)/6.24))) ## ?? Should that 6.24 be an 8?? (in Luo-Rudy, terms from d_inf cancel into tau_d; shouldn't it be the same in here?)
    d = d_inf-(d_inf-d_0)*np.exp(-(dt)/tau)   
    return d
@njit(fastmath=True)
def f(V,t,dt,f_0):
    f_inf = 1/(1+np.exp((V+28)/6.9))
    tau = 9/(0.0197*np.exp(-0.0337**2*(V+5)**2)+0.02)
    f = f_inf-(f_inf-f_0)*np.exp(-(dt)/tau)
    return f
@njit(fastmath=True)
def f_Ca(V,t,Ca_con_i,dt,f_Ca_0):
    f_Ca_inf = 1/(1+Ca_con_i/0.00035)
    tau = 2
    f_Ca = f_Ca_inf-(f_Ca_inf-f_Ca_0)*np.exp(-(dt)/tau)
    return f_Ca

########################
##### I_p_Ca
########################
@njit(fastmath=True)
def I_p_Ca_func(I_p_Ca_max, Ca_con_i):
    I_p_Ca = I_p_Ca_max*Ca_con_i/(0.0005+Ca_con_i)
    return  I_p_Ca

########################
##### I_NaK
########################
@njit(fastmath=True)
def I_NaK_func(I_NaK_max, f_NaK, K_m_Na_i, Na_con_i, K_con_o, K_m_K_o):
    I_NaK = I_NaK_max*f_NaK* 1/(1+(K_m_Na_i/Na_con_i)**1.5) * K_con_o/(K_con_o+K_m_K_o)
    return I_NaK
@njit(fastmath=True)
def f_NaK(F,V,R,T,sigma):
    f_NaK = (1+0.1245*np.exp(-0.1*F*V/(R*T))+0.0365*sigma*np.exp(-F*V/(R*T)))**(-1)
    return f_NaK
@njit(fastmath=True)
def sigma(Na_con_o):
    sigma = 1/7*(np.exp(Na_con_o/67.3)-1)
    return sigma
sigma_val = sigma(Na_con_o)

########################
##### I_NaCa
########################
@njit(fastmath=True)
def I_NaCa_func(I_NaCa_max, gamma, F,V,R,T,Na_con_i, Ca_con_o, Na_con_o,Ca_con_i,K_m_Na,K_m_Ca, k_sat):
    numerator = (np.exp(gamma*F*V/(R*T))*Na_con_i**3*Ca_con_o-np.exp((gamma-1)*F*V/(R*T))*Na_con_o**3*Ca_con_i)
    denominator = (K_m_Na**3+Na_con_o**3)*(K_m_Ca+Ca_con_o)*(1+k_sat*np.exp((gamma-1)*F*V/(R*T)))
    I_NaCa = I_NaCa_max*numerator/denominator
    return I_NaCa

########################
##### I_background
########################
@njit(fastmath=True)
def I_b_Ca_func(g_b_Ca,V,E_Ca):
    I_b_Ca = g_b_Ca *(V-E_Ca)
    return I_b_Ca
@njit(fastmath=True)
def I_b_Na_func(g_b_Na,V,E_Na):
    I_b_Na = g_b_Na *(V-E_Na)
    return I_b_Na

########################
##### I_rel
########################
@njit(fastmath=True)
def I_rel_func(k_rel, u, v, w, Ca_con_rel, Ca_con_i):
    I_rel = k_rel*u**2*v*w*(Ca_con_rel-Ca_con_i)
    return I_rel
@njit(fastmath=True)
def F_n(V_rel,I_rel,I_Ca_L,I_NaCa,F):
    F_n = 10**(-12)*V_rel*I_rel-5*10**(-13)/F*(1/2*I_Ca_L-1/5*I_NaCa)*Cm #*Cm in the second term for dimensional analysis 
    return F_n
@njit(fastmath=True)
def u(V,t,F_n,dt,u_0):
    u_inf = (1+np.exp(-(F_n-0.4*3.4175e-13)/(13.67*10**(-16))))**(-1)
    tau = 8.0
    u = u_inf-(u_inf-u_0)*np.exp(-(dt)/tau)    
    return u
@njit(fastmath=True)
def v(V,t,F_n,dt,v_0):
    v_inf = 1-(1+np.exp(-(F_n-6.835*10**(-14))/(13.67*10**(-16))))**(-1)  
    tau = 1.91+2.09*(1+np.exp(-(F_n-0.4*3.4175*10**(-13))/(13.67*10**(-16))))**(-1)
    v = v_inf - (v_inf-v_0)*np.exp(-(dt)/tau)
    return v
@njit(fastmath=True)
def w(V,t,dt,w_0):
    w_inf = 1-(1+np.exp(-(V-70)/8))**(-1)
    tau = (6.0*(1-np.exp(-(V-7.9)/5))/((1+0.3*np.exp(-(V-7.9)/5))*(V-7.9)))
    w = w_inf - (w_inf-w_0)*np.exp(-(dt)/tau)
    return w

########################
##### I_up_leak
########################
@njit(fastmath=True)
def I_up_leak_func(Ca_con_up,Ca_con_up_max,I_up_max):
    I_up_leak = Ca_con_up/Ca_con_up_max * I_up_max
    return I_up_leak

########################
##### I_up
########################
@njit(fastmath=True)
def I_up_func(I_up_max, K_up, Ca_con_i):
    I_up = I_up_max / (1+(K_up/Ca_con_i))
    return I_up

########################
##### I_tr
########################
tau_tr  = 180
@njit(fastmath=True)
def I_tr_func(Ca_con_up,Ca_con_rel,tau_tr):

    I_tr = (Ca_con_up-Ca_con_rel)/tau_tr

    return I_tr

########################
############################ GET CURRENTS
########################
@njit(fastmath=True)#(parallel=True,cache=True)
def get_currents(V,t,
                 F,R,T,
                 K_Q_10,
                 E_Na,E_K,E_Ca,E_Cl,
                 g_Kr,g_Ks,
                 g_Ca_L,Ca_con_i,
                 I_p_Ca_max,
                 I_NaK_max,Na_con_o,K_m_Na_i,Na_con_i,K_con_o,K_m_K_o,
                 I_NaCa_max,gamma,Ca_con_o,K_m_Ca,k_sat,
                 g_b_Na,g_b_Ca,
                 k_rel,
                 V_rel,I_rel,Ca_con_rel,
                 Ca_con_up,Ca_con_up_max,I_up_max,
                 K_up,
                 tau_tr,
                 dt,
                 m_0,h_0,j_0,u_a_0,u_if_0,u_is_0,x_r_0,x_s_0,d_0,f_0,f_Ca_0,u_0,v_0,w_0,
                 q_Ca_0):

    f_NaK_val = f_NaK(F,V,R,T,sigma_val)

    m_val = m(V,t,dt,m_0);h_val=h(V,t,dt,h_0);j_val=j(V,t,dt,j_0)
    u_a_val=u_a(V,t,K_Q_10,dt,u_a_0);u_if_val=u_if(V,t,K_Q_10,dt,u_if_0);u_is_val = u_if(V,t,K_Q_10,dt,u_is_0)
    x_r_val=x_r(V,t,dt,x_r_0);x_s_val=x_s(V,t,dt,x_s_0)
    d_val = d(V,t,dt,d_0);f_val=f(V,t,dt,f_0);f_Ca_val=f_Ca(V,t,Ca_con_i,dt,f_Ca_0)

    
    I_Na = I_Na_func(m_val,h_val,j_val,V,E_Na)
    I_K1 = I_K1_func(V,E_K)
    
    I_Kur = I_Kur_func(g_Kur(V),u_a_val,u_if_val,u_is_val,V,E_K)
    I_Kr = I_Kr_func(g_Kr,x_r_val,V,E_K)
    I_Ks = I_Ks_func(g_Ks,x_s_val,V,E_K)
    I_Ca_L = I_Ca_L_func(g_Ca_L,d_val,f_val,f_Ca_val,V)
    I_p_Ca = I_p_Ca_func(I_p_Ca_max,Ca_con_i)
    I_NaK = I_NaK_func(I_NaK_max,f_NaK_val,K_m_Na_i,Na_con_i,K_con_o,K_m_K_o)
    I_NaCa = I_NaCa_func(I_NaCa_max,gamma,F,V,R,T,Na_con_i,Ca_con_o,Na_con_o,Ca_con_i,
                         K_m_Na,K_m_Ca,k_sat)
    I_b_Na = I_b_Na_func(g_b_Na,V,E_Na)
    I_b_Ca = I_b_Ca_func(g_b_Ca,V,E_Ca)
    #JSR and NSR
    
    F_n_val=F_n(V_rel,I_rel,I_Ca_L,I_NaCa,F)
    u_val=u(V,t,F_n_val,dt,u_0);v_val=v(V,t,F_n_val,dt,v_0);w_val=w(V,t,dt,w_0)
    q_Ca_val = q_Ca(V,t,F_n_val,q_Ca_0)

    I_to2 = I_to2_func(V,E_Cl,q_Ca_val)

    I_rel = I_rel_func(k_rel,u_val,
                       v_val,
                       w_val,Ca_con_rel,Ca_con_i)
    I_up_leak = I_up_leak_func(Ca_con_up,Ca_con_up_max,I_up_max)
    I_up = I_up_func(I_up_max,K_up,Ca_con_i)
    I_tr = I_tr_func(Ca_con_up,Ca_con_rel,tau_tr)

    I_ion = I_Na+I_K1+I_to2+I_Kur+I_Kr+I_Ks+I_Ca_L+I_p_Ca+I_NaK+I_NaCa+I_b_Na+I_b_Ca
    
    
    m_0 = m_val;h_0=h_val;j_0=j_val
    u_a_0=u_a_val;u_if_0=u_if_val;u_is_0=u_is_val
    x_r_0=x_r_val;x_s_0=x_s_val
    d_0 = d_val;f_0=f_val;f_Ca_0=f_Ca_val
    u_0=u_val;v_0=v_val;w_0=w_val
    q_Ca_0 = q_Ca_val
    
    
    return (I_Na,I_K1,I_to2,I_Kur,I_Kr,I_Ks,I_Ca_L,I_p_Ca,I_NaK,I_NaCa,I_b_Na,I_b_Ca,
            I_ion,I_rel,I_up_leak,I_up,I_tr,
            m_0,h_0,j_0,u_a_0,u_if_0,u_is_0,x_r_0,x_s_0,d_0,f_0,f_Ca_0,u_0,v_0,w_0,q_Ca_0,
            F_n_val)



####################################################################################
'''
    BOUNDARY CONDITIONS
''' 
####################################################################################
@njit(fastmath=True)
def apply_BC(V,myrank):
    V[:,0] = V[:,2]
    V[:,yspace+1] = V[:,yspace-1]

    if myrank==0:
        V[0,:] = V[2,:]

    if myrank==size-1:
        V[mypoints+1,:] = V[mypoints-1,:]

    return V

####################################################################################
'''
    BOUNDARY CONDITIONS SENDING
''' 
####################################################################################

def send_cables(V,myrank):
    cable1 = V[1,:]
    cable3 = V[mypoints,:]

    if size==1: #escape if not running in parallel
        return V

    #leftmost and rightmost processes
    if myrank==0: #always even
        cable4 = V[mypoints+1,:]
        cable2 = V[0,:]
        comm.Send([cable3,MPI.DOUBLE],dest=rightProc,tag=0)
        comm.Recv([cable4,MPI.DOUBLE],source=rightProc,tag=0)
    if myrank==size-1: #always odd, if number of processes is even
        cable2 = V[0,:]
        cable4 = V[mypoints+1,:]
        comm.Recv([cable2,MPI.DOUBLE],source=leftProc,tag=0)
        comm.Send([cable1,MPI.DOUBLE],dest=leftProc,tag=0)

    #all other processes   
    if myrank%2==0 and myrank!=0 and myrank!=size-1:
        cable2 = V[0,:]
        cable4 = V[mypoints+1,:]
        comm.Send([cable1,MPI.DOUBLE],dest=leftProc,tag=0)
        comm.Send([cable3,MPI.DOUBLE],dest=rightProc,tag=0)
        comm.Recv([cable2,MPI.DOUBLE],source=leftProc,tag=0)
        comm.Recv([cable4,MPI.DOUBLE],source=rightProc,tag=0)
    elif myrank%2!=0 and myrank!=0 and myrank!=size-1:
        cable2 = V[0,:]
        cable4 = V[mypoints+1,:]
        comm.Recv([cable2,MPI.DOUBLE],source=leftProc,tag=0)
        comm.Recv([cable4,MPI.DOUBLE],source=rightProc,tag=0)
        comm.Send([cable1,MPI.DOUBLE],dest=leftProc,tag=0)
        comm.Send([cable3,MPI.DOUBLE],dest=rightProc,tag=0)
        
    V[0,:] = cable2
    V[mypoints+1,:] = cable4

    return V

####################################################################################
'''
    UPDATE V FIELD
''' 
####################################################################################
@njit(fastmath=True)
def update_potential(V,I_ion,dt,D,dx):
    #UPDATE V
    shape = np.shape(V)
    V_plusone_x = np.ones(shape)
    V_minusone_x = np.ones(shape)
    V_plusone_y = np.ones(shape)
    V_minusone_y = np.ones(shape)

    for ii in np.arange(shape[0]-2):
        for jj in np.arange(shape[1]-2):
            V_plusone_x[ii+1,jj+1] = V[ii+1+1,jj+1]
            V_minusone_x[ii+1,jj+1] = V[ii+1-1,jj+1]
            V_plusone_y[ii+1,jj+1] = V[ii+1,jj+1+1]
            V_minusone_y[ii+1,jj+1] = V[ii+1,jj+1-1]
    
##    V_plusone_x = np.roll(V,1,axis=1)
##    V_plusone_x = np.ones(shape)
##    V_minusone_x = np.roll(V,-1,axis=1)
##    V_plusone_y = np.roll(V,1,axis=0)
##    V_minusone_y = np.roll(V,-1,axis=0)

    V_temp = -(I_ion)*dt+D*dt/dx**2*(V_plusone_x+V_plusone_y
                                        +V_minusone_x+V_minusone_y
                                        -4*V)+V
            #I_ion includes stimmulus, if stimulated
    return V_temp

####################################################################################
'''
    STIMULUS APPLICATION
''' 
#############################5######################################################
I_st = -7000/Cm #stimulus current in pA/pF
print(I_st)
t_st = 4 #duration of stimulation in ms
t_st_0 = 10 ###IGNORED #initial stimulation time within each cycle

rank_to_stimulate = [0] #list of all ranks that we will stimulate

@njit(fastmath=True)
def apply_stimulus(I_ion, I_st,ii,jj):
    for x in ii:
        for y in jj:
            I_ion[x,y] = I_ion[x,y]+I_st
    return I_ion



################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

################################################################################
'''
    MAIN
'''
################################################################################

##frequencies = np.concatenate((np.array([0.25]),np.arange(0.5,6.5,0.5)))
frequencies = np.concatenate((np.array([0.25]),np.arange(0.5,4.5,0.5)))
print(frequencies)
#frequencies = np.array([2.5])
periods = 1/frequencies

point_field = np.ones((mypoints+2,yspace+2),dtype=np.float64)

####################################################################################
'''
        INITIAL CONDITIONS
'''
####################################################################################
reuse_old_state = False
if reuse_old_state==False:
    V = -8.12*10                            *point_field
    h_0 = 9.65*10**-1                       *point_field
    d_0 = 1.37*10**-4                       *point_field
    x_r_0 = 3.29*10**-5                     *point_field
    Na_con_i = 1.117*10**1                  *point_field
    K_con_i = 1.39*10**2                    *point_field
    Ca_con_rel = 1.488                      *point_field
    o_i_0 = 9.99*10**-1                     *point_field
    u_i_0 = 9.99*10**-1                     *point_field
    u_if_0 = 1                              *point_field
    u_is_0 = 1                              *point_field
    Cmdn_con_i = 2.05*10**-3                *point_field
    Csqn_con_i = 6.51                       *point_field
    v_0 = 1.00                              *point_field
    m_0 = 2.91*10**-3                       *point_field
    j_0 = 9.78*10**-1                       *point_field
    f_0 = 9.99*10**-1                       *point_field
    x_s_0 = 1.87*10**-2                     *point_field
    Ca_con_i = 0.0001013                    *point_field
    Ca_con_up = 1.488                       *point_field
    Cl_con_i = 30                           *point_field
    o_a_0 = 3.04*10**-2                     *point_field
    u_a_0 = 4.96*10**-3                     *point_field
    f_Ca_0 = 7.75*10**-1                    *point_field
    Trpn_con_i = 1.18*10**-2                *point_field
    u_0 = 2.35*10**-112                     *point_field
    w_0 = 9.99*10**-1                       *point_field
    I_rel = 0.0                             *point_field ##### k_rel*u**2*v*w*(Ca_con_rel-Ca_con_i) but u=0.00
    I_b_K = 0.0                             *point_field #### NO SOURCE
    q_Ca_0 = 0.0000001                      *point_field

    t=-100 #ms, initial time
    #####IF LOOKING AT SPIRAL FORMATION, GIVE A BIT MORE INITIAL TIME
    t = -200
else:
    ####### LOAD LAST STATE OF THE SYSTEM

    filetime = open('last_state/time.txt','r')
    t = filetime.readline()
    t = float(t)
    filetime.close()
    V = np.genfromtxt('last_state/V.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    h_0 = np.genfromtxt('last_state/h_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    j_0 = np.genfromtxt('last_state/j_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    d_0 = np.genfromtxt('last_state/d_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    x_r_0 = np.genfromtxt('last_state/x_r_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    Na_con_i = np.genfromtxt('last_state/Na_con_i.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    K_con_i = np.genfromtxt('last_state/K_con_i.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    Ca_con_rel = np.genfromtxt('last_state/Ca_con_rel.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    o_i_0 = np.genfromtxt('last_state/o_i_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    u_i_0 = np.genfromtxt('last_state/u_i_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    u_if_0 = np.genfromtxt('last_state/u_if_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    u_is_0 = np.genfromtxt('last_state/u_is_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    Cmdn_con_i = np.genfromtxt('last_state/Cmdn_con_i.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    Csqn_con_i = np.genfromtxt('last_state/Csqn_con_i.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    v_0 = np.genfromtxt('last_state/v_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    m_0 = np.genfromtxt('last_state/m_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    f_0 = np.genfromtxt('last_state/f_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    x_s_0 = np.genfromtxt('last_state/x_s_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    Ca_con_i = np.genfromtxt('last_state/Ca_con_i.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    Ca_con_up = np.genfromtxt('last_state/Ca_con_up.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    Cl_con_i = np.genfromtxt('last_state/Cl_con_i.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    o_a_0 = np.genfromtxt('last_state/o_a_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    u_a_0 = np.genfromtxt('last_state/u_a_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    f_Ca_0 = np.genfromtxt('last_state/f_Ca_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    Trpn_con_i = np.genfromtxt('last_state/Trpn_con_i.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    u_0 = np.genfromtxt('last_state/u_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    w_0 = np.genfromtxt('last_state/w_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    I_rel = np.genfromtxt('last_state/I_rel.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    I_b_K =np.genfromtxt('last_state/I_b_K.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')
    q_Ca_0 = np.genfromtxt('last_state/q_Ca_0.'+str(round(t))+'.'+str(myrank)+'.csv',delimiter=',')







######### SPECIAL CASE WHEN LOOKING AT RESTITUTION:
V_at0_vect = []
V_middle_vect =[]
V_middle2_vect = []
V_middle3_vect = []
t_vect = []
trace_Na = []
trace_CaL = []
trace_K1 = []
trace_Kr = []
trace_Ks = []
trace_Kur = []
trace_NaK = []
trace_NaCa = []
trace_to2 = []
V_1Hz = []
if myrank==0:
    point_0 = (1,int(yspace/2))
    V_at0 = V[point_0[0],point_0[1]]
    V_at0_vect.append(V_at0)
    t_vect.append(t)
elif myrank==round(size/2):
    point = (1,int(yspace/2))
    V_middle = V[point[0],point[1]]
    V_middle_vect.append(V_middle)
elif myrank==int(size/3):
    point = (1,int(yspace/3))
    V_middle2 = V[point[0],point[1]]
    V_middle2_vect.append(V_middle2)
elif myrank==int(2*size/3):
    point = (1,2*int(yspace/3))
    V_middle3 = V[point[0],point[1]]
    V_middle3_vect.append(V_middle3)

for frequency in frequencies:
    write_counter = 1  #output write counter
    write_counter_restitution = 1

    t_0 = max(0,t)  #Initialize time as the end time of previous frequency cycle, and as 0 if initial case

    
    ####################################################################################
    '''
        TIME-RELATED SIMULATION CONSTANTS
    '''
    ####################################################################################

    dt=0.02 #ms
    T_cycle=1000/frequency #Period (ms)
    if frequency==0.25:
        total_cycles = 8 #number of stimulations at each frequency
    else:
        total_cycles = 5
    writing_interval = int(10/dt) #writing to output every N ms, every int(N/dt) steps
    writing_interval_restitution = int(0.5/dt) #writing every 0.5 ms

    print('Starting Simulation in Process ', myrank)
    print('Stimulation frequency (Hz): ',frequency)

    initial_time = MPI.Wtime()


    #Initializing values
    Ca_con_Cmdn,Ca_con_Trpn,Ca_con_Csqn = get_concentrations(Cmdn_con_max,Ca_con_i,K_m_Cmdn,
                                                         Trpn_con_max,K_m_Trpn,
                                                         Csqn_con_max,Ca_con_rel,K_m_Csqn)
    E_Na,E_K,E_Ca,E_Cl = get_rest_potentials(R,T,z_Na,F,Na_con_o,Na_con_i,
                                    z_K,K_con_o,K_con_i,
                                    z_Ca,Ca_con_o,Ca_con_i,
                                    z_Cl,Cl_con_o,Cl_con_i)
    (I_Na,I_K1,I_to2,I_Kur,I_Kr,I_Ks,I_Ca_L,I_p_Ca,I_NaK,I_NaCa,I_b_Na,I_b_Ca,
     I_ion,I_rel,I_up_leak,I_up,I_tr,
     m_0,h_0,j_0,u_a_0,u_if_0,u_is_0,x_r_0,x_s_0,d_0,f_0,f_Ca_0,u_0,v_0,w_0,q_Ca_0,
     F_n_val
     ) = get_currents(V,t,
                  F,R,T,
                  K_Q_10,
                  E_Na,E_K,E_Ca,E_Cl,
                  g_Kr,g_Ks,
                  g_Ca_L,Ca_con_i,
                  I_p_Ca_max,
                  I_NaK_max,Na_con_o,K_m_Na_i,Na_con_i,K_con_o,K_m_K_o,
                  I_NaCa_max,gamma,Ca_con_o,K_m_Ca,k_sat,
                  g_b_Na,g_b_Ca,
                  k_rel,
                  V_rel,I_rel,Ca_con_rel,
                  Ca_con_up,Ca_con_up_max,I_up_max,
                  K_up,
                  tau_tr,
                  dt,m_0,h_0,j_0,u_a_0,u_if_0,u_is_0,x_r_0,x_s_0,d_0,f_0,f_Ca_0,u_0,v_0,w_0,
                  q_Ca_0)


    counter = 0 #Cycle counter
    
    print('Entering loop in process ',myrank)

    print('TIme:',t)

        

    while counter < total_cycles:
        if t%100==0: print('Time:',myrank,t); np.savetxt('{}.{}.csv'.format(myrank,t),np.array([1,2,3]),delimiter=',')
        
      
        V = apply_BC(V,myrank)

        V = send_cables(V,myrank)

        comm.Barrier()

        #Stimulate, if applicable
        if myrank in rank_to_stimulate:
            if  0 <= t-counter*T_cycle-t_0 < 0+t_st:
                ii= np.array([1])
                jj= np.arange(yspace+2)
                I_ion = apply_stimulus(I_ion,I_st,ii,jj)


                
        #Update Potential
        V_temp = update_potential(V,I_ion,dt,D,dx)
        #UPDATE REST POTENTIALS
        E_Na_temp,E_K_temp,E_Ca_temp,E_Cl_temp = get_rest_potentials(R,T,z_Na,F,Na_con_o,Na_con_i,
                                    z_K,K_con_o,K_con_i,
                                    z_Ca,Ca_con_o,Ca_con_i,
                                    z_Cl,Cl_con_o,Cl_con_i)
        
        #UPDATE CONCENTRATIONS
        (Na_con_i_temp,K_con_i_temp,Ca_con_i_temp,Ca_con_up_temp,Ca_con_rel_temp,Cl_con_i_temp
         )=update_concentrations(
             dt,Na_con_i,K_con_i,Ca_con_i,Ca_con_up,Ca_con_rel,
                          I_NaK,I_NaCa,I_b_Na,I_Na,F,Vol_i,
                          I_K1,I_to2,I_Kur,I_Kr,I_Ks,I_b_K,
                          I_p_Ca,I_Ca_L,I_b_Ca,V_up,I_up_leak,I_up,I_rel,V_rel,
                          Trpn_con_max,K_m_Trpn,Cmdn_con_max,K_m_Cmdn,
                          I_tr,
                          Csqn_con_max,K_m_Csqn,
                          Cl_con_i,Cm)

        #UPDATE BUFFERS
        (Ca_con_Cmdn_temp,Ca_con_Trpn_temp,Ca_con_Csqn_temp) = get_concentrations(Cmdn_con_max,Ca_con_i,K_m_Cmdn,
                                                         Trpn_con_max,K_m_Trpn,
                                                         Csqn_con_max,Ca_con_rel,K_m_Csqn)
        
        #UPDATE CURRENTS
        (I_Na_temp,I_K1_temp,I_to2_temp,I_Kur_temp,I_Kr_temp,I_Ks_temp,I_Ca_L_temp,I_p_Ca_temp,I_NaK_temp,I_NaCa_temp,I_b_Na_temp,I_b_Ca_temp,
         I_ion_temp,I_rel_temp,I_up_leak_temp,I_up_temp,I_tr_temp,
         m_0,h_0,j_0,u_a_0,u_if_0,u_is_0,x_r_0,x_s_0,d_0,f_0,f_Ca_0,u_0,v_0,w_0,q_Ca_0,
         F_n_val
         ) = get_currents(V,t,
                  F,R,T,
                  K_Q_10,
                  E_Na,E_K,E_Ca,E_Cl,
                  g_Kr,g_Ks,
                  g_Ca_L,Ca_con_i,
                  I_p_Ca_max,
                  I_NaK_max,Na_con_o,K_m_Na_i,Na_con_i,K_con_o,K_m_K_o,
                  I_NaCa_max,gamma,Ca_con_o,K_m_Ca,k_sat,
                  g_b_Na,g_b_Ca,
                  k_rel,
                  V_rel,I_rel,Ca_con_rel,
                  Ca_con_up,Ca_con_up_max,I_up_max,
                  K_up,
                  tau_tr,
                  dt,m_0,h_0,j_0,u_a_0,u_if_0,u_is_0,x_r_0,x_s_0,d_0,f_0,f_Ca_0,u_0,v_0,w_0,
                  q_Ca_0)
        
        V = V_temp
        Na_con_i,K_con_i,Ca_con_i,Ca_con_up,Ca_con_rel,Cl_con_i = (Na_con_i_temp,K_con_i_temp,Ca_con_i_temp,Ca_con_up_temp,Ca_con_rel_temp,Cl_con_i_temp)
        Ca_con_Cmdn,Ca_con_Trpn,Ca_con_Csqn = (Ca_con_Cmdn_temp,Ca_con_Trpn_temp,Ca_con_Csqn_temp)
        E_Na,E_K,E_Ca,E_Cl = E_Na_temp,E_K_temp,E_Ca_temp,E_Cl_temp
        (I_Na,I_K1,I_to2,I_Kur,I_Kr,I_Ks,I_Ca_L,I_p_Ca,I_NaK,I_NaCa,I_b_Na,I_b_Ca,
         I_ion,I_rel,I_up_leak,I_up,I_tr) = (I_Na_temp,I_K1_temp,I_to2_temp,I_Kur_temp,I_Kr_temp,I_Ks_temp,I_Ca_L_temp,I_p_Ca_temp,I_NaK_temp,I_NaCa_temp,I_b_Na_temp,I_b_Ca_temp,
         I_ion_temp,I_rel_temp,I_up_leak_temp,I_up_temp,I_tr_temp)


        ####WRITING DOWN TRACES FOR RESTITUTION
        if write_counter_restitution == writing_interval_restitution:
            if abs(frequency-1)<1e-4:
##                print('Writing to restitution in rank {} in frequency'.format(myrank), frequency)
                if myrank==round(size/2):
                    point = (1,int(yspace/2))
                    trace_Na.append(I_Na[point[0],point[1]])
                    trace_CaL.append(I_Ca_L[point[0],point[1]])
                    trace_K1.append(I_K1[point[0],point[1]])
                    trace_Kr.append(I_Kr[point[0],point[1]])
                    trace_Ks.append(I_Ks[point[0],point[1]])
                    trace_Kur.append(I_Kur[point[0],point[1]])
                    trace_NaK.append(I_NaK[point[0],point[1]])
                    trace_NaCa.append(I_NaCa[point[0],point[1]])
                    trace_to2.append(I_to2[point[0],point[1]])
                    V_1Hz.append(V[point[0],point[1]])


        t+=dt
        
        
        if write_counter_restitution == writing_interval_restitution:
            ######### SPECIAL CASE WHEN LOOKING AT RESTITUTION:
         
           # In this case we will only save point_0 and point_middle, to save time,
           # since we don't actually want to look at all the 2D map.
           # Point_0 is at (0,yspace/2) of rank 0 (stimulation)
           # Point_middle is at (0,yspace/2) of rank N/2 (middle of map).
           # 
            if myrank==0:
                point_0 = (1,int(yspace/2))
                V_at0 = V[point_0[0],point_0[1]]
                V_at0_vect.append(V_at0)
                t_vect.append(t)
            elif myrank==round(size/2):
                point = (1,int(yspace/2))
                V_middle = V[point[0],point[1]]
                V_middle_vect.append(V_middle)
            elif myrank==int(size/3):
                point = (1,int(yspace/3))
                V_middle2 = V[point[0],point[1]]
                V_middle2_vect.append(V_middle2)
            elif myrank==int(2*size/3):
                point = (1,2*int(yspace/3))
                V_middle3 = V[point[0],point[1]]
                V_middle3_vect.append(V_middle3)
       
            write_counter_restitution = 1 

        else:
##            print(t,write_counter)
            write_counter_restitution += 1
        

        ###### VALID ONLY IF LOOOKING AT FREQUENCY DEPENDENCE. IF NOT, COMMENT OUT    
        if t-t_0>(counter+1)*T_cycle:
            counter+=1
            if myrank==0:
                print(frequency,counter)
        


if myrank==0:
    V_at0_vect = np.array(V_at0_vect)
    np.savetxt('output_merged/V.at0.csv',V_at0_vect,delimiter=',') #Version with different frequencies of stimulation
    t_vect = np.array(t_vect)    
    np.savetxt('output_merged/time.csv',t_vect,delimiter=',')
elif myrank==round(size/2):
    V_middle_vect = np.array(V_middle_vect)
    np.savetxt('output_merged/V.middle.csv',V_middle_vect,delimiter=',') #Version with different frequencies of stimulation
    np.savetxt('output_merged/V.middle.csv',V_middle_vect,delimiter=',')
    np.savetxt('output_merged/V_1Hz.csv',V_1Hz,delimiter=',')
    np.savetxt('output_merged/trace_Na.csv',trace_Na,delimiter=',')
    np.savetxt('output_merged/trace_CaL.csv',trace_CaL,delimiter=',')
    np.savetxt('output_merged/trace_K1.csv',trace_K1,delimiter=',')
    np.savetxt('output_merged/trace_Kr.csv',trace_Kr,delimiter=',')
    np.savetxt('output_merged/trace_Ks.csv',trace_Ks,delimiter=',')
    np.savetxt('output_merged/trace_Kur.csv',trace_Kur,delimiter=',')
    np.savetxt('output_merged/trace_NaK.csv',trace_NaK,delimiter=',')
    np.savetxt('output_merged/trace_NaCa.csv',trace_NaCa,delimiter=',')
    np.savetxt('output_merged/trace_to2.csv',trace_to2,delimiter=',')
elif myrank==int(size/3):
    V_middle2_vect = np.array(V_middle2_vect)
    np.savetxt('output_merged/V.middle2.csv',V_middle2_vect,delimiter=',')
elif myrank==int(2*size/3):
    V_middle3_vect = np.array(V_middle3_vect)
    np.savetxt('output_merged/V.middle3.csv',V_middle3_vect,delimiter=',')



end_time = MPI.Wtime()
print('Time taken in processor '+str(myrank)+': ', end_time-initial_time)

MPI.Finalize()







































