import numpy as np
from copy import copy

cos=np.cos; sin=np.sin; pi=np.pi


def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.
    """
    # Escriba aqui la matriz de transformacion homogenea en funcion de los valores de d, theta, a, alpha
    cth = cos(theta);    sth = sin(theta)
    ca = cos(alpha);  sa = sin(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                    [sth,  ca*cth, -sa*cth, a*sth],
                    [0,        sa,     ca,      d],
                    [0,         0,      0,      1]])
    return T
    
    

def fkine_7dof(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6,q7]
    """
    # Longitudes (en metros)

    # Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
    T1 = dh( 0.40, 0+q[0],       0.0,  -np.pi/2)
    T2 = dh( 0.0,  -np.pi/2+q[1], 0.50, 0.0)
    T3 = dh( 0.0,  0+q[2],       0.0,  -np.pi/2)
    T4 = dh( 0.45, q[3],   0.0,  np.pi/2)
    T5 = dh( 0.0,  0+q[4],   0.0,  -np.pi/2)
    T6 = dh( 0.20, 0+q[5],       0.0,  0.0)
    T7 = dh( 0.10+q[6], 0.0,       0.0,  0.0)
    # Efector final con respecto a la base
    T = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @T7
    return T



def jacobian_ur5(q, delta=0.0001):
    """
    Jacobiano analítico para la posición. Retorna una matriz de 3x7 y toma como
    entrada el vector de configuración articular q=[q1, q2, q3, q4, q5, q6, q7]
    """
    # Crear una matriz 3x7
    J = np.zeros((3, 7))
    # Calcular la transformación homogénea inicial (usando q)
    T = fkine_7dof(q)
    
    # Iteración para la derivada de cada articulación (columna)
    for i in range(7):
        # Copiar la configuración articular inicial
        dq = q

        if dq[6] >= 0.05:
            dq[6] = 0.05
        elif dq[6] <= 0.0:
            dq[6] = 0.0

        # Calcular nuevamente la transformación homogénea e
        T = fkine_7dof(dq)
        # Incrementar la articulación i-ésima usando un delta
        dq[i] = dq[i] + delta

        # Transformación homogénea luego del incremento (q+delta)
        T_inc = fkine_7dof(dq)
        # Aproximación del Jacobiano de posición usando diferencias finitas
        J[0:3, i] = (T_inc[0:3, 3] - T[0:3, 3]) / delta
    return J

def jacobian_position(q, delta=0.001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6, q7]

    """
    # Alocacion de memoria
    J = np.zeros((3,7))
    # Transformacion homogenea inicial (usando q)
    T = fkine_7dof(q)
    # Iteracion para la derivada de cada columna
    for i in range(7):
        dq = q
        
        if q[6] >= 0.05:
            q[6] = 0.05
        elif q[6] <= 0.0:
            q[6] = 0.0
        T = fkine_7dof(dq)
        dq[i]=dq[i]+delta

        Td=fkine_7dof(dq)
        J[0:3,i] = 1/delta * (Td[0:3,3] - T[0:3,3])
    return J


def jacobian_pose(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion y orientacion (usando un
    cuaternion). Retorna una matriz de 7x6 y toma como entrada el vector de
    configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """
    J = np.zeros((7,7))
    # Implementar este Jacobiano aqui
    # Calcuar la transformacion homogenea inicial (usando q)
    T = fkine_7dof(q)
    for i in range(7):
        

        #Obtener la matriz de rotacion
        R=T[0:2,0:2]
        #Obtener el cuaternion equivalente
        Q=rot2quat(R)

        # Copiar la configuracion articular inicial
        dq = copy(q)

        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta
        
        if dq[6] >= 0.05:
            dq[6] = 0.05
        elif dq[6] <= 0.0:
            dq[6] = 0.0

        # Transformacion homogenea luego del incremento (q+delta)
        dT = fkine_7dof(dq)
        dR = dT[0:2,0:2]
        dQ = rot2quat(dR)

        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[0,i] = (dT[0,3] - T[0,3])/delta #derivadas de x
        J[1,i] = (dT[1,3] - T[1,3])/delta #derivadas de y
        J[2,i] = (dT[2,3] - T[2,3])/delta #derivadas de z
        
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[3,i] = (dQ[0,0] - Q[0,0])/delta #derivadas de w
        J[4,i] = (dQ[0,1] - Q[0,1])/delta #derivadas de ex
        J[5,i] = (dQ[0,2] - Q[0,2])/delta #derivadas de ey
        J[6,i] = (dQ[0,3] - Q[0,3])/delta #derivadas de ez
    return J


def rot2quat(R):
    """
    Convertir una matriz de rotacion en un cuaternion

    Entrada:
      R -- Matriz de rotacion
    Salida:
      Q -- Cuaternion [ew, ex, ey, ez]

    """
    dEpsilon = 1e-6
    quat = 4*[0.,]

    quat[0] = 0.5*np.sqrt(R[0,0]+R[1,1]+R[2,2]+1.0)
    if ( np.fabs(R[0,0]-R[1,1]-R[2,2]+1.0) < dEpsilon ):
        quat[1] = 0.0
    else:
        quat[1] = 0.5*np.sign(R[2,1]-R[1,2])*np.sqrt(R[0,0]-R[1,1]-R[2,2]+1.0)
    if ( np.fabs(R[1,1]-R[2,2]-R[0,0]+1.0) < dEpsilon ):
        quat[2] = 0.0
    else:
        quat[2] = 0.5*np.sign(R[0,2]-R[2,0])*np.sqrt(R[1,1]-R[2,2]-R[0,0]+1.0)
    if ( np.fabs(R[2,2]-R[0,0]-R[1,1]+1.0) < dEpsilon ):
        quat[3] = 0.0
    else:
        quat[3] = 0.5*np.sign(R[1,0]-R[0,1])*np.sqrt(R[2,2]-R[0,0]-R[1,1]+1.0)

    return np.array(quat)


def ikine_ur5(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0. 
    Emplear el metodo de newton
    """
    epsilon  = 0.001
    max_iter = 1000
    delta    = 0.00001

    q  = q0
    
    # Almacenamiento del error
    ee = []
    for i in range(max_iter):
        # Main loop
        J = jacobian_ur5(q,delta)
        T = fkine_7dof(q)
        f = T[0:3, 3]
        e = xdes-f        
        q = q + np.dot(np.linalg.pinv(J),e)
        
        ee.append(np.linalg.norm(e))    # Almacena los errores
        # Condicion de termino
        if (np.linalg.norm(e) < epsilon):
            print("Cinemática inversa por método de newton: solución obtenida")
            break
        if (i==max_iter-1):
            print("El algoritmo no llegó al valor deseado")	
    return q

def ik_gradient_ur5(xdes,q0): # Metodo por gradiente

    epsilon  = 0.001
    max_iter = 1000
    delta    = 0.00001
    alfa = 0.1
   
    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        T=fkine_7dof(q)
        f=T[0:3,3]
       
        error=xdes-f
       
        J=jacobian_ur5(q,delta)
       
        q = q + alfa*np.dot(J.T, error)
       
        if (np.linalg.norm(error) < epsilon):
            break
   
    return q

def TF2xyzquat(T):
    """
    Convert a homogeneous transformation matrix into the a vector containing the
    pose of the robot.

    Input:
      T -- A homogeneous transformation
    Output:
      X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
           is Cartesian coordinates and the last part is a quaternion
    """
    quat = rot2quat(T[0:3,0:3])
    res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
    return np.array(res)


def skew(w):
    R = np.zeros([3,3])
    R[0,1] = -w[2]; R[0,2] = w[1]
    R[1,0] = w[2];  R[1,2] = -w[0]
    R[2,0] = -w[1]; R[2,1] = w[0]
    return R
