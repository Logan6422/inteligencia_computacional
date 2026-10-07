import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# TODO: guardar histórico de las pos de las part. para animación

N = 50
c1 = 0.1
c2 = 0.1
maxGen = 500
tolConverg = 50

historico_pos = []
historico_mejor_global = []

# Función 1:
x1 = np.linspace(-512, 512, 1024)
def func1(x):
    return -1 * x * np.sin(np.sqrt(np.abs(x)))     # La func. a minimizar

# Paso 1: Inicializamos valores
pos_particulas = np.array([random.uniform(-512, 512) for _ in range(N)])
mejor_particulas = pos_particulas   # Fijamos la mejor posición a la pos. inicial
mejor_global = pos_particulas[0]    # Fijamos la mejor pos. global igual a la primera part.
vel_particulas = np.zeros(N)

historico_pos.append(pos_particulas.copy())
historico_mejor_global.append(mejor_global.copy())


# Paso 2: Entrenamiento función 1
it = 0
convergencia = 0
mejor_global_prev = mejor_global
while(it < maxGen and convergencia != tolConverg):
    print(it)
    # Actualizamos valores de c/partícula
    for i in range(N):
        if (func1(pos_particulas[i]) < func1(mejor_particulas[i])):
            mejor_particulas[i] = pos_particulas[i].copy()
        if (func1(mejor_particulas[i]) < func1(mejor_global)):
            mejor_global = mejor_particulas[i].copy()

    # Actualizamos velocidades
    for i in range(N):
        r1 = random.uniform(0,1)
        r2 = random.uniform(0,1)
        vel_particulas[i] = vel_particulas[i] + c1*r1*(mejor_particulas[i] - pos_particulas[i]) + c2*r2*(mejor_global - pos_particulas[i])
        pos_particulas[i] = pos_particulas[i] + vel_particulas[i]
        if (pos_particulas[i] > 512):
            pos_particulas[i] = 512
        if (pos_particulas[i] < -512):
            pos_particulas[i] = -512

    if(mejor_global == mejor_global_prev):
        convergencia += 1
    else:
        convergencia = 0

    historico_pos.append(pos_particulas.copy())
    historico_mejor_global.append(mejor_global.copy())
    mejor_global_prev = mejor_global
    it += 1

print(f'Mejor global final:')
print(f'x = {mejor_global}')
print(f'y = {func1(mejor_global)}')

# Mostramos gráfica con las partículas
fig1 = plt.figure(figsize=(15, 10))
ax1 = fig1.add_subplot()
# ax1.set_box_aspect(None, zoom=1.3)
# fig1, ax1 = plt.subplots()
ax1.plot(x1,func1(x1))
scat_particulas = ax1.scatter(pos_particulas,func1(pos_particulas), c='r')
scat_mejorglobal = ax1.scatter(mejor_global,func1(mejor_global), c='g')
ax1.grid()

historico_mejor_global = np.array(historico_mejor_global)
# print(np.shape(historico_mejor_global))

def update(frame):
    # Scat particulas
    x = historico_pos[frame]
    y = func1(x)
    data = np.stack([x,y]).T
    scat_particulas.set_offsets(data)

    # Scat mejor global
    x = historico_mejor_global[frame]
    y = func1(x)
    data = np.stack([x,y]).T
    scat_mejorglobal.set_offsets(data)

    return scat_particulas, scat_mejorglobal

cant_frames1 = np.shape(historico_mejor_global)[0]

anim = animation.FuncAnimation(fig=fig1, func=update, frames=it, interval=50, repeat=False)
plt.show()


# Función 2
N = 100
maxGen = 1000
c1 = 0.01   # Corrección según experiencia local
c2 = 0.01   # Corrección según exp. global

x2 = np.linspace(-100, 100, 100)
y2 = np.linspace(-100, 100, 100)
def func2(x, y):
    return (x**2 + y**2)**(0.25) * (np.sin(50*(x**2 + y**2)**(0.1))**2 + 1)

# Paso 1: Inicializamos valores
pos_particulas2 = np.random.uniform(-100, 100, N)
pos_particulas2 = np.vstack([pos_particulas2, np.random.uniform(-100, 100, N)])
vel_particulas2 = np.zeros((2,N))
mejor_particulas2 = pos_particulas2
mejor_global2 = pos_particulas2[:,0]

historico_pos2 = []
historico_pos2.append(pos_particulas2.copy())
historico_mejor_global2 = []
historico_mejor_global2.append(mejor_global2.copy())

# Paso 2: Entrenamiento función 2
it = 0
convergencia = 0
mejor_global_prev = mejor_global2
while(it < maxGen and convergencia != tolConverg):
    print(it)
    # Actualizamos valores de c/partícula
    for i in range(N):
        if (func2(pos_particulas2[0,i], pos_particulas2[1,i]) <
            func2(mejor_particulas2[0,i], mejor_particulas2[1,i])):
            mejor_particulas2[:,i] = pos_particulas2[:,i]
        if (func2(mejor_particulas2[0,i],mejor_particulas2[1,i]) <
            func2(mejor_global2[0],mejor_global2[1])):
            mejor_global2 = mejor_particulas2[:,i].copy()

    # Actualizamos velocidades
    for i in range(N):
        r1 = np.random.uniform(0,1,2)
        r2 = np.random.uniform(0,1,2)
        vel_particulas2[:,i] = (vel_particulas2[:,i] +
                            c1*r1*(mejor_particulas2[:,i] - pos_particulas2[:,i]) +
                            c2*r2*(mejor_global2 - pos_particulas2[:,i]))
        pos_particulas2[:,i] = pos_particulas2[:,i] + vel_particulas2[:,i]
        if (pos_particulas2[0,i] > 100):
            pos_particulas2[0,i] = 100
        if (pos_particulas2[0,i] < -100):
            pos_particulas2[0,i] = -100
        if (pos_particulas2[1,i] > 100):
            pos_particulas2[1,i] = 100
        if (pos_particulas2[1,i] < -100):
            pos_particulas2[1,i] = -100

    if(np.array_equal(mejor_global2, mejor_global_prev)):
        convergencia += 1
    else:
        convergencia = 0

    historico_pos2.append(pos_particulas2.copy())
    historico_mejor_global2.append(mejor_global2.copy())
    mejor_global_prev = mejor_global2.copy()
    it += 1

print(f'Mejor global final 2:')
print(f'(x,y) = {mejor_global2[0]}, {mejor_global2[1]}')
print(f'y = {func2(mejor_global2[0], mejor_global2[1])}')

# Gráfica 2
X, Y = np.meshgrid(x2,y2)
Z = func2(X,Y)

fig2 = plt.figure(figsize=(15, 10))
ax2 = fig2.add_subplot(projection='3d')
ax2.plot_wireframe(X, Y, Z, linewidth=0.1, rstride=2, cstride=2,)
ax2.view_init(elev=10, azim=-60)
ax2.set_box_aspect(None, zoom=1.3)

scat_particulas2 = ax2.scatter3D(pos_particulas2[0,:], pos_particulas2[1,:],
 func2(pos_particulas2[0,:], pos_particulas2[1,:]),s=50, c='r')

scat_mejorglobal2 = ax2.scatter3D([mejor_global2[0]], [mejor_global2[1]],
 [func2(mejor_global2[0], mejor_global2[1])],s=100, c='g')

historico_pos2 = np.array(historico_pos2)
historico_mejor_global2 = np.array(historico_mejor_global2)

def update2(frame):
    # Scat de particulas
    x = historico_pos2[frame,0,:]
    y = historico_pos2[frame,1,:]
    z = func2(x,y)
    scat_particulas2._offsets3d = (x,y,z)

    # Scat de mejor global
    x = historico_mejor_global2[frame,0]
    y = historico_mejor_global2[frame,1]
    z = func2(x,y)
    scat_mejorglobal2._offsets3d = ([x],[y],[z])

    return scat_particulas2, scat_mejorglobal2

cant_frames2 = np.shape(historico_mejor_global2)[0]

anim = animation.FuncAnimation(fig=fig2, func=update2, frames=cant_frames2, interval=10, repeat=False)
plt.show()