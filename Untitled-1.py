"""
Simulación de drones interceptores con asignación descentralizada.

Regla de asignación (sin comunicación entre drones):
    Cada dron observa el mismo mapa (posiciones de misiles y drones aliados)
    y, para cada misil, calcula qué dron está más cerca. Si el más cercano
    es él mismo (desempate por menor ID), ese misil es suyo.

Mejoras:
    - Interceptación predictiva: el dron apunta a donde ESTARÁ el misil.
    - Posición defensiva: los drones sin objetivo se colocan entre los
      misiles restantes y la base, en formación separada por ID.
    - Los drones explotan al interceptar (sacrificio).

Uso:
    python main.py               # animación en pantalla (random cada vez)
    python main.py --gif         # guarda demo.gif sin abrir ventana
    python main.py --seed 42     # repite una simulación exacta
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- Parámetros ----------------
N_DRONES = 6
N_MISSILES = 6
WORLD = 100.0               # mundo cuadrado de 100x100
DRONE_SPEED = 1.2
MISSILE_SPEED = 0.7
KILL_RADIUS = 2.0           # distancia a la que el dron intercepta
BASE_POS = np.array([50.0, 5.0])
REASSIGN_EVERY = 2          # frames entre re-asignaciones (casi inmediata)
FORMATION_GAP = 6.0         # separación entre drones en formación defensiva
FRAMES = 400

# Sin semilla fija: cada ejecución genera posiciones de misiles distintas.
# Si quieres repetir una simulación exacta: python main.py --seed 42
_seed = None
if "--seed" in sys.argv:
    _seed = int(sys.argv[sys.argv.index("--seed") + 1])
rng = np.random.default_rng(_seed)

class Missile:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        # Velocidad: vector constante apuntando a la base
        direction = BASE_POS - self.pos
        self.vel = direction / np.linalg.norm(direction) * MISSILE_SPEED
        self.alive = True

    def update(self):
        self.pos += self.vel
        if np.linalg.norm(BASE_POS - self.pos) < 1.0:
            self.alive = False  # impactó la base

def intercept_point(drone_pos, missile):
    """Punto futuro donde el dron cortará el paso al misil.

    Resuelve: ¿en qué tiempo t la distancia dron→punto futuro del misil
    es exactamente la que el dron puede recorrer en t?
    """
    r = missile.pos - drone_pos          # posición relativa
    v = missile.vel                      # velocidad del misil
    s = DRONE_SPEED
    # Ecuación cuadrática: |r + v·t| = s·t
    a = v @ v - s * s
    b = 2 * (r @ v)
    c = r @ r
    t = None
    if abs(a) < 1e-9:                    # caso degenerado (velocidades casi iguales)
        if abs(b) > 1e-9:
            t = -c / b
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            t1 = (-b + np.sqrt(disc)) / (2 * a)
            t2 = (-b - np.sqrt(disc)) / (2 * a)
            positivos = [x for x in (t1, t2) if x > 0]
            if positivos:
                t = min(positivos)       # el encuentro más próximo
    if t is None:
        # No hay solución: el misil es más rápido y se aleja.
        # Último recurso: perseguir su posición actual.
        return missile.pos
    return missile.pos + v * t

class Drone:
    def __init__(self, drone_id, pos):
        self.id = drone_id
        self.pos = np.array(pos, dtype=float)
        self.target = None    # misil asignado
        self.alive = True     # al interceptar, el dron explota

    def update(self, hold_pos=None):
        if not self.alive:
            return
        if self.target is not None and self.target.alive:
            # Apuntar al punto de interceptación, no a la posición actual
            aim = intercept_point(self.pos, self.target)
            direction = aim - self.pos
            dist = np.linalg.norm(direction)
            if dist <= KILL_RADIUS:
                # Interceptación: el misil y el dron se destruyen
                self.target.alive = False
                self.alive = False
                self.target = None
            elif dist > 0:
                self.pos += direction / dist * DRONE_SPEED
        else:
            self.target = None
            # Sin objetivo: ir a SU puesto en la formación defensiva
            if hold_pos is not None:
                direction = hold_pos - self.pos
                dist = np.linalg.norm(direction)
                if dist > 1.0:
                    self.pos += direction / dist * DRONE_SPEED

def defensive_position(missiles, drone_id, n_drones):
    """Puesto de espera de CADA dron: a mitad de camino entre la base y
    el centro de los misiles vivos, pero con un desplazamiento lateral
    según su ID para que no se amontonen todos en el mismo punto."""
    alive = [m for m in missiles if m.alive]
    if not alive:
        anchor = BASE_POS + np.array([0.0, 15.0])  # formación sobre la base
    else:
        center = np.mean([m.pos for m in alive], axis=0)
        anchor = (BASE_POS + center) / 2
    # Abanico horizontal centrado: los IDs se reparten alrededor del ancla
    offset = (drone_id - (n_drones - 1) / 2) * FORMATION_GAP
    return anchor + np.array([offset, 0.0])

def assign(drones, missiles):
    """Regla 'el más cercano gana', calculada por cada dron sobre el mismo mapa."""
    alive_drones = [d for d in drones if d.alive]
    alive_missiles = [m for m in missiles if m.alive]
    for d in alive_drones:
        d.target = None
    taken = set()
    for d in alive_drones:
        best = None
        best_dist = np.inf
        for m in alive_missiles:
            if id(m) in taken:
                continue
            # Distancia de este dron al misil
            my_dist = np.linalg.norm(m.pos - d.pos)
            # ¿Hay algún aliado VIVO más cerca del mismo misil?
            closest_drone = min(alive_drones,
                                key=lambda o: (np.linalg.norm(m.pos - o.pos), o.id))
            if closest_drone is d and my_dist < best_dist:
                best = m
                best_dist = my_dist
        if best is not None:
            d.target = best
            taken.add(id(best))

# ---------------- Inicialización ----------------
drones = [Drone(i, [10 + i * (80 / max(N_DRONES - 1, 1)), 20 + rng.uniform(-3, 3)])
          for i in range(N_DRONES)]
missiles = [Missile([rng.uniform(5, 95), rng.uniform(85, 100)])
            for _ in range(N_MISSILES)]

fig, ax = plt.subplots(figsize=(8, 8))

def draw(frame):
    if frame % REASSIGN_EVERY == 0:
        assign(drones, missiles)
    for m in missiles:
        if m.alive:
            m.update()
    for d in drones:
        d.update(defensive_position(missiles, d.id, N_DRONES))

    ax.clear()
    ax.set_xlim(0, WORLD)
    ax.set_ylim(0, WORLD)
    ax.set_title(f"Drones interceptores — frame {frame}")
    ax.scatter(*BASE_POS, marker="s", s=300, c="green", label="Base")

    for m in missiles:
        if m.alive:
            ax.scatter(*m.pos, marker="v", s=120, c="red")
    for d in drones:
        if not d.alive:
            continue
        ax.scatter(*d.pos, marker="^", s=120, c="blue")
        ax.text(d.pos[0] + 1, d.pos[1] + 1, str(d.id), fontsize=8)
        if d.target is not None and d.target.alive:
            ax.plot([d.pos[0], d.target.pos[0]], [d.pos[1], d.target.pos[1]],
                    "b--", alpha=0.3)

    alive = sum(m.alive for m in missiles)
    drones_left = sum(d.alive for d in drones)
    ax.text(2, 2, f"Misiles activos: {alive}   Drones: {drones_left}", fontsize=10)

if "--gif" in sys.argv:
    from matplotlib.animation import PillowWriter
    anim = FuncAnimation(fig, draw, frames=FRAMES)
    anim.save("demo.gif", writer=PillowWriter(fps=20))
    print("Guardado demo.gif")
else:
    anim = FuncAnimation(fig, draw, frames=FRAMES, interval=50)
    plt.show()