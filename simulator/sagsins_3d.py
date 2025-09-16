# sagsins_3d_extended.py
from vpython import *
from math import sin, cos, radians, pi, degrees
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Classes')))
from node import node
from satellite import Satellite
from gs import Gs
from ss import Ss
from network import Network

#Contain information about all nodes
network = Network()

############################################
# Helpers (same as before)
############################################

def latlon_to_vec(lat_deg, lon_deg, R=1.0):
    lat = radians(lat_deg)
    lon = radians(lon_deg)
    x = R * cos(lat) * cos(lon)
    y = R * cos(lat) * sin(lon)
    z = R * sin(lat)
    return vector(x, y, z)

def great_circle_basis(incl_deg=53.0, raan_deg=0.0):
    inc = radians(incl_deg)
    raan = radians(raan_deg)
    Xn = vector(cos(raan), sin(raan), 0)
    Yn = vector(-sin(raan), cos(raan), 0)
    Zn = vector(0, 0, 1)
    p_hat = norm(Xn)
    q_hat = norm(Yn*cos(inc) + Zn*sin(inc))
    return p_hat, q_hat

def orient_outward(pos_vec):
    axis = norm(pos_vec)
    if abs(dot(axis, vector(0,0,1))) > 0.9:
        up = norm(vector(0,1,0))
    else:
        up = norm(vector(0,0,1))
    return axis, up

############################################
# Scene configuration
############################################
scene.title = "SAGSINs 3D viz — Earth + Satellites + Sea + UAVs"
scene.width = 1100
scene.height = 700
scene.background = color.black
scene.forward = vector(-1.8, 1.2, -1.5)
scene.range = 2.2

R_earth = 1.0

earth = sphere(pos=vector(0,0,0), radius=R_earth, texture=textures.earth)
earth_spin_rate = 2*pi / 60.0

############################################
# Ground stations
############################################
ground_sites = [
    (21.0278, 105.8342, "Hanoi GS"),
    (34.0522, -118.2437, "LA GS"),
    (48.8566, 2.3522, "Paris GS"),
    (10.762622, 106.660172, "HCM GS"),
]
ground_objs = []
for lat, lon, name in ground_sites:
    base = latlon_to_vec(lat, lon, R=R_earth)
    axis_vec, up_vec = orient_outward(base)
    mast = cylinder(pos=base, axis=axis_vec*0.08, radius=0.006, color=color.white)
    dish = cone(pos=mast.pos + mast.axis, axis=axis_vec*0.05, radius=0.03, color=color.orange)
    label(pos=dish.pos + axis_vec*0.07, text=name, xoffset=0, yoffset=0, space=0, height=12, box=False)
    ground_objs.append((mast, dish))

############################################
# Sea stations / ships
############################################
ship_lats = [10.0, -15.0]
ship_speeds = [0.9, 0.6]
num_ships_per_lane = 5
ships = []

for lane_idx, lat in enumerate(ship_lats):
    for k in range(num_ships_per_lane):
        lon = -150 + k*(300/num_ships_per_lane)
        pos = latlon_to_vec(lat, lon, R=R_earth*1.005)
        shp = sphere(pos=pos, radius=0.02, color=color.cyan, make_trail=True, retain=120)
        ships.append({"obj": shp, "lat": lat, "lon": lon, "omega": ship_speeds[lane_idx]})

# Additional sea stations (static)
sea_stations = [
    (0, -45, "Buoy A"),
    (-20, 60, "Buoy B"),
]
for lat, lon, name in sea_stations:
    pos = latlon_to_vec(lat, lon, R=R_earth*1.005)
    sphere(pos=pos, radius=0.025, color=color.yellow)
    label(pos=pos + vector(0,0.05,0), text=name, xoffset=0, yoffset=0, height=10, box=False)

############################################
# Satellites
############################################
sat_params = [
    {"alt":0.35, "incl":53.0, "raan":0.0, "color":color.red},
    {"alt":0.45, "incl":70.0, "raan":45.0, "color":color.green},
]

sats = []
for p in sat_params:
    p_hat, q_hat = great_circle_basis(p["incl"], p["raan"])
    orb_r = R_earth + p["alt"]
    theta = 0.0
    s = sphere(radius=0.03, color=p["color"], make_trail=True, retain=400)
    s.pos = orb_r*(cos(theta)*p_hat + sin(theta)*q_hat)
    lbl = label(text="Satellite", xoffset=10, yoffset=10, height=12, box=False, color=color.white)
    sats.append({"obj":s, "label":lbl, "p_hat":p_hat, "q_hat":q_hat, "r":orb_r, "theta":theta, "omega":2*pi/30})

############################################
# UAVs (orbiting around equator or small patrol)
############################################
uavs = []
num_uavs = 3
for k in range(num_uavs):
    lat = 0
    lon = -120 + k*60
    pos = latlon_to_vec(lat, lon, R=R_earth*1.02)
    u = sphere(pos=pos, radius=0.015, color=color.magenta, make_trail=True, retain=80)
    uavs.append({"obj":u, "lat":lat, "lon":lon, "omega":0.8 + k*0.2})

############################################
# Controls
############################################
paused = False
yaw_rate = radians(5)
pitch_rate = radians(5)

def keydown(evt):
    global paused
    k = evt.key.lower()
    if k == ' ':
        paused = not paused
    else:
        # trục ngang camera = vuông góc với forward và up
        right_axis = norm(cross(scene.forward, scene.up))
        if k == 'w':       # pitch up
            scene.forward = rotate(scene.forward, angle=-pitch_rate, axis=right_axis)
            scene.up = rotate(scene.up, angle=-pitch_rate, axis=right_axis)
        elif k == 's':     # pitch down
            scene.forward = rotate(scene.forward, angle=pitch_rate, axis=right_axis)
            scene.up = rotate(scene.up, angle=pitch_rate, axis=right_axis)
        elif k == 'a':     # yaw left
            scene.forward = rotate(scene.forward, angle=-yaw_rate, axis=vector(0,1,0))
            scene.up = rotate(scene.up, angle=-yaw_rate, axis=vector(0,1,0))
        elif k == 'd':     # yaw right
            scene.forward = rotate(scene.forward, angle=yaw_rate, axis=vector(0,1,0))

scene.bind('keydown', keydown)

############################################
# Simulation loop
############################################
dt = 1/60

while True:
    rate(60)
    if paused:
        continue

    # Spin Earth
    earth.rotate(angle=earth_spin_rate*dt, axis=vector(0,0,1))

    # Update satellites
    for s in sats:
        s["theta"] = (s["theta"] + s["omega"]*dt) % (2*pi)
        s["obj"].pos = s["r"]*(cos(s["theta"])*s["p_hat"] + sin(s["theta"])*s["q_hat"])
        s["label"].pos = s["obj"].pos + vector(0.05,0.05,0.05)

    # Move ships along lanes
    for s in ships:
        s["lon"] += degrees(s["omega"]*dt)
        if s["lon"] > 180: s["lon"] -= 360
        s["obj"].pos = latlon_to_vec(s["lat"], s["lon"], R=R_earth*1.005)

    # Move UAVs
    for u in uavs:
        u["lon"] += degrees(u["omega"]*dt)
        if u["lon"] > 180: u["lon"] -= 360
        u["obj"].pos = latlon_to_vec(u["lat"], u["lon"], R=R_earth*1.02)
