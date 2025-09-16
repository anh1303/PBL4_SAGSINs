from vpython import *
from math import sin, cos, radians, pi, degrees
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Classes')))
from network import Network
from satellite import Satellite
from gs import Gs
from ss import Ss
from colorsys import hsv_to_rgb

############################################
# Helpers
############################################
def latlon_to_vec(lat_deg, lon_deg, R=1.0, alt=0.0):
    lat = radians(lat_deg)
    lon = radians(lon_deg)
    r = R + alt
    x = r * cos(lat) * cos(lon)
    y = r * cos(lat) * sin(lon)
    z = r * sin(lat)
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

def generate_colors(N):
    colors = []
    for i in range(N):
        hue = i / N
        r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(vector(r, g, b))
    return colors

############################################
# Scene configuration
############################################
scene.title = "SAGSINs 3D — DB loaded"
scene.width = 1100
scene.height = 700
scene.background = color.black
scene.forward = vector(-3, 2, -3)
scene.range = 7
scene.userzoom = True
scene.userspin = True

R_earth = 1.0
earth = sphere(pos=vector(0,0,0), radius=R_earth, texture=textures.earth)
earth_spin_rate = 2*pi / 60.0

############################################
# Load network
############################################
network = Network()
vis_nodes = {}
SCALE = 0.5  # global scaling

############################################
# Ground stations
############################################
for gs in network.groundstations.values():
    base = latlon_to_vec(gs.position["lat"], gs.position["lon"], R=R_earth)
    axis_vec, _ = orient_outward(base)
    mast = cylinder(pos=base, axis=axis_vec*0.04*SCALE, radius=0.003*SCALE, color=color.white)
    dish = cone(pos=mast.pos + mast.axis, axis=axis_vec*0.025*SCALE, radius=0.015*SCALE, color=color.orange)
    lbl = label(pos=dish.pos + axis_vec*0.035*SCALE, text=gs.id, height=10, box=False)
    vis_nodes[gs.id] = {"type":"gs","lat":gs.position["lat"],"lon":gs.position["lon"],
                        "mast":mast,"dish":dish,"label":lbl}

############################################
# Sea stations
############################################
for ss in network.seastations.values():
    pos = latlon_to_vec(ss.position["lat"], ss.position["lon"], R=R_earth*1.005)
    shp = sphere(pos=pos, radius=0.015*SCALE, color=color.yellow)
    lbl = label(pos=pos + vector(0,0.03*SCALE,0), text=ss.id, height=8, box=False)
    vis_nodes[ss.id] = {"type":"ss","lat":ss.position["lat"],"lon":ss.position["lon"],
                         "obj":shp,"label":lbl}

############################################
# Satellites
############################################
sat_ids = list(network.satellites.keys())
sat_colors = generate_colors(len(sat_ids))

for idx, sat in enumerate(network.satellites.values()):
    p_hat, q_hat = great_circle_basis(sat.orbit["inclination"], sat.orbit["raan"])
    if sat.position["alt"]/1000 < 2000:  # LEO
        orb_r = R_earth + 0.05
    elif sat.position["alt"]/1000 < 35786:  # MEO
        orb_r = R_earth + 1.5
    else:  # GEO
        orb_r = R_earth + 5.6
    theta = sat.last_theta
    s_obj = sphere(radius=0.015*SCALE, color=sat_colors[idx], make_trail=True, retain=400)
    s_obj.pos = orb_r*(cos(theta)*p_hat + sin(theta)*q_hat)
    lbl = label(pos=s_obj.pos + vector(0.025*SCALE,0.025*SCALE,0.025*SCALE),
                text=sat.id, height=8, box=False)
    vis_nodes[sat.id] = {"type":"sat","obj":s_obj,"label":lbl,
                          "p_hat":p_hat,"q_hat":q_hat,"r":orb_r,
                          "theta":theta,"omega":2*pi/30,
                          "sat_obj":sat}

############################################
# Controls
############################################
paused = False
mode = 1  # 1: all, 2: GS, 3: SS, 4: LEO, 5: GEO

def keydown(evt):
    global paused, mode
    k = evt.key.lower()
    if k==' ':
        paused = not paused
    elif k in ['1','2','3','4','5']:
        mode = int(k)
        # Clear satellite trails for clean view
        for vis in vis_nodes.values():
            if vis["type"]=="sat":
                vis["obj"].clear_trail()
        update_visibility()

scene.bind('keydown', keydown)

def update_visibility():
    for node_id, vis in vis_nodes.items():
        ntype = vis["type"]
        show = False
        if mode==1: show = True
        elif mode==2 and ntype=="gs": show = True
        elif mode==3 and ntype=="ss": show = True
        elif mode==4 and ntype=="sat":
            incl = vis["sat_obj"].orbit.get("inclination",0)
            if incl<60: show=True
        elif mode==5 and ntype=="sat":
            incl = vis["sat_obj"].orbit.get("inclination",0)
            if incl<5: show=True
        if ntype=="gs":
            vis["mast"].visible = show
            vis["dish"].visible = show
            vis["label"].visible = show
        elif ntype=="ss":
            vis["obj"].visible = show
            vis["label"].visible = show
        elif ntype=="sat":
            vis["obj"].visible = show
            vis["label"].visible = show

############################################
# Simulation loop
############################################
dt = 1/60
while True:
    rate(60)
    if paused: continue

    # Rotate Earth
    earth.rotate(angle=earth_spin_rate*dt, axis=vector(0,0,1))
    earth_angle_dt = earth_spin_rate*dt

    # Update GS & SS positions (move with Earth rotation)
    for node_id, vis in vis_nodes.items():
        if vis["type"] in ["gs","ss"]:
            vis["lon"] += degrees(earth_angle_dt)
            if vis["lon"] > 180: vis["lon"] -= 360
            r = R_earth if vis["type"]=="gs" else R_earth*1.005
            pos = latlon_to_vec(vis["lat"], vis["lon"], R=r)
            if vis["type"]=="gs":
                axis_vec,_ = orient_outward(pos)
                vis["mast"].pos = pos
                vis["mast"].axis = axis_vec*0.04*SCALE
                vis["dish"].pos = pos + axis_vec*0.04*SCALE
                vis["dish"].axis = axis_vec*0.025*SCALE
                vis["label"].pos = vis["dish"].pos + axis_vec*0.035*SCALE
            elif vis["type"]=="ss":
                vis["obj"].pos = pos
                vis["label"].pos = pos + vector(0,0.03*SCALE,0)

    # Update satellites positions along orbit
    for node_id, vis in vis_nodes.items():
        if vis["type"]=="sat":
            vis["theta"] = (vis["theta"] + vis["omega"]*dt) % (2*pi)
            vis["obj"].pos = vis["r"]*(cos(vis["theta"])*vis["p_hat"] + sin(vis["theta"])*vis["q_hat"])
            vis["label"].pos = vis["obj"].pos + vector(0.025*SCALE,0.025*SCALE,0.025*SCALE)
