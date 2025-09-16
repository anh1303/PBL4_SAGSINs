from vpython import *
from math import sin, cos, radians, pi, degrees
import sys, os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Classes')))
from network import Network
from satellite import Satellite
from gs import Gs
from ss import Ss

############################################
# Enhanced Visual Constants
############################################
# Colors with better aesthetics
COLORS = {
    'earth': color.blue * 0.8,
    'atmosphere': vector(0.3, 0.6, 1.0),
    'ground_station': vector(0.2, 0.8, 1.0),  # Cyan
    'ground_station_dish': vector(1.0, 0.6, 0.0),  # Orange
    'sea_station': vector(1.0, 0.8, 0.0),  # Gold
    'leo_satellite': vector(1.0, 0.3, 0.3),  # Red
    'meo_satellite': vector(0.3, 1.0, 0.3),  # Green
    'geo_satellite': vector(0.8, 0.3, 1.0),  # Purple
    'uav': vector(1.0, 0.0, 1.0),  # Magenta
    'connection': vector(0.0, 1.0, 0.8),  # Aqua
    'orbit_line': vector(0.5, 0.5, 0.5),  # Gray
    'stars': color.white
}

############################################
# Helpers
############################################
def latlon_to_vec(lat_deg, lon_deg, R=1.0, alt=0.0):
    """Convert lat/lon to 3D vector with correct Earth coordinate system"""
    lat = radians(lat_deg)
    lon = radians(lon_deg)
    r = R + alt
    # Correct spherical to Cartesian conversion for Earth coordinates
    # X: towards 0°N, 0°E (Greenwich meridian)
    # Y: towards 0°N, 90°E 
    # Z: towards North pole
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

def create_starfield(num_stars=500):
    """Create a beautiful starfield background"""
    stars = []
    for _ in range(num_stars):
        # Random position on a large sphere
        theta = random.uniform(0, 2*pi)
        phi = random.uniform(0, pi)
        r = 15  # Far from Earth
        pos = vector(r*sin(phi)*cos(theta), r*sin(phi)*sin(theta), r*cos(phi))
        brightness = random.uniform(0.3, 1.0)
        size = random.uniform(0.02, 0.05)
        star = sphere(pos=pos, radius=size, color=COLORS['stars']*brightness, emissive=True)
        stars.append(star)
    return stars

############################################
# Enhanced Scene Configuration
############################################
scene.title = "🛰️ SAGSINs 3D — Enhanced Visualization"
scene.width = 1400
scene.height = 900
scene.background = vector(0.05, 0.05, 0.15)  # Deep space blue
scene.forward = vector(-3, 2, -3)
scene.range = 8
scene.userzoom = True
scene.userspin = True

# Enhanced lighting
scene.lights = []  # Remove default lights
distant_light(direction=vector(1, 0.5, 0.5), color=color.white * 0.8)
distant_light(direction=vector(-0.5, -1, 0.3), color=vector(0.3, 0.4, 0.6))

# Create starfield
stars = create_starfield(300)

# Earth with enhanced visuals
R_earth = 1.0
earth = sphere(pos=vector(0,0,0), radius=R_earth, texture=textures.earth)
earth.emissive = False

# Atmospheric glow effect
atmosphere = sphere(pos=vector(0,0,0), radius=R_earth*1.02, 
                   color=COLORS['atmosphere'], opacity=0.3)

earth_spin_rate = 2*pi / 60.0

############################################
# Load network
############################################
network = Network()
vis_nodes = {}
SCALE = 0.6  # Slightly larger scale for better visibility

# Store original positions for GS and SS
original_positions = {}

# For orbital visualization
orbit_rings = []

############################################
# Enhanced Ground stations
############################################
for gs in network.groundstations.values():
    base = latlon_to_vec(gs.position["lat"], gs.position["lon"], R=R_earth)
    axis_vec, _ = orient_outward(base)
    
    # Enhanced base platform
    platform = cylinder(pos=base, axis=axis_vec*0.01*SCALE, 
                        radius=0.02*SCALE, color=vector(0.4, 0.4, 0.4))
    
    # Taller, more detailed mast
    mast = cylinder(pos=base + axis_vec*0.01*SCALE, axis=axis_vec*0.06*SCALE, 
                    radius=0.002*SCALE, color=COLORS['ground_station'])
    
    # Enhanced dish with support
    dish_pos = mast.pos + mast.axis
    dish = cone(pos=dish_pos, axis=axis_vec*0.03*SCALE, 
                radius=0.02*SCALE, color=COLORS['ground_station_dish'])
    
    # Add a subtle glow
    glow = sphere(pos=dish_pos, radius=0.035*SCALE, 
                  color=COLORS['ground_station'], opacity=0.2)
    
    # Enhanced label with background
    lbl = label(pos=dish.pos + axis_vec*0.045*SCALE, text=gs.id, 
                height=12, box=True, opacity=0.8, color=color.white,
                background=vector(0.1, 0.1, 0.3))
    
    vis_nodes[gs.id] = {
        "type":"gs", "lat":gs.position["lat"], "lon":gs.position["lon"],
        "platform":platform, "mast":mast, "dish":dish, "glow":glow, "label":lbl
    }
    original_positions[gs.id] = {"lat": gs.position["lat"], "lon": gs.position["lon"]}

############################################
# Sea stations - REMOVED per user request
############################################
# Sea stations have been removed from the visualization

############################################
# Enhanced Satellites with orbital rings
############################################
for sat in network.satellites.values():
    p_hat, q_hat = great_circle_basis(sat.orbit["inclination"], sat.orbit["raan"])
    
    # Determine satellite type and visual properties
    alt_km = sat.position["alt"]/1000
    if alt_km < 2000:  # LEO
        orb_r = R_earth + 0.08
        sat_color = COLORS['leo_satellite']
        sat_type = "LEO"
        trail_retain = 600
    elif alt_km < 35786:  # MEO
        orb_r = R_earth + 2.0
        sat_color = COLORS['meo_satellite'] 
        sat_type = "MEO"
        trail_retain = 400
    else:  # GEO
        orb_r = R_earth + 6.0
        sat_color = COLORS['geo_satellite']
        sat_type = "GEO"
        trail_retain = 200
    
    # Create orbital ring for visual reference
    orbit_points = []
    num_points = 100
    for i in range(num_points + 1):
        theta_ring = 2*pi*i/num_points
        ring_pos = orb_r*(cos(theta_ring)*p_hat + sin(theta_ring)*q_hat)
        orbit_points.append(ring_pos)
    
    orbit_ring = curve(pos=orbit_points, color=COLORS['orbit_line'], 
                       radius=0.001*SCALE, opacity=0.4)
    orbit_rings.append(orbit_ring)
    
    # Enhanced satellite object
    theta = sat.last_theta
    sat_pos = orb_r*(cos(theta)*p_hat + sin(theta)*q_hat)
    
    # Main satellite body
    s_obj = sphere(pos=sat_pos, radius=0.018*SCALE, color=sat_color, 
                   make_trail=True, retain=trail_retain)
    s_obj.emissive = True
    
    # Solar panels
    panel1 = box(pos=sat_pos, length=0.04*SCALE, height=0.008*SCALE, 
                 width=0.002*SCALE, color=vector(0.2, 0.2, 0.4))
    panel2 = box(pos=sat_pos, length=0.04*SCALE, height=0.008*SCALE, 
                 width=0.002*SCALE, color=vector(0.2, 0.2, 0.4))
    
    # Antenna
    antenna = cylinder(pos=sat_pos, axis=vector(0,0,0.025*SCALE), 
                      radius=0.001*SCALE, color=color.white)
    
    # Glow effect
    glow = sphere(pos=sat_pos, radius=0.035*SCALE, 
                  color=sat_color, opacity=0.1)
    
    # Enhanced label
    lbl = label(pos=sat_pos + vector(0.03*SCALE,0.03*SCALE,0.03*SCALE),
                text=f"{sat.id} ({sat_type})", height=10, box=True, opacity=0.9,
                background=vector(0.1, 0.1, 0.2))
    
    vis_nodes[sat.id] = {
        "type":"sat", "obj":s_obj, "panel1":panel1, "panel2":panel2, 
        "antenna":antenna, "glow":glow, "label":lbl,
        "p_hat":p_hat, "q_hat":q_hat, "r":orb_r,
        "theta":theta, "omega":2*pi/30, "sat_obj":sat,
        "orbit_ring":orbit_ring, "sat_type":sat_type
    }

# ############################################
# # Enhanced UAVs
# ############################################
# uavs = []
# num_uavs = 3
# for k in range(num_uavs):
#     lat = 15 * sin(k * 2*pi/3)  # Varied latitudes
#     lon = -120 + k*60
#     pos = latlon_to_vec(lat, lon, R=R_earth*1.03)
    
#     # UAV body
#     u = sphere(pos=pos, radius=0.012*SCALE, color=COLORS['uav'], 
#                make_trail=True, retain=150)
#     u.emissive = True
    
#     # Propellers (simple representation)
#     prop1 = ring(pos=pos, axis=vector(0,0,1), radius=0.008*SCALE, 
#                 thickness=0.001*SCALE, color=color.white, opacity=0.6)
#     prop2 = ring(pos=pos, axis=vector(0,0,1), radius=0.008*SCALE, 
#                 thickness=0.001*SCALE, color=color.white, opacity=0.6)
    
#     uavs.append({
#         "obj":u, "prop1":prop1, "prop2":prop2, "lat":lat, "lon":lon, 
#         "omega":0.8+k*0.2, "prop_angle":0
#     })

############################################
# Enhanced UI and Controls
############################################
paused = False
mode = 1  # 1: all, 2: GS, 3: SS, 4: LEO, 5: GEO
show_orbits = True
show_connections = False

# UI Text
info_text = text(text="🛰️ SAGSINs Network Visualization\n" +
                     "Space: Pause/Resume | 1,2,4,5: Filter modes | O: Toggle orbits | C: Toggle connections",
                 pos=vector(-7, 6, 0), height=0.15, color=color.white,
                 background=vector(0.1, 0.1, 0.3), opacity=0.8)

mode_text = text(text="Mode: All Objects", pos=vector(-7, 5.5, 0), 
                height=0.12, color=color.yellow)

def keydown(evt):
    global paused, mode, show_orbits, show_connections
    k = evt.key.lower()
    if k==' ':
        paused = not paused
    elif k in ['1','2','4','5']:  # Removed mode 3 since sea stations are gone
        mode = int(k)
        # Clear satellite trails for clean view
        for vis in vis_nodes.values():
            if vis["type"]=="sat":
                vis["obj"].clear_trail()
        update_visibility()
    elif k=='3':
        # Mode 3 disabled - show message instead
        print("Mode 3 (Sea Stations) has been removed from this visualization")
    elif k=='o':
        show_orbits = not show_orbits
        update_orbit_visibility()
    elif k=='c':
        show_connections = not show_connections

scene.bind('keydown', keydown)

def update_orbit_visibility():
    for vis in vis_nodes.values():
        if vis["type"]=="sat":
            vis["orbit_ring"].visible = show_orbits

def update_visibility():
    mode_names = {1:"All Objects", 2:"Ground Stations", 
                  4:"LEO Satellites", 5:"GEO Satellites"}
    mode_text.text = f"Mode: {mode_names.get(mode, 'Unknown')}"
    
    for node_id, vis in vis_nodes.items():
        ntype = vis["type"]
        show = False
        
        # Determine if this object should be shown
        if mode == 1: 
            show = True
        elif mode == 2 and ntype == "gs": 
            show = True
        elif mode == 4 and ntype == "sat":
            incl = vis["sat_obj"].orbit.get("inclination", 0)
            if incl < 60: 
                show = True
        elif mode == 5 and ntype == "sat":
            incl = vis["sat_obj"].orbit.get("inclination", 0)
            if incl < 5: 
                show = True
            
        # Update visibility for all components based on type
        try:
            if ntype == "gs":
                for comp in ["platform", "mast", "dish", "glow", "label"]:
                    if comp in vis and hasattr(vis[comp], 'visible'):
                        vis[comp].visible = show
            elif ntype == "sat":
                for comp in ["obj", "panel1", "panel2", "antenna", "glow", "label", "orbit_ring"]:
                    if comp in vis and hasattr(vis[comp], 'visible'):
                        if comp == "orbit_ring":
                            vis[comp].visible = show and show_orbits
                        else:
                            vis[comp].visible = show
        except Exception as e:
            print(f"Error updating visibility for {node_id}: {e}")
            continue

############################################
# Animation Loop
############################################
earth_rotation_angle = 0
frame_count = 0

def rotate_vector_z(v, angle_rad):
    """Rotate a vector around Z-axis by angle_rad"""
    x_new = cos(angle_rad)*v.x - sin(angle_rad)*v.y
    y_new = sin(angle_rad)*v.x + cos(angle_rad)*v.y
    return vector(x_new, y_new, v.z)

while True:
    dt = 1/60
    rate(60)
    frame_count += 1

    if paused: continue

    # Update Earth rotation angle
    earth_rotation_angle += earth_spin_rate * dt
    
    # Rotate Earth and atmosphere
    earth.rotate(angle=earth_spin_rate*dt, axis=vector(0,0,1))
    atmosphere.rotate(angle=earth_spin_rate*dt*0.95, axis=vector(0,0,1))

    # Update GS positions (rotate with Earth)
    for node_id, vis in vis_nodes.items():
        if vis["type"] == "gs":
            orig = original_positions[node_id]
            orig_vec = latlon_to_vec(orig["lat"], orig["lon"], R=R_earth)
            new_pos = rotate_vector_z(orig_vec, earth_rotation_angle)
            axis_vec,_ = orient_outward(new_pos)

            vis["platform"].pos = new_pos
            vis["platform"].axis = axis_vec*0.01*SCALE

            vis["mast"].pos = new_pos + axis_vec*0.01*SCALE
            vis["mast"].axis = axis_vec*0.06*SCALE

            dish_pos = vis["mast"].pos + vis["mast"].axis
            vis["dish"].pos = dish_pos
            vis["dish"].axis = axis_vec*0.03*SCALE

            vis["glow"].pos = dish_pos
            vis["label"].pos = dish_pos + axis_vec*0.045*SCALE

    # Update satellites positions
    for node_id, vis in vis_nodes.items():
        if vis["type"]=="sat":
            vis["theta"] = (vis["theta"] + vis["omega"]*dt) % (2*pi)
            new_pos = vis["r"]*(cos(vis["theta"])*vis["p_hat"] + sin(vis["theta"])*vis["q_hat"])

            vis["obj"].pos = new_pos
            vis["glow"].pos = new_pos
            vis["label"].pos = new_pos + vector(0.03*SCALE,0.03*SCALE,0.03*SCALE)

            panel_axis = cross(vis["p_hat"], vis["q_hat"])
            vis["panel1"].pos = new_pos
            vis["panel1"].axis = panel_axis
            vis["panel2"].pos = new_pos
            vis["panel2"].axis = -panel_axis

            vis["antenna"].pos = new_pos

    # Gentle pulsing atmosphere
    atmosphere.opacity = 0.25 + 0.05*sin(frame_count*0.02)
