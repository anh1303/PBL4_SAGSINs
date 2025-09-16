from vpython import *
from math import sin, cos, radians

#------------------------
# Constants
#------------------------
R_earth = 1.0
SCALE = 0.5

#------------------------
# Helper: lat/lon to vector
#------------------------
def latlon_to_vec(lat_deg, lon_deg, R=1.0, alt=0.0):
    lat = radians(lat_deg)
    lon = radians(lon_deg)
    # This maps VPython coordinates to match the texture orientation
    x = (R + alt) * cos(lat) * sin(lon)
    y = (R + alt) * sin(lat)
    z = (R + alt) * cos(lat) * cos(lon)
    return vector(x, y, z)

def orient_outward(pos_vec):
    axis = norm(pos_vec)
    if abs(dot(axis, vector(0,0,1))) > 0.9:
        up = norm(vector(0,1,0))
    else:
        up = norm(vector(0,0,1))
    return axis, up

#------------------------
# Create Earth
#------------------------
earth = sphere(pos=vector(0,0,0), radius=R_earth, texture=textures.earth)
# Rotate Earth so longitude 0 is at front
earth.rotate(angle=-3.14159/2, axis=vector(0,1,0))

#------------------------
# GS Hanoi
#------------------------
lat_hanoi = 21.0285
lon_hanoi = 105.8542

pos = latlon_to_vec(lat_hanoi, lon_hanoi, R=R_earth, alt=0.001)
axis_vec, _ = orient_outward(pos)

# Mast
mast = cylinder(pos=pos, axis=axis_vec*0.04*SCALE, radius=0.003*SCALE, color=color.white)
# Dish
dish = cone(pos=mast.pos + mast.axis, axis=axis_vec*0.025*SCALE, radius=0.015*SCALE, color=color.orange)
# Label
lbl = label(pos=dish.pos + axis_vec*0.035*SCALE, text="GS_Hanoi", height=10, box=False)

#------------------------
# Keep the window open
#------------------------
while True:
    rate(60)
