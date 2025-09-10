from vpython import *
import math

scene.title = "Earth Interactive Rotation"
scene.width = 900
scene.height = 600
scene.background = color.black
scene.range = 2.2
scene.userspin = False   # tắt xoay camera mặc định
scene.userzoom = True    # vẫn cho zoom


R_EARTH = 1.0
DEG = math.pi/180

def latlon_to_vec(lat_deg, lon_deg, r=R_EARTH):
    lat = lat_deg*DEG
    lon = lon_deg*DEG
    x = r*math.cos(lat)*math.cos(lon)
    y = r*math.sin(lat)
    z = r*math.cos(lat)*math.sin(lon)
    return vector(x,y,z)

# Trái Đất
earth = sphere(radius=R_EARTH, texture=textures.earth, shininess=0.6)

# Gom group xoay cùng Trái Đất
earth_group = [earth]

# Thêm vài trạm
def make_station(lat, lon, name, col=color.yellow):
    pos = latlon_to_vec(lat, lon, R_EARTH*1.01)
    n = norm(pos)
    a = cone(pos=pos, axis=n*0.06, radius=0.02, color=col)
    lbl = label(pos=pos*1.05, text=name, box=False, opacity=0, color=col, height=10)
    earth_group.extend([a, lbl])
    return a

make_station(21.0, 105.8, "Hà Nội", color.green)
make_station(10.8, 106.6, "HCMC", color.green)
make_station(16.0, 108.2, "Sea Đà Nẵng", color.cyan)

# -----------------------
# Xử lý xoay bằng chuột
# -----------------------
dragging = False
last_mouse = None
rotate_speed = 2.0

def on_mousedown(evt):
    global dragging, last_mouse
    if evt.button == 'left':
        dragging = True
        last_mouse = scene.mouse.pos

def on_mouseup(evt):
    global dragging, last_mouse
    dragging = False
    last_mouse = None

def on_mousemove(evt):
    global dragging, last_mouse
    if not dragging or last_mouse is None: 
        return
    curr = scene.mouse.pos
    dx = curr.x - last_mouse.x
    dy = curr.y - last_mouse.y
    last_mouse = curr
    
    angle_y = -dx * rotate_speed
    angle_x =  dy * rotate_speed
    
    for obj in earth_group:
        obj.rotate(angle=angle_y, axis=vector(0,1,0), origin=vector(0,0,0))
        obj.rotate(angle=angle_x, axis=vector(1,0,0), origin=vector(0,0,0))

scene.bind('mousedown', on_mousedown)
scene.bind('mouseup',   on_mouseup)
scene.bind('mousemove', on_mousemove)

# vòng lặp (chỉ cần để VPython chạy, không xoay tự động)
while True:
    rate(60)
    if scene.mouse.events:
        ev = scene.mouse.getevent()
        if ev.press == 'left':
            dragging = True
            last = scene.mouse.pos
        elif ev.release == 'left':
            dragging = False
    if dragging:
        new = scene.mouse.pos
        dx = new.x - last.x
        dy = new.y - last.y
        last = new
        earth.rotate(angle=-dx*2, axis=vector(0,1,0))
        earth.rotate(angle= dy*2, axis=vector(1,0,0))