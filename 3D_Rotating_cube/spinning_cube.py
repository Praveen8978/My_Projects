import numpy as np
import time
import os

#variable initilization
CubeWidth = 20
Width, Height = 160, 44
distanceFromCam = 100
K1 = 40
incrementspeed = 0.6
A, B, C = 0, 0, 0
horizontal_Offset = -2*CubeWidth



sin, cos = np.sin, np.cos


def new_coordinates(x,y,z,A,B,C):
    # define the 3D rotation matrix
    rot_mat = np.array([[cos(B)*cos(C), cos(B)*sin(C), -sin(B)],
                      [sin(A)*sin(B)*cos(C)-cos(A)*sin(C), sin(A)*sin(B)*sin(C)+cos(A)*cos(C), sin(A)*cos(B)],
                      [cos(A)*sin(B)*cos(C)+sin(A)*sin(C), cos(A)*sin(B)*sin(C)-sin(A)*cos(C), cos(A)*cos(B)]])
    
    pos_cor = np.array([x ,y, z]).T
    # Matrix multiplication to get new coordinates
    new_pos_cor = rot_mat @ pos_cor

    return new_pos_cor

def calcualteSurface(x,y,z,ch):
    global zBuffer, buffer
    # getting the new coordinates after rotation
    new_x = new_coordinates(x,y,z, A, B, C)[0]
    new_y = new_coordinates(x,y,z, A, B, C)[1]
    new_z = new_coordinates(x,y,z, A, B, C)[2] + distanceFromCam

    ooz = 1/new_z

    xp = int(Width/2 + horizontal_Offset + K1*new_x*ooz)
    yp = int(Height/2 + K1*new_y*ooz)

    idx = xp + yp*Width

    if idx>0 and idx < Width*Height:
        if ooz > zBuffer[idx]:
            zBuffer[idx] = ooz
            buffer[idx] = ch

if __name__ == "__main__":
    while True:
        # Reset zBuffer and buffer only once per frame
        zBuffer = np.zeros(Width * Height)
        buffer = np.full(Width * Height, ' ', dtype=str)

        # Render all six faces of the cube
        for cubeX in np.arange(-CubeWidth, CubeWidth, incrementspeed):
            for cubeY in np.arange(-CubeWidth, CubeWidth, incrementspeed):
                
                calcualteSurface(cubeX, cubeY, -CubeWidth, '@') # Bottom face
                
                calcualteSurface(cubeX, cubeY, CubeWidth, '+') # Top face
                
                calcualteSurface(-CubeWidth, cubeY, -cubeX, '~') # Left face
                
                calcualteSurface(CubeWidth, cubeY, cubeX, '$') # Right face
                
                calcualteSurface(cubeX, -CubeWidth, -cubeY, ';') # Back face
               
                calcualteSurface(cubeX, CubeWidth, cubeY, '#')  # Front face

        
        print("\033[2J\033[H", end="")

       
        for k in range(0, Width * Height, 1):
            print(buffer[k], end='') if k % Width != 0 else print()

        
        A += 0.05
        B += 0.05
        C += 0.01

        
        time.sleep(0.00016)





    

    

    

