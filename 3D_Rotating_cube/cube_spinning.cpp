#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <unistd.h>



using namespace std;

float CalculateX(float, float, float);
float CalculateY(float, float, float);
float CalculateZ(float, float, float);
void ImageFrameCoords(float, float, float, char, vector<float>&);

float A{}, B{}, C{};
int DistanceToCamera {100};
int Width {160};
int Height {44};
float f {40};
vector<vector<char>> screen(Height, vector<char>(Width, '.'));


int main(){
    int CubeWidth {0};
    float resolution {0.6};
    while(true){
        for (auto& row : screen)
            fill(row.begin(), row.end(), ' ');
        vector<float> zBuffer(Width*Height);
        
        CubeWidth = 20;

        for(float cubeX = -CubeWidth; cubeX < CubeWidth ; cubeX+= resolution){
            for(float cubeY = -CubeWidth; cubeY < CubeWidth; cubeY += resolution){
                ImageFrameCoords(cubeX, cubeY, -CubeWidth, '@', zBuffer);
                ImageFrameCoords(CubeWidth, cubeY, cubeX, '$', zBuffer);
                ImageFrameCoords(-CubeWidth, cubeY, -cubeX, '~', zBuffer);
                ImageFrameCoords(-cubeX, cubeY, CubeWidth, '#', zBuffer);
                ImageFrameCoords(cubeX, -CubeWidth, -cubeY, ';', zBuffer);
                ImageFrameCoords(cubeX, CubeWidth, cubeY, '+', zBuffer);
            }
        }

        static_cast<void>(system("clear"));
        for (const auto& row : screen) {
            for (char c : row) cout << c;
            cout << '\n';
        }

        A += 0.05f;
        B += 0.05f;
        C += 0.01f;

        usleep(8000*2);
    }
    return 0;
}


float CalculateX(float i, float j, float k){
    return j * sin(A) * sin(B) * cos(C) - k * cos(A) * sin(B) * cos(C) +
         j * cos(A) * sin(C) + k * sin(A) * sin(C) + i * cos(B) * cos(C);
}

float CalculateY(float i, float j, float k){
    return j * cos(A) * cos(C) + k * sin(A) * cos(C) -
         j * sin(A) * sin(B) * sin(C) + k * cos(A) * sin(B) * sin(C) -
         i * cos(B) * sin(C);
}

float CalculateZ(float i, float j, float k){
    return k * cos(A) * cos(B) - j * sin(A) * cos(B) + i * sin(B);
}

void ImageFrameCoords(float CubeX, float CubeY, float CubeZ, char ch, vector<float>& zBuffer){
    float x,y,z;
    int xp, yp;
    double ooz;
    x = CalculateX(CubeX, CubeY, CubeZ);
    y = CalculateY(CubeX, CubeY, CubeZ);
    z = CalculateZ(CubeX, CubeY, CubeZ) + DistanceToCamera;

    ooz = 1/z;

    xp = static_cast<int>((Width/2) + f*x*ooz*2); // add an Horizontal Offset and a factor of 
    yp = static_cast<int>((Height/2) + f*y*ooz);

    if (xp >= 0 && xp < Width && yp >= 0 && yp < Height) {
        if (ooz > zBuffer.at(yp * Width + xp)) {
            zBuffer.at(yp * Width + xp) = ooz;
            screen.at(yp).at(xp) = ch;
        }
    }
};