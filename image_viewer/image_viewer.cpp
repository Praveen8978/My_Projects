#include <iostream>
 #include <sstream>
#include <fstream>
#include <string>
#include <SDL2/SDL.h>

int main(){
    std::ifstream in_file("sample_image_b.ppm", std::ios::binary);

    if(!in_file.is_open()){
        std::cerr<< "Unable to open the file!!" << std::endl;
        return 1;
    }

    std::string line;
    // read the first line to check if it's P6(binary) format
    getline(in_file, line);
    if(line != "P6"){
        std::cerr << "File must be in Binary format!!" << std::endl;
        return 1;
    }

    int width{-1};
    int height{-1};

    // read the second line for dimensions
    getline(in_file, line);
    std::istringstream dimensions(line);
    dimensions >> width >> height;

    getline(in_file, line); // discarding the third line with max value info

    SDL_Window* pwindow = SDL_CreateWindow("Image_Window", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                            width, height, 0);
    
    SDL_Surface* psurface =  SDL_GetWindowSurface(pwindow);   
    SDL_Rect rect {0,0,1,1};

    Uint8 r, g, b;
    Uint32 color{0};

    for(int y{0}; y < height; y++){
        for(int x{0}; x < width; x++){

            r = static_cast<Uint8>(in_file.get());
            g = static_cast<Uint8>(in_file.get());
            b = static_cast<Uint8>(in_file.get());

            color = SDL_MapRGB(psurface->format, r, g, b);
            rect.x = x;
            rect.y = y;
            SDL_FillRect(psurface, &rect, color);

        }
    }
    SDL_UpdateWindowSurface(pwindow);

    SDL_Event event {};
    bool running {true};

    while (running){
        while(SDL_PollEvent(&event)){
            if (event.type == SDL_QUIT)
                running = false;
        }
        SDL_Delay(100);
    }
    
    return 0;
}