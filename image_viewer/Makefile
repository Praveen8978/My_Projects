CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -g
SDLFLAGS := $(shell sdl2-config --cflags --libs)
TARGET   := iv
SRC      := image_viewer.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(SDLFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
