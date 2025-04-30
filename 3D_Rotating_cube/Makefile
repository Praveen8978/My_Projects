# Name of the final executable
TARGET = spin

# Source file(s)
SRCS = cube_spinning.cpp

# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

# Build rule
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

# Clean rule
.PHONY: clean
clean:
	rm -f $(TARGET)
