CXX = g++
CXX_STANDARD = c++14
CXXFLAGS = -std=$(CXX_STANDARD) -g
INCLUDES = -Iinclude
SRC_DIR = src
BUILD_DIR = build
BINARY_DIR = bin

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

COMPLEMENTARY_MAIN_OBJ = $(BUILD_DIR)/complementaryFilterMain.o
EKF_MAIN_OBJ = $(BUILD_DIR)/ekfFilterMain.o
MAHONYFILTER_MAIN_OBJ=$(BUILD_DIR)/mahonyFilterMain.cpp.o
EXPLICIT_CF_MAIN_OBJ=$(BUILD_DIR)/explicitComplementaryFilterMain.o

# Create build and binary directories if they don't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BINARY_DIR)

# Compile ALL source files to object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


# Complentary Filter
$(COMPLEMENTARY_MAIN_OBJ): complementaryFilterMain.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

complementaryFilter: $(OBJECTS) $(COMPLEMENTARY_MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $(BINARY_DIR)/complementary.out


# EKF Filter
$(EKF_MAIN_OBJ): ekfFilterMain.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

ekfFilter: $(OBJECTS) $(EKF_MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $(BINARY_DIR)/ekf.out


# Mahony Filter
$(MAHONYFILTER_MAIN_OBJ): mahonyFilterMain.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

mahonyFilter: $(OBJECTS) $(MAHONYFILTER_MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $(BINARY_DIR)/mahony.out


# Explicit Complementary Filter
$(EXPLICIT_CF_MAIN_OBJ): explicitComplementaryFilterMain.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

explicitComplementaryFilter: $(OBJECTS) $(EXPLICIT_CF_MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $(BINARY_DIR)/explicitCF.out


clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(BINARY_DIR)
	rm -rf *.out


# Build all filters
all: complementaryFilter ekfFilter mahonyFilter explicitComplementaryFilter


.PHONY: complementaryFilter ekfFilter mahonyFilter explicitComplementaryFilter clean
