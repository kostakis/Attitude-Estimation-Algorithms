# MSc Thesis: Attitude Estimation Using IMU Data Fusion

**Attitude Estimation Algorithms for UAV Navigation: Classical Filtering vs Deep Learning Approaches**

*Author: Akis Kostakis*  
*MSc in Artificial Intelligence & Deep Learning*  
*Completed: October 2026*

This repository contains the complete implementation of my MSc thesis, comparing traditional sensor fusion algorithms with deep learning approaches for attitude estimation using Inertial Measurement Unit (IMU) data.

## Table of Contents
- [Overview](#overview)
- [Key Findings](#key-findings)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Data](#data)
- [Build & Run](#build--run)
- [Development Setup](#development-setup)
- [Algorithm Details](#algorithm-details)
- [Citation](#citation)

## Overview

This MSc Thesis project implements and compares multiple attitude estimation algorithms for Inertial Measurement Unit (IMU) data fusion:
- **Complementary Filter**: Basic weighted sensor fusion baseline
- **Mahony Filter**: Nonlinear complementary filter on SO(3) - best classical performing filter
- **Explicit Complementary Filter**: Vector-based complementary filter with bias estimation
- **Extended Kalman Filter (EKF)**: Quaternion-based probabilistic state estimation
- **LSTM Multi-Step Ahead Predictor**: Deep learning approach using LibTorch (C++)

The algorithms estimate roll and pitch angles from 6-DOF IMU measurements and are evaluated against ground truth data. This work was completed as part of an MSc in AI & Deep Learning.

## Key Findings

- **Mahony Filter**: Best performing classical filter with tuned parameters (kp=11)
- **Complementary Filter**: Baseline with optimal alpha=0.79 via grid search  
- **LSTM**: Multi-step ahead prediction with 10-step lookback window
- **EKF**: Quaternion-based state estimation with process/measurement noise modeling

All filters demonstrate real-time capability with sampling at 50Hz (dt=0.02s).

## Prerequisites

### Classical Filters (Core)
- **C++ Compiler**: g++ with C++14 support
- **Eigen3**: Linear algebra library (must be in system include paths) - **Version 3.4.1 required**
- **Make**: Build system
- **Python 3** with matplotlib: For result visualization (optional)
- **bear**: For generating compile_commands.json (optional, for IDE support)

### LSTM (Deep Learning Component)
- **LibTorch C++**: PyTorch C++ API for LSTM implementation
- **C++17**: Required for LibTorch compatibility
- **Python 3** with matplotlib, numpy, torch: For training and visualization

**Note**: Eigen 3.4.1 is **specifically required** - other versions may cause compilation failures.

## Project Structure

```
├── Data/              # IMU sensor data and ground truth
├── Results/           # Classical filter outputs and plots
├── RNN/               # LSTM deep learning implementation (LibTorch C++)
├── include/           # Header files (.hpp) - core filter interfaces
├── src/               # Implementation files (.cpp) - filter algorithms
├── build/             # Build artifacts (generated)
├── bin/               # Compiled executables (generated)
├── *FilterMain.cpp    # Entry points for each classical filter
└── AGENTS.md          # Complete documentation for AI agents
```

**Note**: The `Pappers/` and `MDFiles/` directories referenced in older versions have been consolidated into the main documentation.

## Data

All sensor data is located in the `Data/` directory from a 6-DOF IMU sensor.

### Input Files

#### Classical Filter Dataset (Original)

| File | Description | Format | Units |
|------|-------------|--------|-------|
| `gyro.csv` | Angular velocity (3-axis) | 3 columns: X, Y, Z | rad/s |
| `accel.csv` | Linear acceleration (3-axis) | 3 columns: X, Y, Z | m/s² |
| `angles.csv` | Ground truth angles | Column 1: Roll, Column 2: Pitch | radians |

**Note**: Movement profile during data collection: (??) | Ground truth captured using: (??)

#### LSTM Training Dataset (dataset_1.csv)

**Format:** 3397 samples × 9 columns
**Sampling Rate:** Ts = 0.03 sec (33.3 Hz)

| Column | Description | Units |
|--------|-------------|-------|
| 0 | Ground truth Roll angle | radians |
| 1 | Ground truth Pitch angle | radians |
| 2 | Ground truth Yaw angle | radians |
| 3 | Gyroscope ω_roll (rotation rate around roll axis) | rad/s |
| 4 | Gyroscope ω_pitch (rotation rate around pitch axis) | rad/s |
| 5 | Gyroscope ω_yaw (rotation rate around yaw axis) | rad/s |
| 6 | Control torque τ_roll (from PID controller) | N·m |
| 7 | Control torque τ_pitch (from PID controller) | N·m |
| 8 | Control torque τ_yaw (from PID controller) | N·m |

**Usage:** Multi-step ahead prediction with LSTM (see `RNN/` directory) 

#### LSTM Big Training Dataset (all_combined.csv)

The file all_combined.csv combines all the 7 datasets into a single file

| Column | Description | Units |
|--------|-------------|-------|
| 0 | Ground truth Roll angle |  |
| 1 | Ground truth Roll rate |  |
| 2 | Ground truth Pitch angle |  |
| 3 | pitch_rate |  |
| 4 | yaw_angle |  |
| 5 | yaw_rate |  |
| 6 | Control torque τ_roll (from PID controller) | N·m |
| 7 | Control torque τ_pitch (from PID controller) | N·m |
| 8 | Control torque τ_yaw (from PID controller) | N·m |

The all_combined_reorder is the correct order our LSTM needs.

## Build & Run

### Installation

Eigen3 is a compile dependency.  
Best way is to install it through your package manager

```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# macOS
brew install eigen
```

You can also download it from the release and extract to your preffered location, it's easy since it is only header library. Just make sure the location is in the compiler path

It only compiles with the **3.4.1** version.

3.4.1 release: wget https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz  
Extract and copy

### Build

```bash
# Clean previous builds
make clean

# Build all filters (parallel compilation)
make all -j

# Or build individual filters
make mahonyFilter              # Mahony filter only
make complementaryFilter       # Complementary filter only
make ekfFilter                 # EKF filter only
make explicitComplementaryFilter  # Explicit complementary filter only
```

### Run

```bash
# Run Mahony filter (recommended - best current performance)
./bin/mahony.out

# Run Complementary filter
./bin/complementary.out

# Run EKF filter
./bin/ekf.out

# Run Explicit Complementary filter
./bin/explicitCF.out
```

### LSTM (Deep Learning Component)

The LSTM implementation requires LibTorch C++ installation:

```bash
# Navigate to RNN directory
cd RNN

# Edit Makefile to set LIBTORCH_PATH to your libtorch installation
# Example: LIBTORCH_PATH = /path/to/libtorch

# Build LSTM components
make clean && make all

# Train LSTM model (default: 300 epochs)
./lstm.out

# Train with custom epochs
./lstm.out -epochs 1000

# Evaluate trained model
./lstmEval.out lstm_model_epoch_1000.pt --save-all  # Generates predictions
```

### Visualize Results

#### Classical Filters
After running a filter, visualize the results:

```bash
cd Results
python3 plotDiagrams.py
```

Output plots are saved in `Results/Figures/`.

#### LSTM Predictions
Visualize LSTM model predictions:

```bash
cd RNN
python3 plot_predictions.py 0 100  # Plot samples 0 to 100
python3 plot_loss_curve.py        # Plot training loss curve
python3 plot_rmse_per_step.py     # Plot RMSE per prediction step
```


## Development Setup

### clangd for VSCode (Recommended)

VSCode's default C++ IntelliSense can be slow. For better performance, use clangd:

1. Install the **clangd** extension in VSCode
2. Keep the **C/C++** extension installed (needed for debugging)
3. Add to VSCode `settings.json`:  
```json
{
    // Disable Microsoft C++ IntelliSense
    "C_Cpp.intelliSenseEngine": "disabled",
    "C_Cpp.autocomplete": "disabled",
    "C_Cpp.errorSquiggles": "disabled",
    
    // clangd settings
    "clangd.arguments": [
        "--background-index",
        "--clang-tidy",
        "--header-insertion=iwyu",
        "--completion-style=detailed",
        "--function-arg-placeholders",
        "--fallback-style=llvm"
    ],
    "clangd.path": "/usr/bin/clangd",
    
    // Keep C++ debugger functionality
    "C_Cpp.debugShortcut": true
}
```


### Generate compile_commands.json

For clangd to work properly, generate the compilation database:

```bash
# Install bear (if not already installed)
sudo apt install bear

# Generate compile_commands.json
make clean  # Important: must rebuild from scratch
bear -- make all
cat compile_commands.json  # Verify generation
```

## Algorithm Details

### Classical Filters
All classical filters follow a consistent API pattern:
1. **Constructor**: Takes algorithm parameters (dt, gain coefficients, etc.)
2. **setIMUData()**: Load gyroscope and accelerometer matrices
3. **predictForAllData()**: Process all timesteps
4. **getRollEstimation() / getPitchEstimation()**: Retrieve results as Eigen::VectorXd

### LSTM Architecture
- **Lookback window (K)**: 10 timesteps
- **Prediction horizon (N)**: 10 timesteps  
- **Batch size**: 32
- **Learning rate**: 0.001
- **Hidden units**: 64
- **Attention mechanism**: Optional (see `LSTMNetwork.h` vs `LSTMNetworkNoAttention.h`)

### Key Parameters
- **Mahony filter**: kp = 11 (tuned optimal value)
- **Complementary filter**: alpha = 0.79 (grid search optimized)
- **Sampling rate**: 50Hz (dt = 0.02s)
