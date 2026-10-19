#ifndef EXPLICITCOMPLEMENTARYFILTER_HPP
#define EXPLICITCOMPLEMENTARYFILTER_HPP

#include "pch.h"

/**
 * Explicit Complementary Filter on SO(3)
 *
 * Based on: Mahony et al. "Nonlinear Complementary Filters on the Special Orthogonal Group"
 * IEEE Transactions on Automatic Control, 2008
 *
 * This filter works directly with vectorial measurements (accelerometer/magnetometer)
 * without requiring full attitude reconstruction. It implements Section V of the paper
 * with bias estimation (Theorem 5.1 and Corollary 5.2).
 *
 * Key advantages:
 * - No algebraic attitude reconstruction required
 * - Works with single vector measurement (accelerometer only)
 * - Gyro bias estimation for all 3 axes
 * - Low computational cost, ideal for embedded systems
 */
class ExplicitComplementaryFilter {
  public:
    /**
     * Constructor
     * @param dt Sample time in seconds
     * @param kp Proportional gain for correction term (typical: 0.5 - 2.0 rad/s)
     * @param ki Integral gain for bias estimation (typical: 0.1 - 0.5 rad/s)
     */
    ExplicitComplementaryFilter(double dt, double kp, double ki);


    void setIMUData(const Eigen::MatrixXd& gyroData, const Eigen::MatrixXd& accelData);

    void predictForAllData();

    const Eigen::VectorXd& getRollEstimation() const;
    Eigen::VectorXd& getRollEstimationNonConst();


    const Eigen::VectorXd& getPitchEstimation() const;
    Eigen::VectorXd& getPitchEstimationNonConst();


    // Get gyro bias estimation
    const Eigen::Vector3d& getBiasEstimation() const;

  private:
    // Filter parameters
    double dt;  // Sample time
    double kp;  // Proportional gain
    double ki;  // Integral gain

    // State variables
    Eigen::Matrix3d rHat;           // Estimated rotation matrix R^
    Eigen::Vector3d biasEstimate;   // Estimated gyro bias

    // Inertial reference vectors (expressed in inertial frame {A})
    Eigen::Vector3d v0_gravity;     // Gravity direction: e3 = [0, 0, 1]ᵀ

    // IMU Input
    Eigen::MatrixXd accelerometerData;
    Eigen::MatrixXd gyroData;

    // Predictions - In radians
    Eigen::VectorXd rollEstimation;
    Eigen::VectorXd pitchEstimation;

    /**
     * Single filter update step
     * @param omega_y Gyroscope measurement (biased)
     * @param v_measured Measured vector in body frame (normalized accelerometer)
     */
    void update(const Eigen::Vector3d& omega_y, const Eigen::Vector3d& v_measured);

    /**
     * Compute measurement-based correction term ωₘₑₛ
     * Equation (32c) from paper: ωₘₑₛ = Σ kᵢ(vᵢ × v̂ᵢ)
     *
     * @param v_measured Measured vector in body frame
     * @param v_estimated Estimated vector in body frame (R̂ᵀ v₀)
     * @return Correction vector ωₘₑₛ
     */
    Eigen::Vector3d computeOmegaMeasurement(const Eigen::Vector3d& v_measured,
                                             const Eigen::Vector3d& v_estimated) const;

    // Ensure R^ remains in SO(3) using SVD orthonormalization
    void orthonormalize();
};

#endif // EXPLICITCOMPLEMENTARYFILTER_HPP
