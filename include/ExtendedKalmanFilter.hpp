#ifndef EXTENDEDKALMANFILTER_HPP
#define EXTENDEDKALMANFILTER_HPP

#include "pch.h"

class ExtendedKalmanFilter {
  public:
    // Constructor
    ExtendedKalmanFilter(double dt, const Eigen::Vector3d& initial_accel);

    // Set IMU data (matches MahonyFilter pattern)
    void setIMUData(const Eigen::MatrixXd& gyroData, const Eigen::MatrixXd& accelData);

    // Main processing function (matches MahonyFilter pattern)
    void predictForAllData();

    // Get estimations (matches MahonyFilter pattern)
    const Eigen::VectorXd& getRollEstimation() const;
    Eigen::VectorXd& getRollEstimationNonConst();

    const Eigen::VectorXd& getPitchEstimation() const;
    Eigen::VectorXd& getPitchEstimationNonConst();

    // Get current state
    Eigen::VectorXd getState() const {
        return x;
    }
    Eigen::Vector4d getQuaternion() const {
        return x.head<4>();
    }
    Eigen::Vector3d getBias() const {
        return x.tail<3>();
    }

  private:
    // State: [q0, q1, q2, q3, bx, by, bz]^T (7x1)
    Eigen::VectorXd x;

    // Covariance matrix (7x7)
    Eigen::MatrixXd P;

    // Process noise covariance (7x7)
    Eigen::MatrixXd Q;

    // Measurement noise covariance (3x3)
    Eigen::MatrixXd R;

    // Time step
    double dt;

    // IMU Input (matches MahonyFilter pattern)
    Eigen::MatrixXd gyroData;
    Eigen::MatrixXd accelometerData;

    // Predictions - In rad (matches MahonyFilter pattern)
    Eigen::VectorXd rollEstimation;
    Eigen::VectorXd pitchEstimation;

    // Prediction step
    void predict(const Eigen::Vector3d& gyro);

    // Update step
    void update(const Eigen::Vector3d& accel);

    // Get roll and pitch from current quaternion state
    double getRollFromQuaternion() const;
    double getPitchFromQuaternion() const;

    // Quaternion utilities
    void normalizeQuaternion();
    Eigen::Matrix3d quaternionToRotationMatrix(const Eigen::Vector4d& q) const;

    // Jacobians
    Eigen::MatrixXd computeF(const Eigen::Vector3d& gyro_corrected) const;
    Eigen::MatrixXd computeH() const;
};

#endif // EXTENDEDKALMANFILTER_HPP
