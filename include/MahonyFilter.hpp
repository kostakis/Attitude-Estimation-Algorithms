#ifndef MAHONYFILTER_HPP
#define MAHONYFILTER_HPP

#include "pch.h"

class MahonyFilter {
  public:
    MahonyFilter(double dt, double kp);

    void setIMUData(const Eigen::MatrixXd& gyroData, const Eigen::MatrixXd& accelData);
    void predictForAllData();

    const Eigen::VectorXd& getRollEstimation() const;
    Eigen::VectorXd& getRollEstimationNonConst();

    const Eigen::VectorXd& getPitchEstimation() const;
    Eigen::VectorXd& getPitchEstimationNonConst();
  private:
    double dt;
    double kp;

    Eigen::Matrix3d rHat;

    // IMU Input
    Eigen::MatrixXd accelometerData;
    Eigen::MatrixXd gyroData;

    // Predicitons - In rad (based on input)
    Eigen::VectorXd rollEstimation;
    Eigen::VectorXd pitchEstimation;

    void update(const Eigen::Vector3d& omega_y, const Eigen::Matrix3d& R_y);

    // TODO Check why this is needed
    void orthonormalize() {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(rHat, Eigen::ComputeFullU | Eigen::ComputeFullV);

        rHat = svd.matrixU() * svd.matrixV().transpose();

        // Ensure det(R̂) = 1 which is a condition in the SO(3) space
        if(rHat.determinant() < 0) { // Fixed
            Eigen::Matrix3d U = svd.matrixU();
            U.col(2) *= -1;
            rHat = U * svd.matrixV().transpose(); // Fixed
        }
    }
};

#endif // MAHONYFILTER_HPP
