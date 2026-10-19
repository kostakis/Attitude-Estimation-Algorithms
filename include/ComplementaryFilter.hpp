#ifndef COMPLEMENTARY_FILTER_HPP
#define COMPLEMENTARY_FILTER_HPP

#include "pch.h"

class ComplementaryFilter {
  public:
    ComplementaryFilter() = default;
    ~ComplementaryFilter() = default;

    ComplementaryFilter(double alphaC, double dt);

    void setIMUData(const Eigen::MatrixXd& gyroData, const Eigen::MatrixXd accelData);

    inline const Eigen::VectorXd& getRoll() const {
        return rollEstimation;
    }

    inline Eigen::VectorXd& getRoll() {
        return rollEstimation;
    }

    inline Eigen::VectorXd& getPitch() {
        return pitchEstimation;
    }

    void calculateRoll();
    void calculatePitch();

  private:
    // Some default values
    double dt{0.02};
    double alphaCoeff{0.9};

    Eigen::MatrixXd accelometerData;
    Eigen::MatrixXd gyroData;

    Eigen::VectorXd phiG;
    Eigen::VectorXd phiA;
    Eigen::VectorXd thetaG;
    Eigen::VectorXd thetaA;

    // Predictions
    Eigen::VectorXd rollEstimation;
    Eigen::VectorXd pitchEstimation;
};

#endif // COMPLEMENTARY_FILTER_HPP
