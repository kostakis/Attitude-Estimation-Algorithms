#include "ComplementaryFilter.hpp"

#include <iostream>

ComplementaryFilter::ComplementaryFilter(double alphaC, double dtarg) {
    alphaCoeff = alphaC;
    dt = dtarg;
}

void ComplementaryFilter::setIMUData(const Eigen::MatrixXd& gyroData, const Eigen::MatrixXd accelData) {
    this->gyroData = gyroData;
    this->accelometerData = accelData;

    if(gyroData.rows() != accelData.rows()) {
        std::cout << "Wrong data\n";
        exit(-1);
    }

    int numSamples = static_cast<int>(gyroData.rows());
    rollEstimation.resize(numSamples);
    pitchEstimation.resize(numSamples);

    rollEstimation.setZero();
    pitchEstimation.setZero();

    phiA.resize(accelData.rows());
    phiA.setZero();

    phiG.resize(gyroData.rows());
    phiG.setZero();

    thetaA.resize(accelData.rows());
    thetaA.setZero();

    thetaG.resize(gyroData.rows());
    thetaG.setZero();
}

void ComplementaryFilter::calculateRoll() {
    // All data are init to zero no need to initialize
    for(int i = 1; i < gyroData.rows(); i++) { // We start from 1
        double ay = accelometerData(i, 1);
        double az = accelometerData(i, 2);

        phiG(i) = rollEstimation(i - 1) + gyroData(i, 0) * dt;
        phiA(i) = std::atan2(ay, az);

        double rollResult = alphaCoeff * phiG(i) + (1 - alphaCoeff) * phiA(i);
        rollEstimation(i) = rollResult;
    }
}

void ComplementaryFilter::calculatePitch() {
    const double diffConst = 1 - alphaCoeff;

    for(int i = 1; i < gyroData.rows(); i++) {
        double ax = accelometerData(i, 0);
        double ay = accelometerData(i, 1);
        double az = accelometerData(i, 2);
        double squareData = (ay * ay) + (az * az);

        thetaG(i) = pitchEstimation(i - 1) + gyroData(i, 1) * dt;
        thetaA(i) = std::atan2(ax, std::sqrt(squareData));

        double pitchResult = alphaCoeff * thetaG(i) + diffConst * thetaA(i);
        pitchEstimation(i) = pitchResult;
    }
}
