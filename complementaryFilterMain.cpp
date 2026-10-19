#include "ComplementaryFilter.hpp"
#include "Utils.hpp"
#include "csvreader.hpp"

#include <iomanip>
#include <iostream>

int main() {
    std::cout << std::fixed << std::setprecision(6);

    bool calculateBestAlpha = false;

    const double dt = 0.02;
    const double alpha = 0.79; // Calculated via loop - best value
    const int DISPLAY_SAMPLES = 10;

    // Load data
    CsvReader gyroData("Data/gyro.csv");
    gyroData.read();
    Eigen::MatrixXd gyroMeasurements = gyroData.getEigenData();

    CsvReader accelData("Data/accel.csv");
    accelData.read();
    Eigen::MatrixXd accelMeasurements = accelData.getEigenData();

    CsvReader anglesResults("Data/angles.csv");
    anglesResults.read();
    Eigen::MatrixXd groundTruthAngles = anglesResults.getEigenData();

    // Ground truth is in radian convert to degrees
    Eigen::VectorXd rollGroundTruth = Utils::getVectorFromMatrix(groundTruthAngles, 0);
    Eigen::VectorXd pitchGroundTruth = Utils::getVectorFromMatrix(groundTruthAngles, 1);
    Utils::convertToDeg(rollGroundTruth);
    Utils::convertToDeg(pitchGroundTruth);

    const int numSamples = static_cast<int>(gyroMeasurements.rows());
    ComplementaryFilter filter(alpha, dt);
    filter.setIMUData(gyroMeasurements, accelMeasurements);

    if(calculateBestAlpha) {
        double bestAlpha = 0.5;
        double bestRMSE = 100.00;

        // Loop from 0.01 to 0.99 with step 0.01
        for(int i = 1; i < 100; i++) {
            double currentAlpha = i * 0.01;
            ComplementaryFilter filterTmp(currentAlpha, dt);
            filterTmp.setIMUData(gyroMeasurements, accelMeasurements);

            filterTmp.calculateRoll();
            filterTmp.calculatePitch();

            auto rollTmp = filterTmp.getRoll();
            Utils::convertToDeg(rollTmp);

            double tmpBestRmse = Utils::rmse(rollGroundTruth, rollTmp);

            if (tmpBestRmse < bestRMSE) {
                bestRMSE = tmpBestRmse;
                bestAlpha = currentAlpha;
            }
        }

        std::cout << "Best Alpha: " << bestAlpha << " with best RMSE: " << bestRMSE << " degrees" << std::endl;

    } else {
        std::cout << "Complementary Filter Test\n\n";

        std::cout << "Dataset: " << numSamples << " samples\n";
        std::cout << "Time step (dt): " << dt << " seconds\n";
        std::cout << "Alpha coefficient: " << alpha << "\n\n";

        filter.calculateRoll();
        filter.calculatePitch();
    }

    // Convert results to degrees
    auto& roll = filter.getRoll(); // Results are in RAD
    Utils::convertToDeg(roll);     // Now results are in Degree

    auto& pitch = filter.getPitch(); // Results are in RAD
    Utils::convertToDeg(pitch);      // Now results are in Degree

    std::cout << "\n=== Error Metrics (all " << numSamples << " samples) in degrees ===\n";
    std::cout << "Roll RMSE:  " << Utils::rmse(rollGroundTruth, roll) << " degrees\n";
    std::cout << "Roll MEA:   " << Utils::mea(rollGroundTruth, roll) << " degrees\n";
    std::cout << "Pitch RMSE: " << Utils::rmse(pitchGroundTruth, pitch) << " degrees\n";
    std::cout << "Pitch MEA:  " << Utils::mea(pitchGroundTruth, pitch) << " degrees\n";

    Utils::printVecToFile(roll, "Results/Results/ComplementaryRoll_a_0_79.txt");
    Utils::printVecToFile(pitch, "Results/Results/ComplementaryPitch_a_0_79.txt");

    return 0;
}
