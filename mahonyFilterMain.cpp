#include "MahonyFilter.hpp"
#include "Utils.hpp"
#include "csvreader.hpp"

#include <iomanip>
#include <iostream>
#include <string>

int main() {
    std::cout << std::fixed << std::setprecision(6);

    bool calculateBestK = false;

    const double dt = 0.02;
    const double kp = 11; // Was calcualted via searching loop using RMSE for both roll and pitch best value is 11
    const int DISPLAY_SAMPLES = 10;

    // Load data
    CsvReader gyroData("Data/gyro.csv");
    gyroData.read();
    Eigen::MatrixXd gyroMeasurements = gyroData.getEigenData();

    CsvReader accelData("Data/accel.csv");
    accelData.read();
    Eigen::MatrixXd accelMeasurements = accelData.getEigenData();

    CsvReader anglesData("Data/angles.csv");
    anglesData.read();
    Eigen::MatrixXd groundTruthAngles = anglesData.getEigenData();

    // Ground truth (convert from radians to degrees)
    Eigen::VectorXd rollGroundTruth = Utils::getVectorFromMatrix(groundTruthAngles, 0);
    Eigen::VectorXd pitchGroundTruth = Utils::getVectorFromMatrix(groundTruthAngles, 1);
    Utils::convertToDeg(rollGroundTruth);
    Utils::convertToDeg(pitchGroundTruth);

    const int numSamples = static_cast<int>(gyroMeasurements.rows());
    MahonyFilter mahony(dt, kp);
    mahony.setIMUData(gyroMeasurements, accelMeasurements);

    if(calculateBestK) {
        double bestK=1;
        double bestCombinedRMSE = 100.00;
        double bestRollRMSE = 0.0;
        double bestPitchRMSE = 0.0;

        std::cout << "Searching for best kp \n";

        for(int i=1; i < 101; i++) {
            MahonyFilter mahonyTmp(dt, i);
            mahonyTmp.setIMUData(gyroMeasurements, accelMeasurements);

            mahonyTmp.predictForAllData();

            // Convert to degrees for RMSE calculation
            Utils::convertToDeg(mahonyTmp.getRollEstimationNonConst());
            Utils::convertToDeg(mahonyTmp.getPitchEstimationNonConst());

            double rollRMSE = Utils::rmse(rollGroundTruth, mahonyTmp.getRollEstimation());
            double pitchRMSE = Utils::rmse(pitchGroundTruth, mahonyTmp.getPitchEstimation());

            // Combined metric: average of roll and pitch RMSE
            double combinedRMSE = (rollRMSE + pitchRMSE) / 2.0;

            if (combinedRMSE < bestCombinedRMSE) {
                bestCombinedRMSE = combinedRMSE;
                bestRollRMSE = rollRMSE;
                bestPitchRMSE = pitchRMSE;
                bestK = i;
            }

            // Print progress every 10 iterations
            if (i % 10 == 0) {
                std::cout << "kp=" << i << " -> Roll RMSE: " << rollRMSE
                          << "°, Pitch RMSE: " << pitchRMSE
                          << "°, Combined: " << combinedRMSE << "°\n";
            }
        }

        std::cout << "\n=== BEST RESULTS ===\n";
        std::cout << "Best kp: " << bestK << "\n";
        std::cout << "Roll RMSE:  " << bestRollRMSE << " degrees\n";
        std::cout << "Pitch RMSE: " << bestPitchRMSE << " degrees\n";
        std::cout << "Combined RMSE: " << bestCombinedRMSE << " degrees (average)\n";

    } else {
        std::cout << "Mahony Passive Complementary Filter Test\n\n";

        std::cout << "Dataset: " << numSamples << " samples\n";
        std::cout << "Time step (dt): " << dt << " seconds\n";
        std::cout << "Proportional gain (kP): " << kp << "\n\n";

        mahony.predictForAllData();
    }

    // Conver results to degrees
    Utils::convertToDeg(mahony.getRollEstimationNonConst());
    Utils::convertToDeg(mahony.getPitchEstimationNonConst());


    std::cout << "\n=== Error Metrics (all " << numSamples << " samples) in degrees ===\n";
    std::cout << "Roll RMSE:  " << Utils::rmse(rollGroundTruth, mahony.getRollEstimation()) << " degrees\n";
    std::cout << "Roll MEA:   " << Utils::mea(rollGroundTruth, mahony.getRollEstimation()) << " degrees\n";
    std::cout << "Pitch RMSE: " << Utils::rmse(pitchGroundTruth, mahony.getPitchEstimation()) << " degrees\n";
    std::cout << "Pitch MEA:  " << Utils::mea(pitchGroundTruth, mahony.getPitchEstimation()) << " degrees\n";

    Utils::printVecToFile(mahony.getRollEstimation(), "Results/Results/MahonyRoll_kp_" + std::to_string(static_cast<int>(kp)) + ".txt");
    Utils::printVecToFile(mahony.getPitchEstimation(), "Results/Results/MahonyPitch_kp_" + std::to_string(static_cast<int>(kp)) + ".txt");

    return 0;
}
