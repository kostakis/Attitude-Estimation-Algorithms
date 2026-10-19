#include "ExplicitComplementaryFilter.hpp"
#include "Utils.hpp"
#include "csvreader.hpp"

#include <iomanip>
#include <iostream>

int main() {
    bool calculateBestKs = false;

    std::cout << std::fixed << std::setprecision(6);

    const double dt = 0.02;
    const double kp = 11.0;  // Proportional gain - best value
    const double ki = 0.05;  // Integral gain - best value

    std::cout << "Explicit Complementary Filter Test\n";
    std::cout << "===================================\n\n";

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
    ExplicitComplementaryFilter explicitCF(dt, kp, ki);

    if(calculateBestKs) {
        double bestKp = 1.0;
        double bestKi = 0.1;

        double bestCombinedRMSE = 100.00;
        double bestRollRMSE = 0.0;
        double bestPitchRMSE = 0.0;

        std::cout << "Searching for best kp and ki values...\n";
        std::cout << "Testing kp from 1.0 to 15.0 (step 0.5)\n";
        std::cout << "Testing ki from 0.05 to 1.0 (step 0.05)\n\n";

        int totalTests = 0;
        int printCounter = 0;

        // Grid search over kp and ki with finer granularity
        // kp: 1.0 to 15.0 in steps of 0.5 (29 values)
        // ki: 0.05 to 1.0 in steps of 0.05 (20 values)
        // Total: 29 * 20 = 580 combinations
        for(int kp_times_2 = 2; kp_times_2 <= 30; kp_times_2++) {
            double kp_test = kp_times_2 * 0.5;

            for(int ki_times_20 = 1; ki_times_20 <= 20; ki_times_20++) {
                double ki_test = ki_times_20 * 0.05;

                ExplicitComplementaryFilter explicitTmp(dt, kp_test, ki_test);
                explicitTmp.setIMUData(gyroMeasurements, accelMeasurements);
                explicitTmp.predictForAllData();

                // Convert to degrees for RMSE calculation
                Utils::convertToDeg(explicitTmp.getRollEstimationNonConst());
                Utils::convertToDeg(explicitTmp.getPitchEstimationNonConst());

                double rollRMSE = Utils::rmse(rollGroundTruth, explicitTmp.getRollEstimation());
                double pitchRMSE = Utils::rmse(pitchGroundTruth, explicitTmp.getPitchEstimation());

                // Combined metric: average of roll and pitch RMSE
                double combinedRMSE = (rollRMSE + pitchRMSE) / 2.0;

                totalTests++;

                if (combinedRMSE < bestCombinedRMSE) {
                    bestCombinedRMSE = combinedRMSE;
                    bestRollRMSE = rollRMSE;
                    bestPitchRMSE = pitchRMSE;
                    bestKp = kp_test;
                    bestKi = ki_test;

                    // Print when we find a new best
                    std::cout << "NEW BEST! kp=" << kp_test << ", ki=" << ki_test
                              << " -> Roll: " << rollRMSE
                              << "°, Pitch: " << pitchRMSE
                              << "°, Combined: " << combinedRMSE << "°\n";
                }

                // Print progress every 50 tests
                printCounter++;
                if (printCounter % 50 == 0) {
                    std::cout << "Progress: " << totalTests << " tests completed...\n";
                }
            }
        }

        std::cout << "\nTotal tests: " << totalTests << "\n";

        std::cout << "\n=== BEST RESULTS ===\n";
        std::cout << "Best kp: " << bestKp << "\n";
        std::cout << "Best ki: " << bestKi << "\n";
        std::cout << "Roll RMSE:  " << bestRollRMSE << " degrees\n";
        std::cout << "Pitch RMSE: " << bestPitchRMSE << " degrees\n";
        std::cout << "Combined RMSE: " << bestCombinedRMSE << " degrees (average)\n\n";

        return 1;
    } else {
        // Display configuration
        std::cout << "Dataset: " << numSamples << " samples\n";
        std::cout << "Time step (dt): " << dt << " seconds\n";
        std::cout << "Proportional gain (kp): " << kp << " rad/s\n";
        std::cout << "Integral gain (ki): " << ki << " rad/s\n\n";

        // Create and run filter
        explicitCF.setIMUData(gyroMeasurements, accelMeasurements);
        explicitCF.predictForAllData();
    }

    // Convert results to degrees 
    Utils::convertToDeg(explicitCF.getRollEstimationNonConst());
    Utils::convertToDeg(explicitCF.getPitchEstimationNonConst());

    std::cout << "\n=== Error Metrics (all " << numSamples << " samples) in degrees ===\n";
    std::cout << "Roll RMSE:  " << Utils::rmse(rollGroundTruth, explicitCF.getRollEstimation()) << " degrees\n";
    std::cout << "Roll MEA:   " << Utils::mea(rollGroundTruth, explicitCF.getRollEstimation()) << " degrees\n";
    std::cout << "Pitch RMSE: " << Utils::rmse(pitchGroundTruth, explicitCF.getPitchEstimation()) << " degrees\n";
    std::cout << "Pitch MEA:  " << Utils::mea(pitchGroundTruth, explicitCF.getPitchEstimation()) << " degrees\n";

    Utils::printVecToFile(explicitCF.getRollEstimation(), "Results/Results/ExplicitComplementaryRoll.txt");
    Utils::printVecToFile(explicitCF.getPitchEstimation(), "Results/Results/ExplicitComplementaryPitch.txt");

    return 0;
}
