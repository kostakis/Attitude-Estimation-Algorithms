#include "ExplicitComplementaryFilter.hpp"
#include "Utils.hpp"

ExplicitComplementaryFilter::ExplicitComplementaryFilter(double dt, double kp, double ki)
    : dt(dt), kp(kp), ki(ki), rHat(Eigen::Matrix3d::Identity()), biasEstimate(Eigen::Vector3d::Zero()) {

    // Initialize inertial gravity vector: e3 = [0, 0, 1]ᵀ (pointing up in inertial frame)
    v0_gravity << 0.0, 0.0, 1.0;
}

void ExplicitComplementaryFilter::setIMUData(const Eigen::MatrixXd& gyroData,
                                              const Eigen::MatrixXd& accelData) {
    this->gyroData = gyroData;
    this->accelerometerData = accelData;

    if(gyroData.rows() != accelData.rows()) {
        std::cerr << "Error: Gyro and accelerometer data must have same number of samples\n";
        exit(-1);
    }

    int numSamples = static_cast<int>(gyroData.rows());
    rollEstimation.resize(numSamples);
    pitchEstimation.resize(numSamples);
    rollEstimation.setZero();
    pitchEstimation.setZero();
}

void ExplicitComplementaryFilter::predictForAllData() {
    for(int i = 0; i < gyroData.rows(); i++) {
        Eigen::Vector3d gyroReading = gyroData.row(i).transpose();
        Eigen::Vector3d accelReading = accelerometerData.row(i).transpose();

        // Sign correction (based on your sensor configuration)
        accelReading(0) = -accelReading(0);

        // Normalize accelerometer reading to get measured gravity direction
        Eigen::Vector3d v_measured = accelReading.normalized();

        // Run filter update
        update(gyroReading, v_measured);

        // Extract roll and pitch from rotation matrix R̂
        // For ZYX Euler convention:
        // roll = atan2(R̂₃₂, R̂₃₃)
        // pitch = atan2(-R̂₃₁, sqrt(R̂₃₂² + R̂₃₃²))
        double rollEstRad = Utils::calculateEulerRollFromSensor(rHat(2, 1), rHat(2, 2));
        double pitchEstRad = Utils::calculateEulerPitchFromInput(rHat(2, 0), rHat(2, 1), rHat(2, 2));

        rollEstimation(i) = rollEstRad;
        pitchEstimation(i) = pitchEstRad;
    }
}

void ExplicitComplementaryFilter::update(const Eigen::Vector3d& omega_y,
                                          const Eigen::Vector3d& v_measured) {
    // Compute estimated gravity direction in body frame: v̂ = R̂ᵀ v₀
    Eigen::Vector3d v_estimated = rHat.transpose() * v0_gravity;

    // Compute measurement-based correction term (Equation 32c)
    Eigen::Vector3d omega_mes = computeOmegaMeasurement(v_measured, v_estimated);

    // Update bias estimate (Equation 32b): ḃ̂ = -kᵢ ωₘₑₛ
    biasEstimate = biasEstimate - ki * omega_mes * dt;

    // Compute total angular velocity (Equation 32a):
    // ω_total = Ωʸ - b̂ + kₚ ωₘₑₛ
    Eigen::Vector3d omega_total = omega_y - biasEstimate + kp * omega_mes;

    // Update rotation matrix (Equation 32a): Ṙ̂ = R̂ [ω_total]ₓ
    // Discrete integration: R̂(k+1) = R̂(k) + R̂(k) [ω_total]ₓ Δt
    Eigen::Matrix3d omega_skew = Utils::skewMatrixFromVector(omega_total);
    rHat = rHat + rHat * omega_skew * dt;

    // Ensure R̂ stays on SO(3) manifold
    orthonormalize();
}

Eigen::Vector3d ExplicitComplementaryFilter::computeOmegaMeasurement(
    const Eigen::Vector3d& v_measured, const Eigen::Vector3d& v_estimated) const {

    // Equation (32c) from paper: ωₘₑₛ = Σᵢ kᵢ(vᵢ × v̂ᵢ)
    // For single vector (gravity only), this simplifies to:
    // ωₘₑₛ = v × v̂
    //
    // Note: Comparing with Mahony passive filter (Equation 7):
    // Mahony: ω = vex(Pa(R̃)) where R̃ = R̂ᵀRy
    // Explicit: ω = v × v̂ where v = Rᵀv₀, v̂ = R̂ᵀv₀
    // These should be equivalent up to the ki weighting in Eq 34

    return v_measured.cross(v_estimated);
}

void ExplicitComplementaryFilter::orthonormalize() {
    // Use SVD to project R̂ back onto SO(3)
    // Any 3×3 matrix can be decomposed as: R̂ = U Σ Vᵀ
    // The nearest rotation matrix is: R = U Vᵀ (set all singular values to 1)

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(rHat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    rHat = svd.matrixU() * svd.matrixV().transpose();

    // Ensure det(R̂) = +1 (proper rotation, not reflection)
    if(rHat.determinant() < 0) {
        Eigen::Matrix3d U = svd.matrixU();
        U.col(2) *= -1;  // Flip sign of last column
        rHat = U * svd.matrixV().transpose();
    }
}

// Getters
const Eigen::VectorXd& ExplicitComplementaryFilter::getRollEstimation() const {
    return rollEstimation;
}

Eigen::VectorXd& ExplicitComplementaryFilter::getRollEstimationNonConst() {
    return rollEstimation;
}

const Eigen::VectorXd& ExplicitComplementaryFilter::getPitchEstimation() const {
    return pitchEstimation;
}

Eigen::VectorXd& ExplicitComplementaryFilter::getPitchEstimationNonConst() {
    return pitchEstimation;
}

const Eigen::Vector3d& ExplicitComplementaryFilter::getBiasEstimation() const {
    return biasEstimate;
}
