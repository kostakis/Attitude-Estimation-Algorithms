#include "ExtendedKalmanFilter.hpp"

ExtendedKalmanFilter::ExtendedKalmanFilter(double dt, const Eigen::Vector3d& initial_accel) : dt(dt) {
    // Initialize state vector (7x1)
    x = Eigen::VectorXd::Zero(7);

    // Initialize quaternion from accelerometer (assumes static start)
    // Calculate roll and pitch from initial accelerometer
    double roll = std::atan2(initial_accel.y(), initial_accel.z());
    double pitch = std::atan2(-initial_accel.x(),
                              std::sqrt(initial_accel.y() * initial_accel.y() + initial_accel.z() * initial_accel.z()));

    // Convert roll, pitch to quaternion (yaw = 0)
    double cy = std::cos(0.0 * 0.5);
    double sy = std::sin(0.0 * 0.5);
    double cp = std::cos(pitch * 0.5);
    double sp = std::sin(pitch * 0.5);
    double cr = std::cos(roll * 0.5);
    double sr = std::sin(roll * 0.5);

    x(0) = cr * cp * cy + sr * sp * sy; // q0 (w)
    x(1) = sr * cp * cy - cr * sp * sy; // q1 (x)
    x(2) = cr * sp * cy + sr * cp * sy; // q2 (y)
    x(3) = cr * cp * sy - sr * sp * cy; // q3 (z)

    // Initialize covariance matrix (7x7)
    P = Eigen::MatrixXd::Identity(7, 7);
    P.block<4, 4>(0, 0) *= 0.1;  // Quaternion uncertainty
    P.block<3, 3>(4, 4) *= 0.01; // Bias uncertainty

    // Process noise covariance (7x7)
    // TODO check these values
    Q = Eigen::MatrixXd::Identity(7, 7);
    Q.block<4, 4>(0, 0) *= 0.001;  // Quaternion process noise
    Q.block<3, 3>(4, 4) *= 0.0001; // Bias process noise

    // Measurement noise covariance (3x3) - accelerometer
    R = Eigen::MatrixXd::Identity(3, 3) * 0.1;
}

void ExtendedKalmanFilter::predict(const Eigen::Vector3d& gyro) {
    // Extract current state
    Eigen::Vector4d q = x.head(4);
    Eigen::Vector3d bias = x.tail(3);

    // Correct gyro measurement with bias
    Eigen::Vector3d w = gyro - bias;

    // Quaternion derivative matrix (big omega)
    Eigen::Matrix4d Omega;
    Omega << 0, -w(0), -w(1), -w(2), w(0), 0, w(2), -w(1), w(1), -w(2), 0, w(0), w(2), w(1), -w(0), 0;

    // Update quaternion using first-order integration
    Eigen::Vector4d q_new = q + 0.5 * dt * Omega * q;

    // Update state
    x.head<4>() = q_new;
    normalizeQuaternion();
    // Bias remains constant (no process model for bias)

    // Compute state transition Jacobian F
    Eigen::MatrixXd F = computeF(w);

    // Update covariance: P = F * P * F^T + Q
    P = F * P * F.transpose() + Q;
}

void ExtendedKalmanFilter::update(const Eigen::Vector3d& accel) {
    // Expected measurement (gravity in body frame)
    Eigen::Vector4d q = x.head<4>();
    Eigen::Matrix3d R_mat = quaternionToRotationMatrix(q);

    // Gravity in navigation frame
    Eigen::Vector3d g_n(0, 0, 1); // Normalized gravity vector

    // Expected measurement: rotate gravity to body frame
    Eigen::Vector3d h = R_mat.transpose() * g_n;

    // Normalize accelerometer measurement
    Eigen::Vector3d accel_norm = accel.normalized();

    // Innovation (measurement residual)
    Eigen::Vector3d y = accel_norm - h;

    // Compute measurement Jacobian H
    Eigen::MatrixXd H = computeH();

    // Innovation covariance S = H * P * H^T + R
    Eigen::Matrix3d S = H * P * H.transpose() + R;

    // Kalman gain K = P * H^T * S^-1
    Eigen::MatrixXd K = P * H.transpose() * S.inverse();

    // Update state: x = x + K * y
    x = x + K * y;
    normalizeQuaternion();

    // Update covariance: P = (I - K * H) * P
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(7, 7);
    P = (I - K * H) * P;
}

void ExtendedKalmanFilter::setIMUData(const Eigen::MatrixXd& gyroData, const Eigen::MatrixXd& accelData) {
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
}

void ExtendedKalmanFilter::predictForAllData() {
    for(int i = 0; i < gyroData.rows(); i++) {
        Eigen::Vector3d gyroReading = gyroData.row(i).transpose();
        Eigen::Vector3d accelReading = accelometerData.row(i).transpose();
        accelReading(0) = -accelReading(0); // Sign correction

        // Prediction step
        predict(gyroReading);

        // Update step
        update(accelReading);

        // Extract roll and pitch from current quaternion state
        rollEstimation(i) = getRollFromQuaternion();
        pitchEstimation(i) = getPitchFromQuaternion();
    }
}

const Eigen::VectorXd& ExtendedKalmanFilter::getRollEstimation() const {
    return rollEstimation;
}

Eigen::VectorXd& ExtendedKalmanFilter::getRollEstimationNonConst() {
    return rollEstimation;
}

const Eigen::VectorXd& ExtendedKalmanFilter::getPitchEstimation() const {
    return pitchEstimation;
}

Eigen::VectorXd& ExtendedKalmanFilter::getPitchEstimationNonConst() {
    return pitchEstimation;
}

double ExtendedKalmanFilter::getRollFromQuaternion() const {
    Eigen::Vector4d q = x.head<4>();
    double q0 = q(0), q1 = q(1), q2 = q(2), q3 = q(3);

    return std::atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2));
}

double ExtendedKalmanFilter::getPitchFromQuaternion() const {
    Eigen::Vector4d q = x.head<4>();
    double q0 = q(0), q1 = q(1), q2 = q(2), q3 = q(3);

    double sinp = 2 * (q0 * q2 - q3 * q1);

    // Handle gimbal lock
    if(std::abs(sinp) >= 1) {
        return std::copysign(M_PI / 2, sinp);
    } else {
        return std::asin(sinp);
    }
}

void ExtendedKalmanFilter::normalizeQuaternion() {
    Eigen::Vector4d q = x.head<4>();
    x.head<4>() = q / q.norm();
}

Eigen::Matrix3d ExtendedKalmanFilter::quaternionToRotationMatrix(const Eigen::Vector4d& q) const {
    double q0 = q(0), q1 = q(1), q2 = q(2), q3 = q(3);

    Eigen::Matrix3d R;
    R << q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2),
        2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1),
        2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

    return R;
}

Eigen::MatrixXd ExtendedKalmanFilter::computeF(const Eigen::Vector3d& w) const {
    // State transition Jacobian (7x7)
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(7, 7);

    // Extract quaternion
    Eigen::Vector4d q = x.head<4>();

    // Quaternion derivative matrix
    Eigen::Matrix4d Omega;
    Omega << 0, -w(0), -w(1), -w(2), w(0), 0, w(2), -w(1), w(1), -w(2), 0, w(0), w(2), w(1), -w(0), 0;

    // F_qq (4x4) - derivative of quaternion w.r.t quaternion
    F.block<4, 4>(0, 0) = Eigen::Matrix4d::Identity() + 0.5 * dt * Omega;

    // F_qb (4x3) - derivative of quaternion w.r.t bias
    Eigen::Matrix<double, 4, 3> F_qb;
    F_qb.row(0) = 0.5 * dt * Eigen::Vector3d(-q(1), -q(2), -q(3));
    F_qb.row(1) = 0.5 * dt * Eigen::Vector3d(q(0), -q(3), q(2));
    F_qb.row(2) = 0.5 * dt * Eigen::Vector3d(q(3), q(0), -q(1));
    F_qb.row(3) = 0.5 * dt * Eigen::Vector3d(-q(2), q(1), q(0));

    F.block<4, 3>(0, 4) = F_qb;

    // F_bb (3x3) - bias w.r.t bias is identity (already set)

    return F;
}

Eigen::MatrixXd ExtendedKalmanFilter::computeH() const {
    // Measurement Jacobian (3x7)
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 7);

    // Extract quaternion
    Eigen::Vector4d q = x.head<4>();
    double q0 = q(0), q1 = q(1), q2 = q(2), q3 = q(3);

    // Gravity vector in navigation frame
    Eigen::Vector3d g_n(0, 0, 1);

    // H_q (3x4) - derivative of measurement w.r.t quaternion
    // h = R^T * g_n, where R is rotation matrix from quaternion
    // This is the derivative of the body-frame gravity measurement w.r.t quaternion

    H(0, 0) = 2 * (q0 * g_n(0) + q3 * g_n(1) - q2 * g_n(2));
    H(0, 1) = 2 * (q1 * g_n(0) + q2 * g_n(1) + q3 * g_n(2));
    H(0, 2) = 2 * (-q2 * g_n(0) + q1 * g_n(1) - q0 * g_n(2));
    H(0, 3) = 2 * (-q3 * g_n(0) + q0 * g_n(1) + q1 * g_n(2));

    H(1, 0) = 2 * (-q3 * g_n(0) + q0 * g_n(1) + q1 * g_n(2));
    H(1, 1) = 2 * (q2 * g_n(0) - q1 * g_n(1) + q0 * g_n(2));
    H(1, 2) = 2 * (q1 * g_n(0) + q2 * g_n(1) + q3 * g_n(2));
    H(1, 3) = 2 * (-q0 * g_n(0) - q3 * g_n(1) + q2 * g_n(2));

    H(2, 0) = 2 * (q2 * g_n(0) - q1 * g_n(1) + q0 * g_n(2));
    H(2, 1) = 2 * (q3 * g_n(0) - q0 * g_n(1) - q1 * g_n(2));
    H(2, 2) = 2 * (q0 * g_n(0) + q3 * g_n(1) - q2 * g_n(2));
    H(2, 3) = 2 * (q1 * g_n(0) + q2 * g_n(1) + q3 * g_n(2));

    // H_b (3x3) - derivative w.r.t bias is zero (measurement doesn't depend on bias)
    // Already initialized to zero

    return H;
}
