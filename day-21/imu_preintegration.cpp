/**
 * Author: Huseyn Kishiyev
 * ----------------------
 * IMU preintegration between two keyframes.
 * Accumulates Delta_R, Delta_v, Delta_p from raw IMU samples,
 * and propagates the associated covariance matrix.
 *
 * Based on the formulation in Forster et al. TRO 2017
 * "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry"
 *
 * This is a self-contained demo, it generates synthetic IMU data
 * (constant acceleration and angular rate) and preintegrates it,
 * then compares against the analytic ground truth.
 *
 * Build:
 *   g++ -O2 -std=c++17 imu_preintegration.cpp -o imu_preintegration
 *
 * Run:
 *   ./imu_preintegration
 *
 * No external dependencies beyond the C++ standard library.
 */

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <string>

// 3D vector and 3x3 matrix types 
using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;

Vec3 vec3(double x, double y, double z) { return {x, y, z}; }

Vec3 add(const Vec3& a, const Vec3& b) {
    return {a[0]+b[0], a[1]+b[1], a[2]+b[2]};
}

Vec3 scale(const Vec3& a, double s) {
    return {a[0]*s, a[1]*s, a[2]*s};
}

double dot(const Vec3& a, const Vec3& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

double norm(const Vec3& a) {
    return std::sqrt(dot(a, a));
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    };
}

// 3x3 identity
Mat3 eye3() {
    return {{{1,0,0},{0,1,0},{0,0,1}}};
}

// Skew-symmetric matrix of v
Mat3 skew(const Vec3& v) {
    return {{{ 0,   -v[2],  v[1]},
             { v[2],  0,   -v[0]},
             {-v[1],  v[0],  0  }}};
}

// Matrix addition
Mat3 madd(const Mat3& A, const Mat3& B) {
    Mat3 C;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

// Matrix-vector multiply
Vec3 mv(const Mat3& M, const Vec3& v) {
    return {
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2]
    };
}

// Matrix multiply
Mat3 mm(const Mat3& A, const Mat3& B) {
    Mat3 C = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// Scale matrix
Mat3 mscale(const Mat3& M, double s) {
    Mat3 R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R[i][j] = M[i][j] * s;
    return R;
}

// Transpose
Mat3 T(const Mat3& M) {
    Mat3 R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R[i][j] = M[j][i];
    return R;
}

// Rodrigues / Exponential map: rotation vector phi <-> rotation matrix R
// Exp(phi) = I + sin(|phi|)/|phi| * [phi]_x + (1-cos(|phi|))/|phi|^2 * [phi]_x^2
Mat3 exp_so3(const Vec3& phi) {
    double angle = norm(phi);
    if (angle < 1e-10) return eye3();

    Vec3 axis = scale(phi, 1.0 / angle);
    Mat3 K    = skew(axis);
    double s  = std::sin(angle);
    double c  = std::cos(angle);

    // Rodrigues: R = I + sin(a)*K + (1-cos(a))*K^2
    Mat3 K2   = mm(K, K);
    Mat3 R    = eye3();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R[i][j] += s * K[i][j] + (1.0 - c) * K2[i][j];
    return R;
}

// Logarithm map: rotation matrix -> rotation vector
Vec3 log_so3(const Mat3& R) {
    // trace = 1 + 2*cos(angle)
    double tr    = R[0][0] + R[1][1] + R[2][2];
    double cos_a = (tr - 1.0) * 0.5;
    cos_a        = std::clamp(cos_a, -1.0, 1.0);
    double angle = std::acos(cos_a);

    if (std::abs(angle) < 1e-10)
        return {0, 0, 0};

    double coeff = angle / (2.0 * std::sin(angle));
    return {
        coeff * (R[2][1] - R[1][2]),
        coeff * (R[0][2] - R[2][0]),
        coeff * (R[1][0] - R[0][1])
    };
}

// IMU preintegration state
struct PreintState {
    Mat3 Delta_R = eye3();   // preintegrated rotation
    Vec3 Delta_v = {0,0,0}; // preintegrated velocity change
    Vec3 Delta_p = {0,0,0}; // preintegrated position change
    double dt_total = 0.0;

    // Bias estimates (fixed during preintegration; corrected via Jacobians later)
    Vec3 b_g = {0, 0, 0};   // gyroscope bias
    Vec3 b_a = {0, 0, 0};   // accelerometer bias

    // Simple 9x9 covariance (blocks: [R, v, p] each 3x3 diagonal for clarity)
    // Here we track diagonal variances only to keep the code readable
    std::array<double, 9> diag_var = {};  // [var_R x3, var_v x3, var_p x3]
};

// Integrate one IMU sample
void integrate_imu(PreintState& state,
                   const Vec3& gyro_meas,
                   const Vec3& accel_meas,
                   double dt,
                   double gyro_noise_std,
                   double accel_noise_std) {
    // Bias-corrected measurements
    Vec3 omega = {gyro_meas[0]  - state.b_g[0],
                  gyro_meas[1]  - state.b_g[1],
                  gyro_meas[2]  - state.b_g[2]};

    Vec3 accel = {accel_meas[0] - state.b_a[0],
                  accel_meas[1] - state.b_a[1],
                  accel_meas[2] - state.b_a[2]};

    // Incremental rotation from gyro
    Vec3 dphi     = scale(omega, dt);   // small rotation vector
    Mat3 dR       = exp_so3(dphi);

    // Update Delta_p before updating Delta_v and Delta_R (midpoint rule)
    Vec3 accel_world = mv(state.Delta_R, accel);
    state.Delta_p = add(state.Delta_p,
                        add(scale(state.Delta_v, dt),
                            scale(accel_world, 0.5 * dt * dt)));
    state.Delta_v = add(state.Delta_v, scale(accel_world, dt));
    state.Delta_R = mm(state.Delta_R, dR);

    state.dt_total += dt;

    // Propagate diagonal variance (simplified: just accel and gyro noise)
    double gyro_var  = gyro_noise_std  * gyro_noise_std;
    double accel_var = accel_noise_std * accel_noise_std;

    // Rotation variance (rad^2 per axis): gyro noise integrated
    for (int i = 0; i < 3; ++i)
        state.diag_var[i] += gyro_var * dt * dt;

    // Velocity variance: accel noise integrated
    for (int i = 3; i < 6; ++i)
        state.diag_var[i] += accel_var * dt * dt;

    // Position variance: velocity variance propagated
    for (int i = 6; i < 9; ++i)
        state.diag_var[i] += state.diag_var[i-3] * dt * dt
                           + 0.25 * accel_var * dt * dt * dt * dt;
}

// Main: simulate preintegration and compare to ground truth
int main() {
    std::mt19937 rng(0);
    std::normal_distribution<double> noise(0.0, 1.0);

    // Sensor noise levels (realistic MEMS values)
    double gyro_noise_std  = 0.005;  // rad/s/sqrt(Hz)
    double accel_noise_std = 0.01;   // m/s^2/sqrt(Hz)

    // True motion: constant acceleration + yaw rotation
    Vec3 true_accel_body = {0.2, 0.0, 0.0};  // 0.2 m/s^2 forward in body frame
    Vec3 true_omega_body = {0.0, 0.0, 0.1};  // 0.1 rad/s yaw

    double dt       = 0.001;   // 1 kHz IMU
    double duration = 1.0;     // integrate 1 second of IMU data
    int    n_steps  = (int)(duration / dt);

    // Ground truth preintegrated values (analytic for constant motion)
    // Delta_R = Exp(omega * T)
    Vec3 gt_phi     = scale(true_omega_body, duration);
    Mat3 gt_Delta_R = exp_so3(gt_phi);

    // For constant body-frame accel, Delta_v and Delta_p require integration
    // of R(t)*a. For small rotation (10 deg in 1 sec), approximation:
    // Delta_v ~ a*T (if rotation is small), Delta_p ~ 0.5*a*T^2
    Vec3 gt_Delta_v = scale(true_accel_body, duration);
    Vec3 gt_Delta_p = scale(true_accel_body, 0.5 * duration * duration);

    // Run preintegration with noisy IMU data
    PreintState state;
    state.b_g = {0, 0, 0};
    state.b_a = {0, 0, 0};

    for (int k = 0; k < n_steps; ++k) {
        Vec3 gyro_meas = {
            true_omega_body[0] + gyro_noise_std  * noise(rng) / std::sqrt(dt),
            true_omega_body[1] + gyro_noise_std  * noise(rng) / std::sqrt(dt),
            true_omega_body[2] + gyro_noise_std  * noise(rng) / std::sqrt(dt)
        };
        Vec3 accel_meas = {
            true_accel_body[0] + accel_noise_std * noise(rng) / std::sqrt(dt),
            true_accel_body[1] + accel_noise_std * noise(rng) / std::sqrt(dt),
            true_accel_body[2] + accel_noise_std * noise(rng) / std::sqrt(dt)
        };
        integrate_imu(state, gyro_meas, accel_meas, dt,
                      gyro_noise_std, accel_noise_std);
    }

    // Compare preintegrated vs ground truth
    Vec3 err_phi = log_so3(mm(T(gt_Delta_R), state.Delta_R));
    Vec3 err_v   = {state.Delta_v[0] - gt_Delta_v[0],
                    state.Delta_v[1] - gt_Delta_v[1],
                    state.Delta_v[2] - gt_Delta_v[2]};
    Vec3 err_p   = {state.Delta_p[0] - gt_Delta_p[0],
                    state.Delta_p[1] - gt_Delta_p[1],
                    state.Delta_p[2] - gt_Delta_p[2]};

    auto fmt = [](const Vec3& v) {
        char buf[80];
        std::snprintf(buf, sizeof(buf),
                      "[%.6f, %.6f, %.6f]", v[0], v[1], v[2]);
        return std::string(buf);
    };

    std::cout << "IMU Preintegration Test\n";
    std::cout << "  Duration:     " << duration << " s | dt=" << dt*1000 << " ms"
              << " | " << n_steps << " samples\n";
    std::cout << "  Gyro noise:   " << gyro_noise_std  << " rad/s/sqrt(Hz)\n";
    std::cout << "  Accel noise:  " << accel_noise_std << " m/s^2/sqrt(Hz)\n";
    std::cout << "\nResults:\n";
    std::cout << "  Delta_R error (rad): " << fmt(err_phi)
              << "  norm=" << std::to_string(norm(err_phi)) << "\n";
    std::cout << "  Delta_v error (m/s): " << fmt(err_v)
              << "  norm=" << std::to_string(norm(err_v)) << "\n";
    std::cout << "  Delta_p error (m):   " << fmt(err_p)
              << "  norm=" << std::to_string(norm(err_p)) << "\n";
    std::cout << "\nDiagonal variance (sqrt = 1-sigma):\n";
    std::cout << "  R:  " << std::sqrt(state.diag_var[0]) << " rad\n";
    std::cout << "  v:  " << std::sqrt(state.diag_var[3]) << " m/s\n";
    std::cout << "  p:  " << std::sqrt(state.diag_var[6]) << " m\n";

    return 0;
}
