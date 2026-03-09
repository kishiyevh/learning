/**
 * Author: Huseyn Kishiyev
 * Sources: https://docs.px4.io/main/en/ros/external_position_estimation was
 * really helpful.
 * --------------
 * 3D EKF fusing IMU prediction with visual odometry position updates.
 * State: [position (3), velocity (3), attitude quaternion (4)]  = 10 DOF
 * Error state: [dp (3), dv (3), dtheta (3)] = 9 DOF (no quaternion overparameterization)
 *
 * IMU drives the prediction at 200 Hz (simulated).
 * VO provides relative position updates at 10 Hz (simulated).
 *
 * Outputs CSV: ekf_trace.csv
 * Columns: t, x_true, y_true, z_true, x_est, y_est, z_est, pos_err
 *
 * Build:
 *   g++ -O2 -std=c++17 ekf_vo_imu.cpp -o ekf_vo_imu
 *
 * Run:
 *   ./ekf_vo_imu
 *
 * No external dependencies.
 */

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

// Linear algebra
static constexpr int N = 9;  // error-state dimension

using Vec9  = std::array<double, N>;
using Mat9  = std::array<std::array<double, N>, N>;
using Vec3  = std::array<double, 3>;
using Mat3  = std::array<std::array<double, 3>, 3>;
using Mat39 = std::array<std::array<double, N>, 3>;   // 3 x 9
using Mat93 = std::array<std::array<double, 3>, N>;   // 9 x 3

Mat9 eye9() {
    Mat9 M = {};
    for (int i = 0; i < N; ++i) M[i][i] = 1.0;
    return M;
}

Vec9 add9(const Vec9& a, const Vec9& b) {
    Vec9 c;
    for (int i = 0; i < N; ++i) c[i] = a[i] + b[i];
    return c;
}

Mat9 add9m(const Mat9& A, const Mat9& B) {
    Mat9 C;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Mat9 mm9(const Mat9& A, const Mat9& B) {
    Mat9 C = {};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

Vec9 mv9(const Mat9& M, const Vec9& v) {
    Vec9 r = {};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            r[i] += M[i][j] * v[j];
    return r;
}

Mat9 T9(const Mat9& M) {
    Mat9 R;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            R[i][j] = M[j][i];
    return R;
}

// Quaternion for [w, x, y, z]
using Quat = std::array<double, 4>;

Quat quat_identity() { return {1, 0, 0, 0}; }

// Quaternion multiplication
Quat qmul(const Quat& a, const Quat& b) {
    return {
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    };
}

void qnorm(Quat& q) {
    double n = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    for (auto& x : q) x /= n;
}

// Small rotation vector <-> delta quaternion
Quat delta_quat(const Vec3& dphi) {
    double angle = std::sqrt(dphi[0]*dphi[0] + dphi[1]*dphi[1] + dphi[2]*dphi[2]);
    if (angle < 1e-10) return {1, 0.5*dphi[0], 0.5*dphi[1], 0.5*dphi[2]};
    double s = std::sin(angle*0.5) / angle;
    return {std::cos(angle*0.5), s*dphi[0], s*dphi[1], s*dphi[2]};
}

// Rotating vector by quaternion, v' = q * [0,v] * q_conj
Vec3 qrot(const Quat& q, const Vec3& v) {
    // Using the formula: v' = v + 2w*(q_vec x v) + 2*(q_vec x (q_vec x v))
    double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    double tx = 2*(qy*v[2] - qz*v[1]);
    double ty = 2*(qz*v[0] - qx*v[2]);
    double tz = 2*(qx*v[1] - qy*v[0]);
    return {
        v[0] + qw*tx + qy*tz - qz*ty,
        v[1] + qw*ty + qz*tx - qx*tz,
        v[2] + qw*tz + qx*ty - qy*tx
    };
}

double vnorm3(const Vec3& v) {
    return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
}

// Error-state EKF state
struct EKFState {
    Vec3  pos = {0, 0, 0};
    Vec3  vel = {0, 0, 0};
    Quat  att = quat_identity();

    // Error-state covariance (9x9: dp, dv, dtheta)
    Mat9  P   = {};

    EKFState() {
        // Initial uncertainty
        for (int i = 0;   i < 3; ++i) P[i][i]   = 1.0;   // position (m^2)
        for (int i = 3;   i < 6; ++i) P[i][i]   = 0.1;   // velocity (m/s)^2
        for (int i = 6;   i < 9; ++i) P[i][i]   = 0.01;  // attitude (rad^2)
    }
};

// Predict: propagate with IMU measurement
static const Vec3 GRAVITY = {0, 0, 9.81};  // NED: gravity points down (+z)

void predict(EKFState& s,
             const Vec3& accel_meas,   // body frame
             const Vec3& gyro_meas,    // body frame
             double dt,
             double accel_noise_std,
             double gyro_noise_std) {

    // Rotate accel from body to world
    Vec3 a_world = qrot(s.att, accel_meas);
    // Add gravity (NED: subtract because gravity adds to +z measurement)
    // Here I use ENU convention (z up), so gravity = -z
    Vec3 a_net = {a_world[0], a_world[1], a_world[2] - 9.81};

    // Nominal state update
    for (int i = 0; i < 3; ++i) {
        s.pos[i] += s.vel[i] * dt + 0.5 * a_net[i] * dt * dt;
        s.vel[i] += a_net[i] * dt;
    }

    Vec3 dphi = {gyro_meas[0]*dt, gyro_meas[1]*dt, gyro_meas[2]*dt};
    Quat dq   = delta_quat(dphi);
    s.att     = qmul(s.att, dq);
    qnorm(s.att);

    // Simplified F matrix (linearized error-state transition)
    // F = [ I  I*dt  0   ]
    //     [ 0  I     A*dt]  where A = -R * [a_body]_x
    //     [ 0  0     I   ]
    Mat9 F = eye9();
    // dp += dv * dt
    F[0][3] = dt; F[1][4] = dt; F[2][5] = dt;

    // Process noise: Q = diag(accel_var * dt^2 for vel, gyro_var * dt^2 for att)
    double avar2 = accel_noise_std * accel_noise_std * dt * dt;
    double gvar2 = gyro_noise_std  * gyro_noise_std  * dt * dt;
    // Also propagate position noise from velocity noise
    double pvar2 = avar2 * dt * dt;

    Mat9 Q = {};
    Q[0][0] = Q[1][1] = Q[2][2] = pvar2;
    Q[3][3] = Q[4][4] = Q[5][5] = avar2;
    Q[6][6] = Q[7][7] = Q[8][8] = gvar2;

    s.P = add9m(mm9(mm9(F, s.P), T9(F)), Q);
}

// Update: VO gives a noisy position measurement
void update_position(EKFState& s, const Vec3& z_pos, double pos_noise_std) {
    // H = [I3 | 0 | 0]  (position is directly observed)
    // Innovation y = z - H*x_nominal
    Vec3 innov = {z_pos[0] - s.pos[0],
                  z_pos[1] - s.pos[1],
                  z_pos[2] - s.pos[2]};

    // S = H P H^T + R (3x3)
    double R_noise = pos_noise_std * pos_noise_std;
    // H P H^T is just the top-left 3x3 block of P
    std::array<std::array<double,3>,3> S = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            S[i][j] = s.P[i][j];
    for (int i = 0; i < 3; ++i) S[i][i] += R_noise;

    // Invert S (3x3, assume diagonal dominant, use exact 3x3 inverse)
    // Compute det
    double det = S[0][0]*(S[1][1]*S[2][2]-S[1][2]*S[2][1])
               - S[0][1]*(S[1][0]*S[2][2]-S[1][2]*S[2][0])
               + S[0][2]*(S[1][0]*S[2][1]-S[1][1]*S[2][0]);
    if (std::abs(det) < 1e-12) return;  // singular — skip update

    std::array<std::array<double,3>,3> Sinv = {};
    Sinv[0][0] =  (S[1][1]*S[2][2]-S[1][2]*S[2][1]) / det;
    Sinv[0][1] = -(S[0][1]*S[2][2]-S[0][2]*S[2][1]) / det;
    Sinv[0][2] =  (S[0][1]*S[1][2]-S[0][2]*S[1][1]) / det;
    Sinv[1][0] = -(S[1][0]*S[2][2]-S[1][2]*S[2][0]) / det;
    Sinv[1][1] =  (S[0][0]*S[2][2]-S[0][2]*S[2][0]) / det;
    Sinv[1][2] = -(S[0][0]*S[1][2]-S[0][2]*S[1][0]) / det;
    Sinv[2][0] =  (S[1][0]*S[2][1]-S[1][1]*S[2][0]) / det;
    Sinv[2][1] = -(S[0][0]*S[2][1]-S[0][1]*S[2][0]) / det;
    Sinv[2][2] =  (S[0][0]*S[1][1]-S[0][1]*S[1][0]) / det;

    // K = P H^T S^{-1}  (9x3 * 3x3 = 9x3, where H^T picks columns 0-2 of P)
    std::array<std::array<double,3>,N> K = {};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                K[i][j] += s.P[i][k] * Sinv[k][j];

    // Correction dx = K * innov
    Vec9 dx = {};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < 3; ++j)
            dx[i] += K[i][j] * innov[j];

    // Apply corrections
    s.pos[0] += dx[0]; s.pos[1] += dx[1]; s.pos[2] += dx[2];
    s.vel[0] += dx[3]; s.vel[1] += dx[4]; s.vel[2] += dx[5];

    Vec3 dphi = {dx[6], dx[7], dx[8]};
    Quat dq   = delta_quat(dphi);
    s.att     = qmul(s.att, dq);
    qnorm(s.att);

    // P = (I - KH) P
    Mat9 KH = {};
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < 3; ++j)
            KH[i][j] += K[i][j];  // H is identity on position rows 0-2

    Mat9 IminusKH = eye9();
    for (int i = 0; i < N; ++i)
        IminusKH[i][i] -= KH[i][i < 3 ? i : 0] * (i < 3 ? 1.0 : 0.0);

    s.P = mm9(IminusKH, s.P);
}

// Main
int main() {
    std::mt19937 rng(7);
    auto randn = [&](double std) {
        return std::normal_distribution<double>(0, std)(rng);
    };

    // Simulation parameters
    const double DT_IMU    = 0.005;   // 200 Hz IMU
    const double DT_VO     = 0.1;     // 10 Hz VO
    const double DURATION  = 30.0;
    const int    N_STEPS   = (int)(DURATION / DT_IMU);
    const int    VO_EVERY  = (int)(DT_VO / DT_IMU);

    const double ACCEL_NOISE = 0.05;  // m/s^2/sqrt(Hz)
    const double GYRO_NOISE  = 0.003; // rad/s/sqrt(Hz)
    const double VO_NOISE    = 0.15;  // m position noise

    EKFState state;
    Vec3     x_true = {0, 0, 0};
    Vec3     v_true = {0, 0, 0};

    std::ofstream csv("ekf_trace.csv");
    csv << "t,x_true,y_true,z_true,x_est,y_est,z_est,pos_err,pos_std\n";

    for (int k = 0; k < N_STEPS; ++k) {
        double t = k * DT_IMU;

        // Smooth circular trajectory in XY, constant Z
        double speed  = 1.0;  // m/s
        double radius = 5.0;  // m
        double omega  = speed / radius;

        // True acceleration (centripetal)
        Vec3 a_true_world = {
            -omega * omega * x_true[0],
            -omega * omega * x_true[1],
            0.0
        };
        Vec3 gyro_true = {0, 0, omega};

        // True body accel (in world frame for simplicity — ignoring attitude body transform)
        Vec3 accel_meas = {
            a_true_world[0] + randn(ACCEL_NOISE / std::sqrt(DT_IMU)),
            a_true_world[1] + randn(ACCEL_NOISE / std::sqrt(DT_IMU)),
            randn(ACCEL_NOISE / std::sqrt(DT_IMU))
        };
        Vec3 gyro_meas = {
            gyro_true[0] + randn(GYRO_NOISE / std::sqrt(DT_IMU)),
            gyro_true[1] + randn(GYRO_NOISE / std::sqrt(DT_IMU)),
            gyro_true[2] + randn(GYRO_NOISE / std::sqrt(DT_IMU))
        };

        // True state update
        for (int i = 0; i < 3; ++i) {
            x_true[i] += v_true[i] * DT_IMU + 0.5 * a_true_world[i] * DT_IMU * DT_IMU;
            v_true[i] += a_true_world[i] * DT_IMU;
        }
        // Circular init
        if (k == 0) { x_true = {radius, 0, 0}; v_true = {0, speed, 0}; }

        predict(state, accel_meas, gyro_meas, DT_IMU, ACCEL_NOISE, GYRO_NOISE);

        // VO update at 10 Hz
        if (k % VO_EVERY == 0) {
            Vec3 z_vo = {
                x_true[0] + randn(VO_NOISE),
                x_true[1] + randn(VO_NOISE),
                x_true[2] + randn(VO_NOISE)
            };
            update_position(state, z_vo, VO_NOISE);
        }

        // Log every 10 IMU steps
        if (k % 10 == 0) {
            double pos_err = vnorm3({state.pos[0]-x_true[0],
                                     state.pos[1]-x_true[1],
                                     state.pos[2]-x_true[2]});
            double pos_std = std::sqrt(state.P[0][0] + state.P[1][1] + state.P[2][2]);
            csv << std::fixed << std::setprecision(4)
                << t << ","
                << x_true[0] << "," << x_true[1] << "," << x_true[2] << ","
                << state.pos[0] << "," << state.pos[1] << "," << state.pos[2] << ","
                << pos_err << "," << pos_std << "\n";

            if (k % (VO_EVERY * 10) == 0)
                std::printf("t=%5.1f s | pos_err=%.3f m | pos_std=%.3f m\n",
                            t, pos_err, pos_std);
        }
    }
    csv.close();

    std::cout << "\nOutput: ekf_trace.csv\n";

    return 0;
}
