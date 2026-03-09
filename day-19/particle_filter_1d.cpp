/**
 * Author: Huseyn Kishiyev
 * ----------------------
 * 1D particle filter simulating a robot moving along a corridor.
 * The corridor has discrete landmarks (door positions). The robot
 * gets noisy range measurements to nearby landmarks and uses them
 * to localize from a uniform (global) initialization.
 *
 * Demonstrates;
 *   - Uniform initialization across corridor (global localization)
 *   - Motion model with additive Gaussian noise
 *   - Measurement likelihood from landmark range sensor
 *   - Systematic resampling
 *   - Convergence tracking via effective sample size
 *
 * Build:
 *   g++ -O2 -std=c++17 -o particle_filter_1d particle_filter_1d.cpp
 *
 * Run:
 *   ./particle_filter_1d
 *
 * Output: CSV file "pf_trace.csv" with columns:
 *   step, x_true, x_est, x_std, eff_n, landmark_detected
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Simulation parameters 
static constexpr double CORRIDOR_LENGTH = 20.0;   // meters
static constexpr double SENSOR_RANGE    =  1.5;   // detection radius (m)
static constexpr double SENSOR_NOISE    =  0.30;  // std of range measurement (m)
static constexpr double MOTION_NOISE    =  0.10;  // std of displacement noise (m)
static constexpr double CMD_VEL         =  0.20;  // commanded step (m)
static constexpr int    N_PARTICLES     =  500;
static constexpr int    N_STEPS         =  100;

// Landmarks (door positions along corridor)
static const std::vector<double> LANDMARKS = {2.0, 5.5, 9.0, 14.0, 17.5};

static std::mt19937 rng(42);

double randn(double mean = 0.0, double std = 1.0) {
    std::normal_distribution<double> d(mean, std);
    return d(rng);
}

double randu(double lo, double hi) {
    std::uniform_real_distribution<double> d(lo, hi);
    return d(rng);
}

// Gaussian PDF (unnormalized weight, since we normalise anyway)
double gaussian_likelihood(double x, double mean, double sigma) {
    double diff = x - mean;
    return std::exp(-0.5 * diff * diff / (sigma * sigma));
}

// Particle filter core
struct ParticleFilter {
    std::vector<double> particles;
    std::vector<double> weights;

    ParticleFilter(int n, double lo, double hi) : particles(n), weights(n, 1.0 / n) {
        for (auto& p : particles)
            p = randu(lo, hi);
    }

    // Motion update: move each particle by u + noise
    void predict(double u, double noise_std) {
        for (auto& p : particles) {
            p += u + randn(0.0, noise_std);
            p = std::clamp(p, 0.0, CORRIDOR_LENGTH);
        }
    }

    // Measurement update: weight particles by P(z | particle, landmark)
    void update(double z_range, double landmark_pos, double sensor_noise) {
        for (int i = 0; i < (int)particles.size(); ++i) {
            double expected = std::abs(particles[i] - landmark_pos);
            weights[i] *= gaussian_likelihood(z_range, expected, sensor_noise);
            weights[i] += 1e-12;  // prevent exact zero
        }
        normalise_weights();
    }

    void normalise_weights() {
        double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        for (auto& w : weights) w /= sum;
    }

    // Effective sample size: 1 / sum(w_i^2)
    double effective_n() const {
        double sum_sq = 0.0;
        for (double w : weights) sum_sq += w * w;
        return 1.0 / sum_sq;
    }

    // Systematic resampling
    void resample() {
        int N = (int)particles.size();
        std::vector<double> cumsum(N);
        cumsum[0] = weights[0];
        for (int i = 1; i < N; ++i) cumsum[i] = cumsum[i-1] + weights[i];

        double step = 1.0 / N;
        double start = randu(0.0, step);

        std::vector<double> new_particles(N);
        int j = 0;
        for (int i = 0; i < N; ++i) {
            double pos = start + i * step;
            while (j < N - 1 && cumsum[j] < pos) ++j;
            new_particles[i] = particles[j];
        }
        particles = new_particles;
        std::fill(weights.begin(), weights.end(), 1.0 / N);
    }

    double mean() const {
        double m = 0.0;
        for (int i = 0; i < (int)particles.size(); ++i)
            m += weights[i] * particles[i];
        return m;
    }

    double stddev() const {
        double mu = mean();
        double var = 0.0;
        for (int i = 0; i < (int)particles.size(); ++i)
            var += weights[i] * (particles[i] - mu) * (particles[i] - mu);
        return std::sqrt(var);
    }
};

int main() {
    // CSV output
    std::ofstream csv("pf_trace.csv");
    csv << "step,x_true,x_est,x_std,eff_n,landmark_detected\n";

    double x_true = 3.0;  // true starting position

    ParticleFilter pf(N_PARTICLES, 0.0, CORRIDOR_LENGTH);

    // Initial estimate (uniform — very uncertain)
    double est = pf.mean();
    double std = pf.stddev();
    csv << 0 << "," << x_true << "," << est << "," << std << ","
        << pf.effective_n() << ",0\n";

    std::cout << "Step  x_true  x_est   x_std   eff_N   landmark\n";
    std::cout << std::string(55, '-') << "\n";

    for (int step = 1; step <= N_STEPS; ++step) {
        // True motion (small extra noise on truth)
        x_true += CMD_VEL + randn(0.0, 0.02);
        x_true  = std::clamp(x_true, 0.0, CORRIDOR_LENGTH);

        // Particle motion update
        pf.predict(CMD_VEL, MOTION_NOISE);

        // Check landmark detections (within SENSOR_RANGE of true position)
        bool detected = false;
        for (double lm : LANDMARKS) {
            if (std::abs(x_true - lm) < SENSOR_RANGE) {
                // Noisy range measurement
                double z = std::abs(x_true - lm) + randn(0.0, SENSOR_NOISE);
                z = std::max(0.0, z);
                pf.update(z, lm, SENSOR_NOISE);
                detected = true;
            }
        }

        // Resample if Neff drops below 50% of N
        if (pf.effective_n() < 0.5 * N_PARTICLES)
            pf.resample();

        est = pf.mean();
        std = pf.stddev();

        if (step % 10 == 0 || detected) {
            std::printf("%4d  %6.2f  %6.2f  %6.3f  %6.1f  %s\n",
                        step, x_true, est, std, pf.effective_n(),
                        detected ? "YES" : "---");
        }

        csv << step << "," << x_true << "," << est << "," << std << ","
            << pf.effective_n() << "," << (detected ? 1 : 0) << "\n";
    }

    csv.close();

    double final_error = std::abs(x_true - est);
    std::cout << "\nFinal position error: " << final_error << " m\n";
    std::cout << "Trace written to pf_trace.csv\n";

    return 0;
}
