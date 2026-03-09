/**
 * Author: Huseyn Kishiyev
 * ---------------
 * Monocular visual odometry front-end using OpenCV feature tracking.
 * Uses ORB features, BFMatcher, Essential matrix RANSAC, recoverPose.
 * Accumulates camera trajectory and writes poses to "vo_trajectory.csv".
 *
 * For production use, we can swap ORB with SuperPoint+LightGlue via ONNX Runtime C++ API.
 * The geometry and accumulation logic here is identical either way.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. && make -j4
 *
 * Or manually:
 *   g++ -O2 -std=c++17 vo_frontend.cpp \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -o vo_frontend
 *
 * Run:
 *   ./vo_frontend --images /path/to/frame/dir --fx 800 --fy 800 --cx 320 --cy 240
 *
 * Image directory should contain sequentially named images like 0000.png, 0001.png, ...
 *
 * Dependencies: OpenCV 4.x
 *   sudo apt install libopencv-dev
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>  // KLT optical flow

namespace fs = std::filesystem;

// Camera intrinsics 
struct Camera {
    double fx, fy, cx, cy;

    cv::Mat K() const {
        return (cv::Mat_<double>(3, 3) <<
            fx,  0, cx,
             0, fy, cy,
             0,  0,  1);
    }

    cv::Point2f normalize(const cv::Point2f& px) const {
        return {(float)((px.x - cx) / fx),
                (float)((px.y - cy) / fy)};
    }
};

// Feature extraction and matching
struct Frame {
    cv::Mat        image;
    std::vector<cv::KeyPoint> kpts;
    cv::Mat        descs;
    int            id = -1;
};

Frame extract_features(const cv::Mat& gray, int frame_id,
                        cv::Ptr<cv::ORB> detector) {
    Frame f;
    f.image = gray.clone();
    f.id    = frame_id;
    detector->detectAndCompute(gray, cv::noArray(), f.kpts, f.descs);
    return f;
}

struct MatchResult {
    std::vector<cv::Point2f> pts0, pts1;  // matched pixel coordinates
    int raw_matches = 0;
};

MatchResult match_frames(const Frame& f0, const Frame& f1, float ratio = 0.75f) {
    if (f0.descs.empty() || f1.descs.empty())
        return {};

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(f0.descs, f1.descs, knn, 2);

    MatchResult res;
    res.raw_matches = (int)knn.size();

    for (const auto& pair : knn) {
        if (pair.size() < 2) continue;
        if (pair[0].distance < ratio * pair[1].distance) {
            res.pts0.push_back(f0.kpts[pair[0].queryIdx].pt);
            res.pts1.push_back(f1.kpts[pair[0].trainIdx].pt);
        }
    }
    return res;
}

// Geometry
struct PoseResult {
    cv::Mat R, t;      // rotation and (unit) translation
    int     inliers;
    bool    valid = false;
};

PoseResult recover_pose(const MatchResult& matches, const Camera& cam) {
    if ((int)matches.pts0.size() < 8)
        return {};

    // Convert to normalized coordinates
    std::vector<cv::Point2f> n0, n1;
    n0.reserve(matches.pts0.size());
    n1.reserve(matches.pts1.size());
    for (size_t i = 0; i < matches.pts0.size(); ++i) {
        n0.push_back(cam.normalize(matches.pts0[i]));
        n1.push_back(cam.normalize(matches.pts1[i]));
    }

    // Threshold in normalized coords (~0.5 px / focal_length)
    double focal_avg = (cam.fx + cam.fy) * 0.5;
    double thresh_n  = 1.0 / focal_avg;

    cv::Mat E, mask;
    E = cv::findEssentialMat(n0, n1, 1.0, cv::Point2d(0, 0),
                             cv::RANSAC, 0.999, thresh_n, mask);
    if (E.empty()) return {};

    cv::Mat R, t;
    int inliers = cv::recoverPose(E, n0, n1, R, t, 1.0, cv::Point2d(0, 0), mask);

    if (inliers < 5) return {};

    return {R, t, inliers, true};
}

// Pose accumulation
struct Pose {
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t = cv::Mat::zeros(3, 1, CV_64F);

    // Apply relative transform (R_rel, t_rel) to the current accumulated pose.
    // Scale factor s is applied to translation (1.0 = unit, requires external scale).
    void compose(const cv::Mat& R_rel, const cv::Mat& t_rel, double scale = 1.0) {
        t = t + scale * (R * t_rel);
        R = R_rel * R;
    }

    cv::Vec3d position() const {
        return {t.at<double>(0), t.at<double>(1), t.at<double>(2)};
    }
};

// Keyframe selection (new keyframe if median optical flow > threshold)
double median_flow(const std::vector<cv::Point2f>& pts0,
                   const std::vector<cv::Point2f>& pts1) {
    if (pts0.size() != pts1.size() || pts0.empty()) return 0.0;
    std::vector<double> flows;
    flows.reserve(pts0.size());
    for (size_t i = 0; i < pts0.size(); ++i) {
        double dx = pts1[i].x - pts0[i].x;
        double dy = pts1[i].y - pts0[i].y;
        flows.push_back(std::sqrt(dx*dx + dy*dy));
    }
    std::sort(flows.begin(), flows.end());
    return flows[flows.size() / 2];
}

// Main
int main(int argc, char** argv) {
    // Parse args
    std::string image_dir = ".";
    double fx = 800, fy = 800, cx = 320, cy = 240;
    bool   visualize = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--images" && i+1 < argc) image_dir = argv[++i];
        else if (arg == "--fx"     && i+1 < argc) fx = std::stod(argv[++i]);
        else if (arg == "--fy"     && i+1 < argc) fy = std::stod(argv[++i]);
        else if (arg == "--cx"     && i+1 < argc) cx = std::stod(argv[++i]);
        else if (arg == "--cy"     && i+1 < argc) cy = std::stod(argv[++i]);
        else if (arg == "--vis")                  visualize = true;
        else if (arg == "--help") {
            std::cout << "Usage: vo_frontend --images DIR [--fx F] [--fy F] "
                         "[--cx C] [--cy C] [--vis]\n";
            return 0;
        }
    }

    Camera cam{fx, fy, cx, cy};

    // Collect sorted image paths
    std::vector<fs::path> image_paths;
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        auto ext = entry.path().extension().string();
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
            image_paths.push_back(entry.path());
    }
    std::sort(image_paths.begin(), image_paths.end());

    if (image_paths.size() < 2) {
        std::cerr << "Need at least 2 images in " << image_dir << "\n";
        return 1;
    }

    std::cout << "Processing " << image_paths.size() << " frames from " << image_dir << "\n";

    auto orb = cv::ORB::create(2000);

    // CSV output
    std::ofstream csv("vo_trajectory.csv");
    csv << "frame,x,y,z,inliers,matches\n";

    Pose  world_pose;
    Frame ref_frame;
    bool  initialized = false;

    // KLT tracking points for inter-keyframe tracking
    std::vector<cv::Point2f> track_pts;
    cv::Mat                   prev_gray;

    // Keyframe decision threshold (pixels)
    const double KF_FLOW_THRESH = 15.0;

    for (int idx = 0; idx < (int)image_paths.size(); ++idx) {
        cv::Mat bgr  = cv::imread(image_paths[idx].string());
        if (bgr.empty()) {
            std::cerr << "Could not read: " << image_paths[idx] << "\n";
            continue;
        }
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        if (!initialized) {
            ref_frame   = extract_features(gray, idx, orb);
            prev_gray   = gray.clone();
            initialized = true;
            csv << idx << ",0,0,0,0,0\n";
            std::cout << "Frame " << idx << " (init keyframe)\n";
            continue;
        }

        // KLT tracking to check if we need a new keyframe
        bool need_keyframe = false;
        if (!track_pts.empty()) {
            std::vector<cv::Point2f> tracked;
            std::vector<uchar>       status;
            std::vector<float>       err;
            cv::calcOpticalFlowPyrLK(prev_gray, gray, track_pts, tracked,
                                     status, err, cv::Size(21, 21), 3);
            std::vector<cv::Point2f> good0, good1;
            for (size_t i = 0; i < status.size(); ++i)
                if (status[i]) { good0.push_back(track_pts[i]); good1.push_back(tracked[i]); }

            double flow = median_flow(good0, good1);
            need_keyframe = (flow > KF_FLOW_THRESH) || (good0.size() < 80);
        } else {
            need_keyframe = true;
        }

        if (!need_keyframe) {
            // Update tracking points for next iter
            if (!ref_frame.kpts.empty()) {
                track_pts.clear();
                for (const auto& kp : ref_frame.kpts)
                    track_pts.push_back(kp.pt);
            }
            prev_gray = gray.clone();
            continue;
        }

        // New keyframe — run full matching
        Frame cur_frame = extract_features(gray, idx, orb);

        if (ref_frame.descs.empty() || cur_frame.descs.empty()) {
            ref_frame  = cur_frame;
            prev_gray  = gray.clone();
            continue;
        }

        MatchResult matches = match_frames(ref_frame, cur_frame);
        PoseResult  pose    = recover_pose(matches, cam);

        auto pos = world_pose.position();
        if (pose.valid) {
            // NOTE: t is unit vector — scale=1.0 gives relative trajectory shape,
            // not metric. For metric scale: integrate IMU or use known baseline.
            world_pose.compose(pose.R, pose.t, 1.0);
            pos = world_pose.position();

            std::printf("Frame %4d | matches %4zu | inliers %3d | "
                        "pos (%.3f, %.3f, %.3f)\n",
                        idx,
                        matches.pts0.size(),
                        pose.inliers,
                        pos[0], pos[1], pos[2]);
        } else {
            std::printf("Frame %4d | matches %4zu | pose FAILED\n",
                        idx, matches.pts0.size());
        }

        csv << idx << "," << pos[0] << "," << pos[1] << "," << pos[2]
            << "," << (pose.valid ? pose.inliers : 0)
            << "," << matches.pts0.size() << "\n";

        if (visualize && pose.valid) {
            // Draw matches
            cv::Mat match_img;
            std::vector<cv::DMatch> dummy_matches;
            for (size_t i = 0; i < matches.pts0.size(); ++i)
                dummy_matches.emplace_back((int)i, (int)i, 0.0f);

            // Convert pts to keypoints for drawMatches
            std::vector<cv::KeyPoint> kp0, kp1;
            for (auto& p : matches.pts0) kp0.emplace_back(p, 5.0f);
            for (auto& p : matches.pts1) kp1.emplace_back(p, 5.0f);

            cv::drawMatches(ref_frame.image, kp0, gray, kp1,
                            dummy_matches, match_img,
                            cv::Scalar::all(-1), cv::Scalar::all(-1),
                            {}, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            cv::imshow("VO Matches", match_img);
            if (cv::waitKey(30) == 27) break;  // ESC to quit
        }

        // Update reference keyframe and tracking state
        ref_frame  = cur_frame;
        track_pts.clear();
        for (const auto& kp : cur_frame.kpts)
            track_pts.push_back(kp.pt);
        prev_gray = gray.clone();
    }

    csv.close();
    std::cout << "\nTrajectory written to vo_trajectory.csv\n";

    return 0;
}
