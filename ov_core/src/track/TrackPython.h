/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2021 Patrick Geneva
 * Copyright (C) 2021 Guoquan Huang
 * Copyright (C) 2021 OpenVINS Contributors
 * Copyright (C) 2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_CORE_TRACK_DESC_BASE_H
#define OV_CORE_TRACK_DESC_BASE_H

#include <opencv2/features2d.hpp>

#include "TrackBase.h"
#include "TrackDescriptor.h"

namespace ov_core {

/**
 * Base class for Python based trackers. It provides some helper functions to get the atomic ID.
 */
class TrackPythonBase: public TrackBase{
public:
  explicit TrackPythonBase(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool binocular,
      HistogramMethod histmethod)
  : TrackBase(cameras, numfeats, numaruco, binocular, histmethod){  }

  /**
 * @brief Process a new image
 * @param message Contains our timestamp, images, and camera ids
 */
  virtual void feed_new_camera(const CameraData &message) = 0;

  inline size_t get_currid_value() const{
    return currid;
  }

  inline void set_currid(const size_t val){
    currid=val;
  }

  inline void increment_currid(){
    currid++;
  }
};

/**
 * @brief Descriptor-based visual tracking
 *
 * Here we use descriptor matching to track features from one frame to the next.
 * We track both temporally, and across stereo pairs to get stereo constraints.
 * Right now we use ORB descriptors as we have found it is the fastest when computing descriptors.
 * Tracks are then rejected based on a ratio test and ransac.
 */
class TrackDescriptorPythonBase : public TrackDescriptor {

public:


  explicit TrackDescriptorPythonBase(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool binocular,
      HistogramMethod histmethod, int fast_threshold, int gridx, int gridy, int minpxdist, double knnratio, const std::string& descriptor_name)
  : TrackDescriptor(cameras, numfeats, numaruco, binocular, histmethod, fast_threshold, grid_x, gridy,  minpxdist, knnratio) {
    matcher=cv::DescriptorMatcher::create(descriptor_name);
  }

  inline size_t get_currid_value() const{
    return currid;
  }

  inline void set_currid(const size_t val){
    currid=val;
  }

  inline void increment_currid(){
    currid++;
  }

  /**
 * @brief Detects new features in the current image
 * @param img0 image we will detect features on
 * @param mask0 mask which has what ROI we do not want features in
 * @param pts0 vector of extracted keypoints
 * @param desc0 vector of the extracted descriptors
 * @param ids0 vector of all new IDs
 *
 * Given a set of images, and their currently extracted features, this will try to add new features.
 * We return all extracted descriptors here since we DO NOT need to do stereo tracking left to right.
 * Our vector of IDs will be later overwritten when we match features temporally to the previous frame's features.
 * See robust_match() for the matching.
 */

  virtual std::tuple<std::vector<cv::KeyPoint>,cv::Mat,std::vector<size_t>> perform_detection_monocular(const cv::Mat &img0, const cv::Mat &mask0)=0;


protected:
  /**
   * @brief Process a new monocular image
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id the camera index in message data vector
   */
  void feed_monocular(const CameraData &message, size_t msg_id);

  /**
   * @brief Process new stereo pair of images
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id_left first image index in message data vector
   * @param msg_id_right second image index in message data vector
   */
  void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_DESC_BASE_H */
