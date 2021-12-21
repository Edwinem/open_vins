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

#include "TrackPython.h"
#include "track/Grider_FAST.h"

using namespace ov_core;

void TrackDescriptorPythonBase::feed_monocular(const CameraData &message, size_t msg_id) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Lock this data feed for this camera
  size_t cam_id = message.sensor_ids.at(msg_id);
  std::unique_lock<std::mutex> lck(mtx_feeds.at(cam_id));

  // Histogram equalize
  cv::Mat img, mask;
  if (histogram_method == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(message.images.at(msg_id), img);
  } else if (histogram_method == HistogramMethod::CLAHE) {
    double eq_clip_limit = 10.0;
    cv::Size eq_win_size = cv::Size(8, 8);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    clahe->apply(message.images.at(msg_id), img);
  } else {
    img = message.images.at(msg_id);
  }
  mask = message.masks.at(msg_id);

  // If we are the first frame (or have lost tracking), initialize our descriptors
  if (pts_last.find(cam_id) == pts_last.end() || pts_last[cam_id].empty()) {
    auto ret_tuple=perform_detection_monocular(img, mask);
    pts_last[cam_id]=std::get<0>(ret_tuple);
    desc_last[cam_id]=std::get<1>(ret_tuple);
    ids_last[cam_id]=std::get<2>(ret_tuple);
    img_last[cam_id] = img;
    img_mask_last[cam_id] = mask;
    return;
  }

  // Our new keypoints and descriptor for the new image
  std::vector<cv::KeyPoint> pts_new;
  cv::Mat desc_new;
  std::vector<size_t> ids_new;

  // First, extract new descriptors for this new image
  auto ret_tuple=perform_detection_monocular(img, mask);
  pts_new=std::get<0>(ret_tuple);
  desc_new=std::get<1>(ret_tuple);
  ids_new=std::get<2>(ret_tuple);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Our matches temporally
  std::vector<cv::DMatch> matches_ll;

  // Lets match temporally
  robust_match(pts_last[cam_id], pts_new, desc_last[cam_id], desc_new, cam_id, cam_id, matches_ll);
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left;
  std::vector<size_t> good_ids_left;
  cv::Mat good_desc_left;

  // Count how many we have tracked from the last time
  int num_tracklast = 0;

  // Loop through all current left to right points
  // We want to see if any of theses have matches to the previous frame
  // If we have a match new->old then we want to use that ID instead of the new one
  for (size_t i = 0; i < pts_new.size(); i++) {

    // Loop through all left matches, and find the old "train" id
    int idll = -1;
    for (size_t j = 0; j < matches_ll.size(); j++) {
      if (matches_ll[j].trainIdx == (int)i) {
        idll = matches_ll[j].queryIdx;
      }
    }

    // Then lets replace the current ID with the old ID if found
    // Else just append the current feature and its unique ID
    good_left.push_back(pts_new[i]);
    good_desc_left.push_back(desc_new.row((int)i));
    if (idll != -1) {
      good_ids_left.push_back(ids_last[cam_id][idll]);
      num_tracklast++;
    } else {
      good_ids_left.push_back(ids_new[i]);
    }
  }
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x, npt_l.y);
  }

  // Debug info
  // PRINT_DEBUG("LtoL = %d | good = %d | fromlast = %d\n",(int)matches_ll.size(),(int)good_left.size(),num_tracklast);

  // Move forward in time
  img_last[cam_id] = img;
  img_mask_last[cam_id] = mask;
  pts_last[cam_id] = good_left;
  ids_last[cam_id] = good_ids_left;
  desc_last[cam_id] = good_desc_left;
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Our timing information
  // PRINT_DEBUG("[TIME-DESC]: %.4f seconds for detection\n",(rT2-rT1).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-DESC]: %.4f seconds for matching\n",(rT3-rT2).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-DESC]: %.4f seconds for merging\n",(rT4-rT3).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-DESC]: %.4f seconds for feature DB update (%d features)\n",(rT5-rT4).total_microseconds() * 1e-6,
  // (int)good_left.size()); PRINT_DEBUG("[TIME-DESC]: %.4f seconds for total\n",(rT5-rT1).total_microseconds() * 1e-6);
}

void TrackDescriptorPythonBase::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Lock this data feed for this camera
  size_t cam_id_left = message.sensor_ids.at(msg_id_left);
  size_t cam_id_right = message.sensor_ids.at(msg_id_right);
  std::unique_lock<std::mutex> lck1(mtx_feeds.at(cam_id_left));
  std::unique_lock<std::mutex> lck2(mtx_feeds.at(cam_id_right));

  // Histogram equalize images
  cv::Mat img_left, img_right, mask_left, mask_right;
  if (histogram_method == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(message.images.at(msg_id_left), img_left);
    cv::equalizeHist(message.images.at(msg_id_right), img_right);
  } else if (histogram_method == HistogramMethod::CLAHE) {
    double eq_clip_limit = 10.0;
    cv::Size eq_win_size = cv::Size(8, 8);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    clahe->apply(message.images.at(msg_id_left), img_left);
    clahe->apply(message.images.at(msg_id_right), img_right);
  } else {
    img_left = message.images.at(msg_id_left);
    img_right = message.images.at(msg_id_right);
  }
  mask_left = message.masks.at(msg_id_left);
  mask_right = message.masks.at(msg_id_right);

  // If we are the first frame (or have lost tracking), initialize our descriptors
  if (pts_last[cam_id_left].empty() || pts_last[cam_id_right].empty()) {
    perform_detection_stereo(img_left, img_right, mask_left, mask_right, pts_last[cam_id_left], pts_last[cam_id_right],
                             desc_last[cam_id_left], desc_last[cam_id_right], cam_id_left, cam_id_right, ids_last[cam_id_left],
                             ids_last[cam_id_right]);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    return;
  }

  // Our new keypoints and descriptor for the new image
  std::vector<cv::KeyPoint> pts_left_new, pts_right_new;
  cv::Mat desc_left_new, desc_right_new;
  std::vector<size_t> ids_left_new, ids_right_new;

  // First, extract new descriptors for this new image
  perform_detection_stereo(img_left, img_right, mask_left, mask_right, pts_left_new, pts_right_new, desc_left_new, desc_right_new,
                           cam_id_left, cam_id_right, ids_left_new, ids_right_new);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Our matches temporally
  std::vector<cv::DMatch> matches_ll, matches_rr;
  parallel_for_(cv::Range(0, 2), LambdaBody([&](const cv::Range &range) {
    for (int i = range.start; i < range.end; i++) {
      bool is_left = (i == 0);
      robust_match(pts_last[is_left ? cam_id_left : cam_id_right], is_left ? pts_left_new : pts_right_new,
                   desc_last[is_left ? cam_id_left : cam_id_right], is_left ? desc_left_new : desc_right_new,
                   is_left ? cam_id_left : cam_id_right, is_left ? cam_id_left : cam_id_right,
                   is_left ? matches_ll : matches_rr);
    }
  }));
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left, good_right;
  std::vector<size_t> good_ids_left, good_ids_right;
  cv::Mat good_desc_left, good_desc_right;

  // Points must be of equal size
  assert(pts_last[cam_id_left].size() == pts_last[cam_id_right].size());
  assert(pts_left_new.size() == pts_right_new.size());

  // Count how many we have tracked from the last time
  int num_tracklast = 0;

  // Loop through all current left to right points
  // We want to see if any of theses have matches to the previous frame
  // If we have a match new->old then we want to use that ID instead of the new one
  for (size_t i = 0; i < pts_left_new.size(); i++) {

    // Loop through all left matches, and find the old "train" id
    int idll = -1;
    for (size_t j = 0; j < matches_ll.size(); j++) {
      if (matches_ll[j].trainIdx == (int)i) {
        idll = matches_ll[j].queryIdx;
      }
    }

    // Loop through all left matches, and find the old "train" id
    int idrr = -1;
    for (size_t j = 0; j < matches_rr.size(); j++) {
      if (matches_rr[j].trainIdx == (int)i) {
        idrr = matches_rr[j].queryIdx;
      }
    }

    // If we found a good stereo track from left to left, and right to right
    // Then lets replace the current ID with the old ID
    // We also check that we are linked to the same past ID value
    if (idll != -1 && idrr != -1 && ids_last[cam_id_left][idll] == ids_last[cam_id_right][idrr]) {
      good_left.push_back(pts_left_new[i]);
      good_right.push_back(pts_right_new[i]);
      good_desc_left.push_back(desc_left_new.row((int)i));
      good_desc_right.push_back(desc_right_new.row((int)i));
      good_ids_left.push_back(ids_last[cam_id_left][idll]);
      good_ids_right.push_back(ids_last[cam_id_right][idrr]);
      num_tracklast++;
    } else {
      // Else just append the current feature and its unique ID
      good_left.push_back(pts_left_new[i]);
      good_right.push_back(pts_right_new[i]);
      good_desc_left.push_back(desc_left_new.row((int)i));
      good_desc_right.push_back(desc_right_new.row((int)i));
      good_ids_left.push_back(ids_left_new[i]);
      good_ids_right.push_back(ids_left_new[i]);
    }
  }
  rT4 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  //===================================================================================

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    // Assert that our IDs are the same
    assert(good_ids_left.at(i) == good_ids_right.at(i));
    // Try to undistort the point
    cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(good_left.at(i).pt);
    cv::Point2f npt_r = camera_calib.at(cam_id_right)->undistort_cv(good_right.at(i).pt);
    // Append to the database
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                             npt_l.y);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                             npt_r.y);
  }

  // Debug info
  // PRINT_DEBUG("LtoL = %d | RtoR = %d | LtoR = %d | good = %d | fromlast = %d\n", (int)matches_ll.size(),
  //       (int)matches_rr.size(),(int)ids_left_new.size(),(int)good_left.size(),num_tracklast);

  // Move forward in time
  img_last[cam_id_left] = img_left;
  img_last[cam_id_right] = img_right;
  img_mask_last[cam_id_left] = mask_left;
  img_mask_last[cam_id_right] = mask_right;
  pts_last[cam_id_left] = good_left;
  pts_last[cam_id_right] = good_right;
  ids_last[cam_id_left] = good_ids_left;
  ids_last[cam_id_right] = good_ids_right;
  desc_last[cam_id_left] = good_desc_left;
  desc_last[cam_id_right] = good_desc_right;
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Our timing information
  // PRINT_DEBUG("[TIME-DESC]: %.4f seconds for detection\n",(rT2-rT1).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-DESC]: %.4f seconds for matching\n",(rT3-rT2).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-DESC]: %.4f seconds for merging\n",(rT4-rT3).total_microseconds() * 1e-6);
  // PRINT_DEBUG("[TIME-DESC]: %.4f seconds for feature DB update (%d features)\n",(rT5-rT4).total_microseconds() * 1e-6,
  // (int)good_left.size()); PRINT_DEBUG("[TIME-DESC]: %.4f seconds for total\n",(rT5-rT1).total_microseconds() * 1e-6);
}
