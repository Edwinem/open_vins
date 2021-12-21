

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include "alignment/AlignTrajectory.h"
#include "alignment/AlignUtils.h"
#include "calc/ResultTrajectory.h"
#include "cam/CamBase.h"
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "cvbind.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "track/TrackKLT.h"
#include "track/TrackPython.h"
#include "types/LandmarkRepresentation.h"
#include "utils/Loader.h"
#include "utils/Statistics.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"
#include "types/Type.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::map<size_t, bool>);
PYBIND11_MAKE_OPAQUE(std::map<size_t, Eigen::VectorXd>);
PYBIND11_MAKE_OPAQUE(std::map<size_t, std::pair<int, int>>);

//PYBIND11_MAKE_OPAQUE(std::vector<cv::KeyPoint>);
//namespace pybind11::detail {
//template<> struct is_comparable<cv::Mat> : std::false_type {};
//}

void show_image(cv::Mat image) {
  cv::imshow("image_from_Cpp", image);
  cv::waitKey(0);
}

PYBIND11_MODULE(PyOpenVINS, M) {
  M.doc() = "OpenVINS python bindings";


  M.def("show_image", &show_image, "A function that show an image",
        py::arg("image"));
  {
    py::bind_map<std::map<size_t, bool>>(M, "Map_Sizet_Bool");
    py::bind_map<std::map<size_t, Eigen::VectorXd>>(M, "Map_Sizet_VectorXd");
    py::bind_map<std::map<size_t, std::pair<int, int>>>(M, "Map_Sizet_Pair_Int_Int");
//    py::bind_vector<std::vector<cv::KeyPoint>>(M, "Vector_cvKeypoint");
  }
  using namespace ov_core;
  {
    M.def("rot_2_quat", &rot_2_quat);
    M.def("skew_x", &skew_x);
    M.def("quat_2_Rot", &quat_2_Rot);
    M.def("quat_multiply", &quat_multiply);
    M.def("vee", &vee);
    M.def("exp_so3", &exp_so3);
    M.def("log_so3", &log_so3);
    M.def("exp_se3", &exp_se3);
    M.def("log_se3", &log_se3);
    M.def("hat_se3", &hat_se3);
    M.def("Inv_se3", &Inv_se3);
    M.def("Inv", &Inv);
    M.def("Inv_se3", &Inv_se3);
    M.def("Omega", &Omega);
    M.def("quatnorm", &quatnorm);
    M.def("Jl_so3", &Jl_so3);
    M.def("Jr_so3", &Jr_so3);
  }

  {
      M.def("grid_fast",[](const cv::Mat &img, const cv::Mat &mask, int num_features, int grid_x,
                           int grid_y, int threshold, bool nonmaxSuppression){
        std::vector<cv::KeyPoint> pts;
        Grider_FAST::perform_griding(img,mask,pts,num_features,grid_x,grid_y,threshold,nonmaxSuppression);
        return pts;
      });
  }


  { // ov_core::CamBase file: line:42
    pybind11::class_<ov_core::CamBase, std::shared_ptr<ov_core::CamBase>> cl(
        M, "CamBase",
        "Base pinhole camera model class\n\n This is the base class for all our camera models.\n All these models are pinhole cameras, "
        "thus just have standard reprojection logic.\n\n See each base class for detailed examples on each model:\n  - \n\n\n  - \n\n ");
    cl.def("assign", (class ov_core::CamBase & (ov_core::CamBase::*)(const class ov_core::CamBase &)) & ov_core::CamBase::operator=,
           "C++: ov_core::CamBase::operator=(const class ov_core::CamBase &) --> class ov_core::CamBase &",
           pybind11::return_value_policy::automatic, pybind11::arg(""));
  }

  { // ov_core::Feature file: line:39
    pybind11::class_<ov_core::Feature, std::shared_ptr<ov_core::Feature>> cl(
        M, "Feature",
        "Sparse feature class used to collect measurements\n\n This feature class allows for holding of all tracking information for a "
        "given feature.\n Each feature has a unique ID assigned to it, and should have a set of feature tracks alongside it.\n See the "
        "FeatureDatabase class for details on how we load information into this, and how we delete features.");
    cl.def(pybind11::init([](ov_core::Feature const &o) { return new ov_core::Feature(o); }));
    cl.def(pybind11::init([]() { return new ov_core::Feature(); }));
    cl.def_readwrite("featid", &ov_core::Feature::featid);
    cl.def_readwrite("to_delete", &ov_core::Feature::to_delete);
    cl.def_readwrite("uvs", &ov_core::Feature::uvs);
    cl.def_readwrite("uvs_norm", &ov_core::Feature::uvs_norm);
    cl.def_readwrite("timestamps", &ov_core::Feature::timestamps);
    cl.def_readwrite("anchor_cam_id", &ov_core::Feature::anchor_cam_id);
    cl.def_readwrite("anchor_clone_timestamp", &ov_core::Feature::anchor_clone_timestamp);
    cl.def_readwrite("p_FinA", &ov_core::Feature::p_FinA);
    cl.def_readwrite("p_FinG", &ov_core::Feature::p_FinG);
    cl.def("clean_old_measurements",
           (void (ov_core::Feature::*)(const class std::vector<double, class std::allocator<double>> &)) &
               ov_core::Feature::clean_old_measurements,
           "Remove measurements that do not occur at passed timestamps.\n\n Given a series of valid timestamps, this will remove all "
           "measurements that have not occurred at these times.\n This would normally be used to ensure that the measurements that we have "
           "occur at our clone times.\n\n \n Vector of timestamps that our measurements must occur at\n\nC++: "
           "ov_core::Feature::clean_old_measurements(const class std::vector<double, class std::allocator<double> > &) --> void",
           pybind11::arg("valid_times"));
    cl.def("clean_invalid_measurements",
           (void (ov_core::Feature::*)(const class std::vector<double, class std::allocator<double>> &)) &
               ov_core::Feature::clean_invalid_measurements,
           "Remove measurements that occur at the invalid timestamps\n\n Given a series of invalid timestamps, this will remove all "
           "measurements that have occurred at these times.\n\n \n Vector of timestamps that our measurements should not\n\nC++: "
           "ov_core::Feature::clean_invalid_measurements(const class std::vector<double, class std::allocator<double> > &) --> void",
           pybind11::arg("invalid_times"));
    cl.def("clean_older_measurements", (void (ov_core::Feature::*)(double)) & ov_core::Feature::clean_older_measurements,
           "Remove measurements that are older then the specified timestamp.\n\n Given a valid timestamp, this will remove all "
           "measurements that have occured earlier then this.\n\n \n Timestamps that our measurements must occur after\n\nC++: "
           "ov_core::Feature::clean_older_measurements(double) --> void",
           pybind11::arg("timestamp"));
    cl.def("assign", (class ov_core::Feature & (ov_core::Feature::*)(const class ov_core::Feature &)) & ov_core::Feature::operator=,
           "C++: ov_core::Feature::operator=(const class ov_core::Feature &) --> class ov_core::Feature &",
           pybind11::return_value_policy::automatic, pybind11::arg(""));
  }
  { // ov_core::FeatureDatabase file: line:53
    pybind11::class_<ov_core::FeatureDatabase, std::shared_ptr<ov_core::FeatureDatabase>> cl(
        M, "FeatureDatabase",
        "Database containing features we are currently tracking.\n\n Each visual tracker has this database in it and it contains all "
        "features that we are tracking.\n The trackers will insert information into this database when they get new measurements from "
        "doing tracking.\n A user would then query this database for features that can be used for update and remove them after they have "
        "been processed.\n\n _class{m-note m-warning}\n\n \n A Note on Multi-Threading Support\n There is some support for asynchronous "
        "multi-threaded access.\n Since each feature is a pointer just directly returning and using them is not thread safe.\n Thus, to be "
        "thread safe, use the \"remove\" flag for each function which will remove it from this feature database.\n This prevents the "
        "trackers from adding new measurements and editing the feature information.\n For example, if you are asynchronous tracking "
        "cameras and you chose to update the state, then remove all features you will use in update.\n The feature trackers will continue "
        "to add features while you update, whose measurements can be used in the next update step!\n\n ");
    cl.def(pybind11::init([]() { return new ov_core::FeatureDatabase(); }));
    cl.def(
        "get_feature",
        [](ov_core::FeatureDatabase &o, unsigned long const &a0) -> std::shared_ptr<class ov_core::Feature> { return o.get_feature(a0); },
        "", pybind11::arg("id"));
    cl.def("get_feature",
           (class std::shared_ptr<class ov_core::Feature>(ov_core::FeatureDatabase::*)(unsigned long, bool)) &
               ov_core::FeatureDatabase::get_feature,
           "Get a specified feature\n \n\n What feature we want to get\n \n\n Set to true if you want to remove the feature from the "
           "database (you will need to handle the freeing of memory)\n \n\n Either a feature object, or null if it is not in the "
           "database.\n\nC++: ov_core::FeatureDatabase::get_feature(unsigned long, bool) --> class std::shared_ptr<class ov_core::Feature>",
           pybind11::arg("id"), pybind11::arg("remove"));
    cl.def("update_feature",
           (void (ov_core::FeatureDatabase::*)(unsigned long, double, unsigned long, float, float, float, float)) &
               ov_core::FeatureDatabase::update_feature,
           "Update a feature object\n \n\n ID of the feature we will update\n \n\n time that this measurement occured at\n \n\n which "
           "camera this measurement was from\n \n\n raw u coordinate\n \n\n raw v coordinate\n \n\n undistorted/normalized u coordinate\n "
           "\n\n undistorted/normalized v coordinate\n\n This will update a given feature based on the passed ID it has.\n It will create "
           "a new feature, if it is an ID that we have not seen before.\n\nC++: ov_core::FeatureDatabase::update_feature(unsigned long, "
           "double, unsigned long, float, float, float, float) --> void",
           pybind11::arg("id"), pybind11::arg("timestamp"), pybind11::arg("cam_id"), pybind11::arg("u"), pybind11::arg("v"),
           pybind11::arg("u_n"), pybind11::arg("v_n"));
    cl.def(
        "features_not_containing_newer",
        [](ov_core::FeatureDatabase &o,
           double const &a0) -> std::vector<class std::shared_ptr<class ov_core::Feature>,
                                            class std::allocator<class std::shared_ptr<class ov_core::Feature>>> {
          return o.features_not_containing_newer(a0);
        },
        "", pybind11::arg("timestamp"));
    cl.def(
        "features_not_containing_newer",
        [](ov_core::FeatureDatabase &o, double const &a0,
           bool const &a1) -> std::vector<class std::shared_ptr<class ov_core::Feature>,
                                          class std::allocator<class std::shared_ptr<class ov_core::Feature>>> {
          return o.features_not_containing_newer(a0, a1);
        },
        "", pybind11::arg("timestamp"), pybind11::arg("remove"));
    cl.def(
        "features_not_containing_newer",
        (class std::vector<class std::shared_ptr<class ov_core::Feature>,
                           class std::allocator<class std::shared_ptr<class ov_core::Feature>>>(ov_core::FeatureDatabase::*)(double, bool,
                                                                                                                             bool)) &
            ov_core::FeatureDatabase::features_not_containing_newer,
        "Get features that do not have newer measurement then the specified time.\n\n This function will return all features that do not a "
        "measurement at a time greater than the specified time.\n For example this could be used to get features that have not been "
        "successfully tracked into the newest frame.\n All features returned will not have any measurements occurring at a time greater "
        "then the specified.\n\nC++: ov_core::FeatureDatabase::features_not_containing_newer(double, bool, bool) --> class "
        "std::vector<class std::shared_ptr<class ov_core::Feature>, class std::allocator<class std::shared_ptr<class ov_core::Feature> > >",
        pybind11::arg("timestamp"), pybind11::arg("remove"), pybind11::arg("skip_deleted"));
    cl.def(
        "features_containing_older",
        [](ov_core::FeatureDatabase &o, double const &a0)
            -> std::vector<class std::shared_ptr<class ov_core::Feature>,
                           class std::allocator<class std::shared_ptr<class ov_core::Feature>>> { return o.features_containing_older(a0); },
        "", pybind11::arg("timestamp"));
    cl.def(
        "features_containing_older",
        [](ov_core::FeatureDatabase &o, double const &a0,
           bool const &a1) -> std::vector<class std::shared_ptr<class ov_core::Feature>,
                                          class std::allocator<class std::shared_ptr<class ov_core::Feature>>> {
          return o.features_containing_older(a0, a1);
        },
        "", pybind11::arg("timestamp"), pybind11::arg("remove"));
    cl.def(
        "features_containing_older",
        (class std::vector<class std::shared_ptr<class ov_core::Feature>,
                           class std::allocator<class std::shared_ptr<class ov_core::Feature>>>(ov_core::FeatureDatabase::*)(double, bool,
                                                                                                                             bool)) &
            ov_core::FeatureDatabase::features_containing_older,
        "Get features that has measurements older then the specified time.\n\n This will collect all features that have measurements "
        "occurring before the specified timestamp.\n For example, we would want to remove all features older then the last clone/state in "
        "our sliding window.\n\nC++: ov_core::FeatureDatabase::features_containing_older(double, bool, bool) --> class std::vector<class "
        "std::shared_ptr<class ov_core::Feature>, class std::allocator<class std::shared_ptr<class ov_core::Feature> > >",
        pybind11::arg("timestamp"), pybind11::arg("remove"), pybind11::arg("skip_deleted"));
    cl.def(
        "features_containing",
        [](ov_core::FeatureDatabase &o, double const &a0)
            -> std::vector<class std::shared_ptr<class ov_core::Feature>,
                           class std::allocator<class std::shared_ptr<class ov_core::Feature>>> { return o.features_containing(a0); },
        "", pybind11::arg("timestamp"));
    cl.def(
        "features_containing",
        [](ov_core::FeatureDatabase &o, double const &a0, bool const &a1)
            -> std::vector<class std::shared_ptr<class ov_core::Feature>,
                           class std::allocator<class std::shared_ptr<class ov_core::Feature>>> { return o.features_containing(a0, a1); },
        "", pybind11::arg("timestamp"), pybind11::arg("remove"));
    cl.def("features_containing",
           (class std::vector<class std::shared_ptr<class ov_core::Feature>,
                              class std::allocator<class std::shared_ptr<class ov_core::Feature>>>(ov_core::FeatureDatabase::*)(
               double, bool, bool)) &
               ov_core::FeatureDatabase::features_containing,
           "Get features that has measurements at the specified time.\n\n This function will return all features that have the specified "
           "time in them.\n This would be used to get all features that occurred at a specific clone/state.\n\nC++: "
           "ov_core::FeatureDatabase::features_containing(double, bool, bool) --> class std::vector<class std::shared_ptr<class "
           "ov_core::Feature>, class std::allocator<class std::shared_ptr<class ov_core::Feature> > >",
           pybind11::arg("timestamp"), pybind11::arg("remove"), pybind11::arg("skip_deleted"));
    cl.def("cleanup", (void (ov_core::FeatureDatabase::*)()) & ov_core::FeatureDatabase::cleanup,
           "This function will delete all features that have been used up.\n\n If a feature was unable to be used, it will still remain "
           "since it will not have a delete flag set\n\nC++: ov_core::FeatureDatabase::cleanup() --> void");
    cl.def("cleanup_measurements", (void (ov_core::FeatureDatabase::*)(double)) & ov_core::FeatureDatabase::cleanup_measurements,
           "This function will delete all feature measurements that are older then the specified timestamp\n\nC++: "
           "ov_core::FeatureDatabase::cleanup_measurements(double) --> void",
           pybind11::arg("timestamp"));
    cl.def("cleanup_measurements_exact",
           (void (ov_core::FeatureDatabase::*)(double)) & ov_core::FeatureDatabase::cleanup_measurements_exact,
           "This function will delete all feature measurements that are at the specified timestamp\n\nC++: "
           "ov_core::FeatureDatabase::cleanup_measurements_exact(double) --> void",
           pybind11::arg("timestamp"));
    cl.def("size", (unsigned long (ov_core::FeatureDatabase::*)()) & ov_core::FeatureDatabase::size,
           "Returns the size of the feature database\n\nC++: ov_core::FeatureDatabase::size() --> unsigned long");
    cl.def("get_internal_data",
           (class std::unordered_map<
               unsigned long, class std::shared_ptr<class ov_core::Feature>, struct std::hash<unsigned long>,
               struct std::equal_to<unsigned long>,
               class std::allocator<struct std::pair<const unsigned long, class std::shared_ptr<class ov_core::Feature>>>>(
               ov_core::FeatureDatabase::*)()) &
               ov_core::FeatureDatabase::get_internal_data,
           "Returns the internal data (should not normally be used)\n\nC++: ov_core::FeatureDatabase::get_internal_data() --> class "
           "std::unordered_map<unsigned long, class std::shared_ptr<class ov_core::Feature>, struct std::hash<unsigned long>, struct "
           "std::equal_to<unsigned long>, class std::allocator<struct std::pair<const unsigned long, class std::shared_ptr<class "
           "ov_core::Feature> > > >");
    cl.def("append_new_measurements",
           (void (ov_core::FeatureDatabase::*)(const class std::shared_ptr<class ov_core::FeatureDatabase> &)) &
               ov_core::FeatureDatabase::append_new_measurements,
           "Will update the passed database with this database's latest feature information.\n\nC++: "
           "ov_core::FeatureDatabase::append_new_measurements(const class std::shared_ptr<class ov_core::FeatureDatabase> &) --> void",
           pybind11::arg("database"));
  }

  { // ov_core::ImuData file: line:34
    pybind11::class_<ov_core::ImuData, std::shared_ptr<ov_core::ImuData>> cl(M, "ImuData",
                                                                             "Struct for a single imu measurement (time, wm, am)");
    cl.def(pybind11::init([](ov_core::ImuData const &o) { return new ov_core::ImuData(o); }));
    cl.def(pybind11::init([]() { return new ov_core::ImuData(); }));
    cl.def_readwrite("timestamp", &ov_core::ImuData::timestamp);
    cl.def_readwrite("wm", &ov_core::ImuData::wm);
    cl.def_readwrite("am", &ov_core::ImuData::am);
    cl.def("assign", (struct ov_core::ImuData & (ov_core::ImuData::*)(const struct ov_core::ImuData &)) & ov_core::ImuData::operator=,
           "C++: ov_core::ImuData::operator=(const struct ov_core::ImuData &) --> struct ov_core::ImuData &",
           pybind11::return_value_policy::automatic, pybind11::arg(""));
  }
  { // ov_core::CameraData file: line:55
    pybind11::class_<ov_core::CameraData, std::shared_ptr<ov_core::CameraData>> cl(
        M, "CameraData",
        "Struct for a collection of camera measurements.\n\n For each image we have a camera id and timestamp that it occured at.\n If "
        "there are multiple cameras we will treat it as pair-wise stereo tracking.");
    cl.def(pybind11::init([](ov_core::CameraData const &o) { return new ov_core::CameraData(o); }));
    cl.def(pybind11::init([]() { return new ov_core::CameraData(); }));
    cl.def_readwrite("timestamp", &ov_core::CameraData::timestamp);
    cl.def_readwrite("sensor_ids", &ov_core::CameraData::sensor_ids);
    cl.def_readwrite("images", &ov_core::CameraData::images);
    cl.def_readwrite("masks", &ov_core::CameraData::masks);
    cl.def("assign",
           (struct ov_core::CameraData & (ov_core::CameraData::*)(const struct ov_core::CameraData &)) & ov_core::CameraData::operator=,
           "C++: ov_core::CameraData::operator=(const struct ov_core::CameraData &) --> struct ov_core::CameraData &",
           pybind11::return_value_policy::automatic, pybind11::arg(""));
  }

  {
    pybind11::class_<ov_type::IMU, std::shared_ptr<ov_type::IMU>> cl(
        M, "IMUType");

    cl.def("Rot",&ov_type::IMU::Rot);
    cl.def("quat",&ov_type::IMU::quat);
    cl.def("vel",&ov_type::IMU::vel);
    cl.def("pos",&ov_type::IMU::pos);
    cl.def("quat",&ov_type::IMU::quat);


  }

  { // ov_type::LandmarkRepresentation file: line:32

    pybind11::enum_<ov_type::LandmarkRepresentation::Representation>(M, "LandmarkRepresentationTypes", pybind11::arithmetic(),
                                                                     "What feature representation our state can use")
        .value("GLOBAL_3D", ov_type::LandmarkRepresentation::GLOBAL_3D)
        .value("GLOBAL_FULL_INVERSE_DEPTH", ov_type::LandmarkRepresentation::GLOBAL_FULL_INVERSE_DEPTH)
        .value("ANCHORED_3D", ov_type::LandmarkRepresentation::ANCHORED_3D)
        .value("ANCHORED_FULL_INVERSE_DEPTH", ov_type::LandmarkRepresentation::ANCHORED_FULL_INVERSE_DEPTH)
        .value("ANCHORED_MSCKF_INVERSE_DEPTH", ov_type::LandmarkRepresentation::ANCHORED_MSCKF_INVERSE_DEPTH)
        .value("ANCHORED_INVERSE_DEPTH_SINGLE", ov_type::LandmarkRepresentation::ANCHORED_INVERSE_DEPTH_SINGLE)
        .value("UNKNOWN", ov_type::LandmarkRepresentation::UNKNOWN)
        .export_values();

    pybind11::class_<ov_type::LandmarkRepresentation, std::shared_ptr<ov_type::LandmarkRepresentation>> cl(
        M, "LandmarkRepresentation", "Class has useful feature representation types");

    cl.def_static("as_string",
                  (std::string(*)(enum ov_type::LandmarkRepresentation::Representation)) & ov_type::LandmarkRepresentation::as_string,
                  "Returns a string representation of this enum value.\n Used to debug print out what the user has selected as the "
                  "representation.\n \n\n  Representation we want to check\n \n\n String version of the passed enum\n\nC++: "
                  "ov_type::LandmarkRepresentation::as_string(enum ov_type::LandmarkRepresentation::Representation) --> std::string",
                  pybind11::arg("feat_representation"));
    cl.def_static(
        "from_string",
        (enum ov_type::LandmarkRepresentation::Representation(*)(const std::string &)) & ov_type::LandmarkRepresentation::from_string,
        "Returns a string representation of this enum value.\n Used to debug print out what the user has selected as the representation.\n "
        "\n\n String we want to find the enum of\n \n\n Representation, will be \"unknown\" if we coun't parse it\n\nC++: "
        "ov_type::LandmarkRepresentation::from_string(const std::string &) --> enum ov_type::LandmarkRepresentation::Representation",
        pybind11::arg("feat_representation"));
    cl.def_static(
        "is_relative_representation",
        (bool (*)(enum ov_type::LandmarkRepresentation::Representation)) & ov_type::LandmarkRepresentation::is_relative_representation,
        "Helper function that checks if the passed feature representation is a relative or global\n \n\n Representation we want to check\n "
        "\n\n True if it is a relative representation\n\nC++: ov_type::LandmarkRepresentation::is_relative_representation(enum "
        "ov_type::LandmarkRepresentation::Representation) --> bool",
        pybind11::arg("feat_representation"));
  }

  struct PyCallBack_ov_core_TrackBase : public ov_core::TrackBase {
    using ov_core::TrackBase::TrackBase;

    void feed_new_camera(const struct ov_core::CameraData & a0) override {
      pybind11::gil_scoped_acquire gil;
      pybind11::function overload = pybind11::get_overload(static_cast<const ov_core::TrackBase *>(this), "feed_new_camera");
      if (overload) {
        auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
        if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
          static pybind11::detail::override_caster_t<void> caster;
          return pybind11::detail::cast_ref<void>(std::move(o), caster);
        }
        else return pybind11::detail::cast_safe<void>(std::move(o));
      }
      pybind11::pybind11_fail("Tried to call pure virtual function \"TrackBase::feed_new_camera\"");
    }
  };

//  py::class_<std::atomic<size_t>> cl(M, "AtomicSizet");
//  cl.def

    { // ov_core::TrackBase file:ov_core/src/track/TrackBase.h line:70
      pybind11::enum_<ov_core::TrackBase::HistogramMethod>(M, "HistogramMethod", pybind11::arithmetic(), "Desired pre-processing image method.")
          .value("NONE", ov_core::TrackBase::NONE)
          .value("HISTOGRAM", ov_core::TrackBase::HISTOGRAM)
          .value("CLAHE", ov_core::TrackBase::CLAHE)
          .export_values();

      pybind11::class_<ov_core::TrackBase, std::shared_ptr<ov_core::TrackBase>, PyCallBack_ov_core_TrackBase> cl(M, "TrackBase", "Visual feature tracking base class\n\n This is the base class for all our visual trackers.\n The goal here is to provide a common interface so all underlying trackers can simply hide away all the complexities.\n We have something called the \"feature database\" which has all the tracking information inside of it.\n The user can ask this database for features which can then be used in an MSCKF or batch-based setting.\n The feature tracks store both the raw (distorted) and undistorted/normalized values.\n Right now we just support two camera models, see: undistort_point_brown() and undistort_point_fisheye().\n\n _class{m-note m-warning}\n\n \n A Note on Multi-Threading Support\n There is some support for asynchronous multi-threaded feature tracking of independent cameras.\n The key assumption during implementation is that the user will not try to track on the same camera in parallel, and instead call on\n different cameras. For example, if I have two cameras, I can either sequentially call the feed function, or I spin each of these into\n separate threads and wait for their return. The \n\n\n that all features have unique id values. We also have mutex for access for the calibration and previous images and tracks (used during\n visualization). It should be noted that if a thread calls visualization, it might hang or the feed thread might, due to acquiring the\n mutex for that specific camera id / feed.\n\n This base class also handles most of the heavy lifting with the visualization, but the sub-classes can override\n this and do their own logic if they want (i.e. the TrackAruco has its own logic for visualization).\n This visualization needs access to the prior images and their tracks, thus must synchronise in the case of multi-threading.\n This shouldn't impact performance, but high frequency visualization calls can negatively effect the performance.");
      cl.def( pybind11::init<class std::unordered_map<unsigned long, class std::shared_ptr<class ov_core::CamBase>, struct std::hash<unsigned long>, struct std::equal_to<unsigned long>, class std::allocator<struct std::pair<const unsigned long, class std::shared_ptr<class ov_core::CamBase> > > >, int, int, bool, enum ov_core::TrackBase::HistogramMethod>(), pybind11::arg("cameras"), pybind11::arg("numfeats"), pybind11::arg("numaruco"), pybind11::arg("stereo"), pybind11::arg("histmethod") );



      cl.def("feed_new_camera", (void (ov_core::TrackBase::*)(const struct ov_core::CameraData &)) &ov_core::TrackBase::feed_new_camera, "Process a new image\n \n\n Contains our timestamp, images, and camera ids\n\nC++: ov_core::TrackBase::feed_new_camera(const struct ov_core::CameraData &) --> void", pybind11::arg("message"));
      cl.def("get_feature_database", (class std::shared_ptr<class ov_core::FeatureDatabase> (ov_core::TrackBase::*)()) &ov_core::TrackBase::get_feature_database, "Get the feature database with all the track information\n \n\n FeatureDatabase pointer that one can query for features\n\nC++: ov_core::TrackBase::get_feature_database() --> class std::shared_ptr<class ov_core::FeatureDatabase>");
      cl.def("change_feat_id", (void (ov_core::TrackBase::*)(unsigned long, unsigned long)) &ov_core::TrackBase::change_feat_id, "Changes the ID of an actively tracked feature to another one.\n\n This function can be helpfull if you detect a loop-closure with an old frame.\n One could then change the id of an active feature to match the old feature id!\n\n \n Old id we want to change\n \n\n Id we want to change the old id to\n\nC++: ov_core::TrackBase::change_feat_id(unsigned long, unsigned long) --> void", pybind11::arg("id_old"), pybind11::arg("id_new"));
    }


  {

    struct PyCallBack_ov_core_TrackPythonBase : public ov_core::TrackPythonBase {
      using ov_core::TrackPythonBase::TrackPythonBase;

      void feed_new_camera(const struct ov_core::CameraData & a0) override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function overload = pybind11::get_overload(static_cast<const ov_core::TrackPythonBase *>(this), "feed_new_camera");
        if (overload) {
          auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
          if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
            static pybind11::detail::override_caster_t<void> caster;
            return pybind11::detail::cast_ref<void>(std::move(o), caster);
          }
          else return pybind11::detail::cast_safe<void>(std::move(o));
        }
        pybind11::pybind11_fail("Tried to call pure virtual function \"TrackBase::feed_new_camera\"");
      }
    };
    pybind11::class_<ov_core::TrackPythonBase, std::shared_ptr<ov_core::TrackPythonBase>, PyCallBack_ov_core_TrackPythonBase> cl(M, "TrackPythonBase", "Visual feature tracking base class\n\n This is the base class for all our visual trackers.\n The goal here is to provide a common interface so all underlying trackers can simply hide away all the complexities.\n We have something called the \"feature database\" which has all the tracking information inside of it.\n The user can ask this database for features which can then be used in an MSCKF or batch-based setting.\n The feature tracks store both the raw (distorted) and undistorted/normalized values.\n Right now we just support two camera models, see: undistort_point_brown() and undistort_point_fisheye().\n\n _class{m-note m-warning}\n\n \n A Note on Multi-Threading Support\n There is some support for asynchronous multi-threaded feature tracking of independent cameras.\n The key assumption during implementation is that the user will not try to track on the same camera in parallel, and instead call on\n different cameras. For example, if I have two cameras, I can either sequentially call the feed function, or I spin each of these into\n separate threads and wait for their return. The \n\n\n that all features have unique id values. We also have mutex for access for the calibration and previous images and tracks (used during\n visualization). It should be noted that if a thread calls visualization, it might hang or the feed thread might, due to acquiring the\n mutex for that specific camera id / feed.\n\n This base class also handles most of the heavy lifting with the visualization, but the sub-classes can override\n this and do their own logic if they want (i.e. the TrackAruco has its own logic for visualization).\n This visualization needs access to the prior images and their tracks, thus must synchronise in the case of multi-threading.\n This shouldn't impact performance, but high frequency visualization calls can negatively effect the performance.");

    //cl.def("feed_new_camera", (void (ov_core::TrackPythonBase::*)(const struct ov_core::CameraData &)) &ov_core::TrackPythonBase::feed_new_camera, "Process a new image\n \n\n Contains our timestamp, images, and camera ids\n\nC++: ov_core::TrackBase::feed_new_camera(const struct ov_core::CameraData &) --> void", pybind11::arg("message"));

    cl.def("get_currid_value",&ov_core::TrackPythonBase::get_currid_value);
    cl.def("increment_currid",&ov_core::TrackPythonBase::increment_currid);
    cl.def("set_currid",&ov_core::TrackPythonBase::set_currid);
  }

  struct PyCallBack_ov_core_TrackDescriptorPythonBase : public ov_core::TrackDescriptorPythonBase {
    using ov_core::TrackDescriptorPythonBase::TrackDescriptorPythonBase;

    std::tuple<std::vector<cv::KeyPoint>,cv::Mat,std::vector<size_t>> perform_detection_monocular(const cv::Mat &img0, const cv::Mat &mask0) override {
      pybind11::gil_scoped_acquire gil;
      pybind11::function overload = pybind11::get_overload(static_cast<const ov_core::TrackDescriptorPythonBase *>(this), "perform_detection_monocular");
      if (overload) {
        auto o = overload.operator()<pybind11::return_value_policy::reference>(img0,mask0);
        if (!pybind11::isinstance<pybind11::tuple>(o))
        {
          std::logic_error("Type should be a tuple!");
        }
        if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
          static pybind11::detail::override_caster_t<void> caster;
          return pybind11::detail::cast_ref<std::tuple<std::vector<cv::KeyPoint>,cv::Mat,std::vector<size_t>>>(std::move(o), caster);
        }
        else{
          pybind11::tuple tup_ret_val = pybind11::reinterpret_borrow<pybind11::tuple>(o);
          return std::make_tuple<std::vector<cv::KeyPoint>,cv::Mat,std::vector<size_t>>(tup_ret_val[0].cast<std::vector<cv::KeyPoint>>(),tup_ret_val[1].cast<cv::Mat>().clone(),tup_ret_val[2].cast<std::vector<size_t>>());
        }
        }
      pybind11::pybind11_fail("Tried to call pure virtual function \"TrackDescriptorBase::perform_detection_monocular\"");
    }
  };

  { // ov_core::TrackDescriptorBase file:ov_core/src/track/TrackDescriptorBase.h line:70

    pybind11::class_<ov_core::TrackDescriptorPythonBase, std::shared_ptr<ov_core::TrackDescriptorPythonBase>, PyCallBack_ov_core_TrackDescriptorPythonBase,ov_core::TrackBase> cl(M, "TrackDescriptorBase", "Base class for descriptor based trackers.");
    cl.def( pybind11::init<class std::unordered_map<unsigned long, class std::shared_ptr<class ov_core::CamBase>, struct std::hash<unsigned long>, struct std::equal_to<unsigned long>, class std::allocator<struct std::pair<const unsigned long, class std::shared_ptr<class ov_core::CamBase> > > >, int, int, bool, enum ov_core::TrackBase::HistogramMethod, int, int, int, int,double,const std::string&>() );



   // cl.def("feed_new_camera", (void (ov_core::TrackDescriptorPythonBase::*)(const struct ov_core::CameraData &)) &ov_core::TrackDescriptorPythonBase::feed_new_camera, "Process a new image\n \n\n Contains our timestamp, images, and camera ids\n\nC++: ov_core::TrackBase::feed_new_camera(const struct ov_core::CameraData &) --> void", pybind11::arg("message"));
//    cl.def("perform_detection_monocular",&TrackDescriptorBase::perform_detection_monocular);
  }

  { // ov_core::TrackDescriptorBase file:ov_core/src/track/TrackDescriptorBase.h line:70

    pybind11::class_<ov_core::TrackDescriptor, std::shared_ptr<ov_core::TrackDescriptor>, ov_core::TrackBase> cl(M, "TrackDescriptor", "Descriptor based tracker.");
    cl.def( pybind11::init<class std::unordered_map<unsigned long, class std::shared_ptr<class ov_core::CamBase>, struct std::hash<unsigned long>, struct std::equal_to<unsigned long>, class std::allocator<struct std::pair<const unsigned long, class std::shared_ptr<class ov_core::CamBase> > > >, int, int, bool, enum ov_core::TrackBase::HistogramMethod, int, int, int, int,double>() );



  }

      { // ov_core::TrackKLT file: line:39
        pybind11::class_<ov_core::TrackKLT, std::shared_ptr<ov_core::TrackKLT>, ov_core::TrackBase> cl(M, "TrackKLT", "KLT tracking of features.\n\n This is the implementation of a KLT visual frontend for tracking sparse features.\n We can track either monocular cameras across time (temporally) along with\n stereo cameras which we also track across time (temporally) but track from left to right\n to find the stereo correspondence information also.\n This uses the [calcOpticalFlowPyrLK](https://github.com/opencv/opencv/blob/master/modules/video/src/lkpyramid.cpp)\n OpenCV function to do the KLT tracking.");
        cl.def( pybind11::init<class std::unordered_map<unsigned long, class std::shared_ptr<class ov_core::CamBase>, struct std::hash<unsigned long>, struct std::equal_to<unsigned long>, class std::allocator<struct std::pair<const unsigned long, class std::shared_ptr<class ov_core::CamBase> > > >, int, int, bool, enum ov_core::TrackBase::HistogramMethod, int, int, int, int>(), pybind11::arg("cameras"), pybind11::arg("numfeats"), pybind11::arg("numaruco"), pybind11::arg("binocular"), pybind11::arg("histmethod"), pybind11::arg("fast_threshold"), pybind11::arg("gridx"), pybind11::arg("gridy"), pybind11::arg("minpxdist") );

        cl.def("feed_new_camera", (void (ov_core::TrackKLT::*)(const struct ov_core::CameraData &)) &ov_core::TrackKLT::feed_new_camera, "Process a new image\n \n\n Contains our timestamp, images, and camera ids\n\nC++: ov_core::TrackKLT::feed_new_camera(const struct ov_core::CameraData &) --> void", pybind11::arg("message"));
      }

  { // ov_msckf::StateOptions file: line:35
    pybind11::class_<ov_msckf::StateOptions, std::shared_ptr<ov_msckf::StateOptions>> cl(M, "StateOptions",
                                                                                         "Struct which stores all our filter options");
    cl.def(pybind11::init([](ov_msckf::StateOptions const &o) { return new ov_msckf::StateOptions(o); }));
    cl.def(pybind11::init([]() { return new ov_msckf::StateOptions(); }));
    cl.def_readwrite("do_fej", &ov_msckf::StateOptions::do_fej);
    cl.def_readwrite("imu_avg", &ov_msckf::StateOptions::imu_avg);
    cl.def_readwrite("use_rk4_integration", &ov_msckf::StateOptions::use_rk4_integration);
    cl.def_readwrite("do_calib_camera_pose", &ov_msckf::StateOptions::do_calib_camera_pose);
    cl.def_readwrite("do_calib_camera_intrinsics", &ov_msckf::StateOptions::do_calib_camera_intrinsics);
    cl.def_readwrite("do_calib_camera_timeoffset", &ov_msckf::StateOptions::do_calib_camera_timeoffset);
    cl.def_readwrite("max_clone_size", &ov_msckf::StateOptions::max_clone_size);
    cl.def_readwrite("max_slam_features", &ov_msckf::StateOptions::max_slam_features);
    cl.def_readwrite("max_slam_in_update", &ov_msckf::StateOptions::max_slam_in_update);
    cl.def_readwrite("max_msckf_in_update", &ov_msckf::StateOptions::max_msckf_in_update);
    cl.def_readwrite("max_aruco_features", &ov_msckf::StateOptions::max_aruco_features);
    cl.def_readwrite("num_cameras", &ov_msckf::StateOptions::num_cameras);
    cl.def_readwrite("feat_rep_msckf", &ov_msckf::StateOptions::feat_rep_msckf);
    cl.def_readwrite("feat_rep_slam", &ov_msckf::StateOptions::feat_rep_slam);
    cl.def_readwrite("feat_rep_aruco", &ov_msckf::StateOptions::feat_rep_aruco);
    cl.def("print", (void (ov_msckf::StateOptions::*)()) & ov_msckf::StateOptions::print,
           "Nice print function of what parameters we have loaded\n\nC++: ov_msckf::StateOptions::print() --> void");
    cl.def("assign",
           (struct ov_msckf::StateOptions & (ov_msckf::StateOptions::*)(const struct ov_msckf::StateOptions &)) &
               ov_msckf::StateOptions::operator=,
           "C++: ov_msckf::StateOptions::operator=(const struct ov_msckf::StateOptions &) --> struct ov_msckf::StateOptions &",
           pybind11::return_value_policy::automatic, pybind11::arg(""));
  }
  { // ov_msckf::StateHelper file: line:44
    pybind11::class_<ov_msckf::StateHelper, std::shared_ptr<ov_msckf::StateHelper>> cl(M, "StateHelper", "Helper which manipulates the State and its covariance.\n\n In general, this class has all the core logic for an Extended Kalman Filter (EKF)-based system.\n This has all functions that change the covariance along with addition and removing elements from the state.\n All functions here are static, and thus are self-contained so that in the future multiple states could be tracked and updated.\n We recommend you look directly at the code for this class for clarity on what exactly we are doing in each and the matching documentation\n pages.");
    cl.def_static("marginalize", (void (*)(class std::shared_ptr<class ov_msckf::State>, class std::shared_ptr<class ov_type::Type>)) &ov_msckf::StateHelper::marginalize, "Marginalizes a variable, properly modifying the ordering/covariances in the state\n\n This function can support any Type variable out of the box.\n Right now the marginalization of a sub-variable/type is not supported.\n For example if you wanted to just marginalize the orientation of a PoseJPL, that isn't supported.\n We will first remove the rows and columns corresponding to the type (i.e. do the marginalization).\n After we update all the type ids so that they take into account that the covariance has shrunk in parts of it.\n\n \n Pointer to state\n \n\n Pointer to variable to marginalize\n\nC++: ov_msckf::StateHelper::marginalize(class std::shared_ptr<class ov_msckf::State>, class std::shared_ptr<class ov_type::Type>) --> void", pybind11::arg("state"), pybind11::arg("marg"));
    cl.def_static("clone", (class std::shared_ptr<class ov_type::Type> (*)(class std::shared_ptr<class ov_msckf::State>, class std::shared_ptr<class ov_type::Type>)) &ov_msckf::StateHelper::clone, "Clones \"variable to clone\" and places it at end of covariance\n \n\n Pointer to state\n \n\n Pointer to variable that will be cloned\n\nC++: ov_msckf::StateHelper::clone(class std::shared_ptr<class ov_msckf::State>, class std::shared_ptr<class ov_type::Type>) --> class std::shared_ptr<class ov_type::Type>", pybind11::arg("state"), pybind11::arg("variable_to_clone"));
    cl.def_static("marginalize_old_clone", (void (*)(class std::shared_ptr<class ov_msckf::State>)) &ov_msckf::StateHelper::marginalize_old_clone, "Remove the oldest clone, if we have more then the max clone count!!\n\n This will marginalize the clone from our covariance, and remove it from our state.\n This is mainly a helper function that we can call after each update.\n It will marginalize the clone specified by State::margtimestep() which should return a clone timestamp.\n\n \n Pointer to state\n\nC++: ov_msckf::StateHelper::marginalize_old_clone(class std::shared_ptr<class ov_msckf::State>) --> void", pybind11::arg("state"));
    cl.def_static("marginalize_slam", (void (*)(class std::shared_ptr<class ov_msckf::State>)) &ov_msckf::StateHelper::marginalize_slam, "Marginalize bad SLAM features\n \n\n Pointer to state\n\nC++: ov_msckf::StateHelper::marginalize_slam(class std::shared_ptr<class ov_msckf::State>) --> void", pybind11::arg("state"));
    cl.def_static("get_marginal_covariance",ov_msckf::StateHelper::get_marginal_covariance);
    cl.def_static("get_full_covariance",ov_msckf::StateHelper::get_full_covariance);
  }
  { // ov_msckf::Propagator file: line:40
    pybind11::class_<ov_msckf::Propagator, std::shared_ptr<ov_msckf::Propagator>> cl(M, "Propagator", "Performs the state covariance and mean propagation using imu measurements\n\n We will first select what measurements we need to propagate with.\n We then compute the state transition matrix at each step and update the state and covariance.\n For derivations look at \n\n\n ");
    cl.def( pybind11::init<struct ov_msckf::Propagator::NoiseManager, double>(), pybind11::arg("noises"), pybind11::arg("gravity_mag") );

    cl.def( pybind11::init( [](ov_msckf::Propagator const &o){ return new ov_msckf::Propagator(o); } ) );
    cl.def("feed_imu", (void (ov_msckf::Propagator::*)(const struct ov_core::ImuData &)) &ov_msckf::Propagator::feed_imu, "Stores incoming inertial readings\n \n\n Contains our timestamp and inertial information\n\nC++: ov_msckf::Propagator::feed_imu(const struct ov_core::ImuData &) --> void", pybind11::arg("message"));
    cl.def("propagate_and_clone", (void (ov_msckf::Propagator::*)(class std::shared_ptr<class ov_msckf::State>, double)) &ov_msckf::Propagator::propagate_and_clone, "Propagate state up to given timestamp and then clone\n\n This will first collect all imu readings that occured between the\n *current* state time and the new time we want the state to be at.\n If we don't have any imu readings we will try to extrapolate into the future.\n After propagating the mean and covariance using our dynamics,\n We clone the current imu pose as a new clone in our state.\n\n \n Pointer to state\n \n\n Time to propagate to and clone at\n\nC++: ov_msckf::Propagator::propagate_and_clone(class std::shared_ptr<class ov_msckf::State>, double) --> void", pybind11::arg("state"), pybind11::arg("timestamp"));
    cl.def_static("select_imu_readings", [](const class std::vector<struct ov_core::ImuData, class std::allocator<struct ov_core::ImuData> > & a0, double const & a1, double const & a2) -> std::vector<struct ov_core::ImuData, class std::allocator<struct ov_core::ImuData> > { return ov_msckf::Propagator::select_imu_readings(a0, a1, a2); }, "", pybind11::arg("imu_data"), pybind11::arg("time0"), pybind11::arg("time1"));
    cl.def_static("select_imu_readings", (class std::vector<struct ov_core::ImuData, class std::allocator<struct ov_core::ImuData> > (*)(const class std::vector<struct ov_core::ImuData, class std::allocator<struct ov_core::ImuData> > &, double, double, bool)) &ov_msckf::Propagator::select_imu_readings, "Helper function that given current imu data, will select imu readings between the two times.\n\n This will create measurements that we will integrate with, and an extra measurement at the end.\n We use the \n\n\n The timestamps passed should already take into account the time offset values.\n\n \n IMU data we will select measurements from\n \n\n Start timestamp\n \n\n End timestamp\n \n\n If we should warn if we don't have enough IMU to propagate with (e.g. fast prop will get warnings otherwise)\n \n\n Vector of measurements (if we could compute them)\n\nC++: ov_msckf::Propagator::select_imu_readings(const class std::vector<struct ov_core::ImuData, class std::allocator<struct ov_core::ImuData> > &, double, double, bool) --> class std::vector<struct ov_core::ImuData, class std::allocator<struct ov_core::ImuData> >", pybind11::arg("imu_data"), pybind11::arg("time0"), pybind11::arg("time1"), pybind11::arg("warn"));
    cl.def_static("interpolate_data", (struct ov_core::ImuData (*)(const struct ov_core::ImuData &, const struct ov_core::ImuData &, double)) &ov_msckf::Propagator::interpolate_data, "Nice helper function that will linearly interpolate between two imu messages.\n\n This should be used instead of just \"cutting\" imu messages that bound the camera times\n Give better time offset if we use this function, could try other orders/splines if the imu is slow.\n\n \n imu at begining of interpolation interval\n \n\n imu at end of interpolation interval\n \n\n Timestamp being interpolated to\n\nC++: ov_msckf::Propagator::interpolate_data(const struct ov_core::ImuData &, const struct ov_core::ImuData &, double) --> struct ov_core::ImuData", pybind11::arg("imu_1"), pybind11::arg("imu_2"), pybind11::arg("timestamp"));

    { // ov_msckf::Propagator::NoiseManager file: line:46
      auto & enclosing_class = cl;
      pybind11::class_<ov_msckf::Propagator::NoiseManager, std::shared_ptr<ov_msckf::Propagator::NoiseManager>> cl(enclosing_class, "NoiseManager", "Struct of our imu noise parameters");
      cl.def( pybind11::init( [](ov_msckf::Propagator::NoiseManager const &o){ return new ov_msckf::Propagator::NoiseManager(o); } ) );
      cl.def( pybind11::init( [](){ return new ov_msckf::Propagator::NoiseManager(); } ) );
      cl.def_readwrite("sigma_w", &ov_msckf::Propagator::NoiseManager::sigma_w);
      cl.def_readwrite("sigma_w_2", &ov_msckf::Propagator::NoiseManager::sigma_w_2);
      cl.def_readwrite("sigma_wb", &ov_msckf::Propagator::NoiseManager::sigma_wb);
      cl.def_readwrite("sigma_wb_2", &ov_msckf::Propagator::NoiseManager::sigma_wb_2);
      cl.def_readwrite("sigma_a", &ov_msckf::Propagator::NoiseManager::sigma_a);
      cl.def_readwrite("sigma_a_2", &ov_msckf::Propagator::NoiseManager::sigma_a_2);
      cl.def_readwrite("sigma_ab", &ov_msckf::Propagator::NoiseManager::sigma_ab);
      cl.def_readwrite("sigma_ab_2", &ov_msckf::Propagator::NoiseManager::sigma_ab_2);
      cl.def("print", (void (ov_msckf::Propagator::NoiseManager::*)()) &ov_msckf::Propagator::NoiseManager::print, "Nice print function of what parameters we have loaded\n\nC++: ov_msckf::Propagator::NoiseManager::print() --> void");
      cl.def("assign", (struct ov_msckf::Propagator::NoiseManager & (ov_msckf::Propagator::NoiseManager::*)(const struct ov_msckf::Propagator::NoiseManager &)) &ov_msckf::Propagator::NoiseManager::operator=, "C++: ov_msckf::Propagator::NoiseManager::operator=(const struct ov_msckf::Propagator::NoiseManager &) --> struct ov_msckf::Propagator::NoiseManager &", pybind11::return_value_policy::automatic, pybind11::arg(""));
    }

  }
  { // ov_msckf::UpdaterOptions file: line:30
    pybind11::class_<ov_msckf::UpdaterOptions, std::shared_ptr<ov_msckf::UpdaterOptions>> cl(M, "UpdaterOptions", "Struct which stores general updater options");
    cl.def( pybind11::init( [](ov_msckf::UpdaterOptions const &o){ return new ov_msckf::UpdaterOptions(o); } ) );
    cl.def( pybind11::init( [](){ return new ov_msckf::UpdaterOptions(); } ) );
    cl.def_readwrite("chi2_multipler", &ov_msckf::UpdaterOptions::chi2_multipler);
    cl.def_readwrite("sigma_pix", &ov_msckf::UpdaterOptions::sigma_pix);
    cl.def_readwrite("sigma_pix_sq", &ov_msckf::UpdaterOptions::sigma_pix_sq);
    cl.def("print", (void (ov_msckf::UpdaterOptions::*)()) &ov_msckf::UpdaterOptions::print, "Nice print function of what parameters we have loaded\n\nC++: ov_msckf::UpdaterOptions::print() --> void");
    cl.def("assign", (struct ov_msckf::UpdaterOptions & (ov_msckf::UpdaterOptions::*)(const struct ov_msckf::UpdaterOptions &)) &ov_msckf::UpdaterOptions::operator=, "C++: ov_msckf::UpdaterOptions::operator=(const struct ov_msckf::UpdaterOptions &) --> struct ov_msckf::UpdaterOptions &", pybind11::return_value_policy::automatic, pybind11::arg(""));
  }

  {
    pybind11::class_<ov_init::InertialInitializerOptions, std::shared_ptr<ov_init::InertialInitializerOptions>> cl(M, "InertialInitializerOptions", "Struct which stores all options needed for state estimation.");
    cl.def_readwrite("init_window_time",&ov_init::InertialInitializerOptions::init_window_time);
    cl.def_readwrite("init_imu_thresh",&ov_init::InertialInitializerOptions::init_imu_thresh);
    cl.def_readwrite("init_max_disparity",&ov_init::InertialInitializerOptions::init_max_disparity);
    cl.def_readwrite("gravity_mag",&ov_init::InertialInitializerOptions::gravity_mag);
    cl.def_readwrite("num_cameras",&ov_init::InertialInitializerOptions::num_cameras);
  }

  {
    pybind11::class_<ov_core::YamlParser, std::shared_ptr<ov_core::YamlParser>> cl(
        M, "YamlParser",
        "Helper class to do OpenCV yaml parsing from both file and ROS.");
    cl.def( pybind11::init<const std::string& , bool>() );
    cl.def("get_config_folder",&ov_core::YamlParser::get_config_folder);
    cl.def("successful",&ov_core::YamlParser::successful);

  }

  { // ov_msckf::VioManagerOptions file:ov_msckf/src/core/VioManagerOptions.h line:50
    pybind11::class_<ov_msckf::VioManagerOptions, std::shared_ptr<ov_msckf::VioManagerOptions>> cl(
        M, "VioManagerOptions",
        "Struct which stores all options needed for state estimation.\n\n This is broken into a few different parts: estimator, trackers, "
        "and simulation.\n If you are going to add a parameter here you will need to add it to the parsers.\n You will also need to add it "
        "to the print statement at the bottom of each.");
    cl.def(pybind11::init([](ov_msckf::VioManagerOptions const &o) { return new ov_msckf::VioManagerOptions(o); }));
    cl.def(pybind11::init([]() { return new ov_msckf::VioManagerOptions(); }));
    cl.def("print_and_load",&ov_msckf::VioManagerOptions::print_and_load);
    cl.def_readwrite("state_options", &ov_msckf::VioManagerOptions::state_options);
    cl.def_readwrite("dt_slam_delay", &ov_msckf::VioManagerOptions::dt_slam_delay);
    cl.def_readwrite("try_zupt", &ov_msckf::VioManagerOptions::try_zupt);
    cl.def_readwrite("zupt_max_velocity", &ov_msckf::VioManagerOptions::zupt_max_velocity);
    cl.def_readwrite("zupt_noise_multiplier", &ov_msckf::VioManagerOptions::zupt_noise_multiplier);
    cl.def_readwrite("zupt_max_disparity", &ov_msckf::VioManagerOptions::zupt_max_disparity);
    cl.def_readwrite("zupt_only_at_beginning", &ov_msckf::VioManagerOptions::zupt_only_at_beginning);
    cl.def_readwrite("record_timing_information", &ov_msckf::VioManagerOptions::record_timing_information);
    cl.def_readwrite("record_timing_filepath", &ov_msckf::VioManagerOptions::record_timing_filepath);
    cl.def_readwrite("imu_noises", &ov_msckf::VioManagerOptions::imu_noises);
    cl.def_readwrite("msckf_options", &ov_msckf::VioManagerOptions::msckf_options);
    cl.def_readwrite("slam_options", &ov_msckf::VioManagerOptions::slam_options);
    cl.def_readwrite("aruco_options", &ov_msckf::VioManagerOptions::aruco_options);
    cl.def_readwrite("zupt_options", &ov_msckf::VioManagerOptions::zupt_options);
    cl.def_readwrite("gravity_mag", &ov_msckf::VioManagerOptions::gravity_mag);
    cl.def_readwrite("calib_camimu_dt", &ov_msckf::VioManagerOptions::calib_camimu_dt);
//    cl.def_readwrite("camera_fisheye", &ov_msckf::VioManagerOptions::camera_fisheye);
    cl.def_readwrite("camera_intrinsics", &ov_msckf::VioManagerOptions::camera_intrinsics);
    cl.def_readwrite("camera_extrinsics", &ov_msckf::VioManagerOptions::camera_extrinsics);
//    cl.def_readwrite("camera_wh", &ov_msckf::VioManagerOptions::camera_wh);
    cl.def_readwrite("use_stereo", &ov_msckf::VioManagerOptions::use_stereo);
    cl.def_readwrite("use_klt", &ov_msckf::VioManagerOptions::use_klt);
    cl.def_readwrite("use_aruco", &ov_msckf::VioManagerOptions::use_aruco);
    cl.def_readwrite("downsize_aruco", &ov_msckf::VioManagerOptions::downsize_aruco);
    cl.def_readwrite("downsample_cameras", &ov_msckf::VioManagerOptions::downsample_cameras);
    cl.def_readwrite("use_multi_threading", &ov_msckf::VioManagerOptions::use_multi_threading);
    cl.def_readwrite("num_pts", &ov_msckf::VioManagerOptions::num_pts);
    cl.def_readwrite("fast_threshold", &ov_msckf::VioManagerOptions::fast_threshold);
    cl.def_readwrite("grid_x", &ov_msckf::VioManagerOptions::grid_x);
    cl.def_readwrite("grid_y", &ov_msckf::VioManagerOptions::grid_y);
    cl.def_readwrite("min_px_dist", &ov_msckf::VioManagerOptions::min_px_dist);
    cl.def_readwrite("histogram_method", &ov_msckf::VioManagerOptions::histogram_method);
    cl.def_readwrite("knn_ratio", &ov_msckf::VioManagerOptions::knn_ratio);
    cl.def_readwrite("use_mask", &ov_msckf::VioManagerOptions::use_mask);
    cl.def_readwrite("masks", &ov_msckf::VioManagerOptions::masks);
    cl.def_readwrite("featinit_options", &ov_msckf::VioManagerOptions::featinit_options);
    cl.def_readwrite("sim_traj_path", &ov_msckf::VioManagerOptions::sim_traj_path);
    cl.def_readwrite("sim_distance_threshold", &ov_msckf::VioManagerOptions::sim_distance_threshold);
    cl.def_readwrite("sim_freq_cam", &ov_msckf::VioManagerOptions::sim_freq_cam);
    cl.def_readwrite("sim_freq_imu", &ov_msckf::VioManagerOptions::sim_freq_imu);
    cl.def_readwrite("sim_seed_state_init", &ov_msckf::VioManagerOptions::sim_seed_state_init);
    cl.def_readwrite("sim_seed_preturb", &ov_msckf::VioManagerOptions::sim_seed_preturb);
    cl.def_readwrite("sim_seed_measurements", &ov_msckf::VioManagerOptions::sim_seed_measurements);
    cl.def_readwrite("sim_do_perturbation", &ov_msckf::VioManagerOptions::sim_do_perturbation);
    cl.def_readwrite("init_options",&ov_msckf::VioManagerOptions::init_options);
    cl.def("print_and_load_estimator",&ov_msckf::VioManagerOptions::print_and_load_estimator);
    cl.def("print_and_load_noise",&ov_msckf::VioManagerOptions::print_and_load_noise);
    cl.def("print_and_load_state",&ov_msckf::VioManagerOptions::print_and_load_state);
    cl.def("print_and_load_trackers",&ov_msckf::VioManagerOptions::print_and_load_trackers);
    cl.def("print_and_load_simulation",&ov_msckf::VioManagerOptions::print_and_load_simulation);
    cl.def(
        "assign",
        (struct ov_msckf::VioManagerOptions & (ov_msckf::VioManagerOptions::*)(const struct ov_msckf::VioManagerOptions &)) &
            ov_msckf::VioManagerOptions::operator=,
        "C++: ov_msckf::VioManagerOptions::operator=(const struct ov_msckf::VioManagerOptions &) --> struct ov_msckf::VioManagerOptions &",
        pybind11::return_value_policy::automatic, pybind11::arg(""));
  }

  { // ov_msckf::State file: line:50
    pybind11::class_<ov_msckf::State, std::shared_ptr<ov_msckf::State>> cl(
        M, "State",
        "State of our filter\n\n This state has all the current estimates for the filter.\n This system is modeled after the MSCKF filter, "
        "thus we have a sliding window of clones.\n We additionally have more parameters for online estimation of calibration and SLAM "
        "features.\n We also have the covariance of the system, which should be managed using the StateHelper class.");
    cl.def(pybind11::init<struct ov_msckf::StateOptions &>(), pybind11::arg("options_"));

    cl.def(pybind11::init([](ov_msckf::State const &o) { return new ov_msckf::State(o); }));
    cl.def_readwrite("timestamp", &ov_msckf::State::_timestamp);
    cl.def_readwrite("options", &ov_msckf::State::_options);
    cl.def_readwrite("imu", &ov_msckf::State::_imu);
    cl.def_readwrite("clones_IMU", &ov_msckf::State::_clones_IMU);
    cl.def_readwrite("features_SLAM", &ov_msckf::State::_features_SLAM);
    cl.def_readwrite("calib_dt_CAMtoIMU", &ov_msckf::State::_calib_dt_CAMtoIMU);
    cl.def_readwrite("calib_IMUtoCAM", &ov_msckf::State::_calib_IMUtoCAM);
    cl.def_readwrite("cam_intrinsics", &ov_msckf::State::_cam_intrinsics);
    cl.def_readwrite("cam_intrinsics_cameras", &ov_msckf::State::_cam_intrinsics_cameras);
    cl.def("get_pose_covariance",[](std::shared_ptr<ov_msckf::State>& myself)
    {
      std::vector<std::shared_ptr<ov_type::Type>> statevars;
      statevars.push_back(myself->_imu->pose()->p());
      statevars.push_back(myself->_imu->pose()->q());
      Eigen::Matrix<double, 6, 6> covariance_posori = ov_msckf::StateHelper::get_marginal_covariance(myself, statevars);
      for (int r = 0; r < 6; r++) {
        for (int c = 0; c < 6; c++) {
          covariance_posori(r,c) = covariance_posori(r, c);
        }
      }
      return covariance_posori;
    });
    cl.def("margtimestep", (double (ov_msckf::State::*)()) & ov_msckf::State::margtimestep,
           "Will return the timestep that we will marginalize next.\n As of right now, since we are using a sliding window, this is the "
           "oldest clone.\n But if you wanted to do a keyframe system, you could selectively marginalize clones.\n \n\n timestep of clone "
           "we will marginalize\n\nC++: ov_msckf::State::margtimestep() --> double");
    cl.def("max_covariance_size", (int (ov_msckf::State::*)()) & ov_msckf::State::max_covariance_size,
           "Calculates the current max size of the covariance\n \n\n Size of the current covariance matrix\n\nC++: "
           "ov_msckf::State::max_covariance_size() --> int");
    cl.def("assign", (class ov_msckf::State & (ov_msckf::State::*)(const class ov_msckf::State &)) & ov_msckf::State::operator=,
           "C++: ov_msckf::State::operator=(const class ov_msckf::State &) --> class ov_msckf::State &",
           pybind11::return_value_policy::automatic, pybind11::arg(""));
  }
  {

    pybind11::class_<ov_msckf::VioManager, std::shared_ptr<ov_msckf::VioManager>> cl(
        M, "VioManager",
        "Core class that manages the entire system\n\n This class contains the state and other algorithms needed for the MSCKF to work.\n "
        "We feed in measurements into this class and send them to their respective algorithms.\n If we have measurements to propagate or "
        "update with, this class will call on our state to do that.");
    cl.def(pybind11::init<struct ov_msckf::VioManagerOptions &>(), pybind11::arg("params_"));

    cl.def("feed_measurement_imu",
           (void (ov_msckf::VioManager::*)(const struct ov_core::ImuData &)) & ov_msckf::VioManager::feed_measurement_imu,py::call_guard<py::gil_scoped_release>(),
           "Feed function for inertial data\n \n\n Contains our timestamp and inertial information\n\nC++: "
           "ov_msckf::VioManager::feed_measurement_imu(const struct ov_core::ImuData &) --> void",
           pybind11::arg("message"));
    cl.def("feed_measurement_camera",
           (void (ov_msckf::VioManager::*)(const struct ov_core::CameraData &)) & ov_msckf::VioManager::feed_measurement_camera,py::call_guard<py::gil_scoped_release>(),
           "Feed function for camera measurements\n \n\n Contains our timestamp, images, and camera ids\n\nC++: "
           "ov_msckf::VioManager::feed_measurement_camera(const struct ov_core::CameraData &) --> void",
           pybind11::arg("message"));
    cl.def("initialize_with_gt",&ov_msckf::VioManager::initialize_with_gt);
    cl.def("initialized", (bool (ov_msckf::VioManager::*)()) & ov_msckf::VioManager::initialized,
           "If we are initialized or not\n\nC++: ov_msckf::VioManager::initialized() --> bool");
    cl.def("initialized_time", (double (ov_msckf::VioManager::*)()) & ov_msckf::VioManager::initialized_time,
           "Timestamp that the system was initialized at\n\nC++: ov_msckf::VioManager::initialized_time() --> double");
    cl.def("get_params", (struct ov_msckf::VioManagerOptions(ov_msckf::VioManager::*)()) & ov_msckf::VioManager::get_params,
           "Accessor for current system parameters\n\nC++: ov_msckf::VioManager::get_params() --> struct ov_msckf::VioManagerOptions");
    cl.def("get_state", (class std::shared_ptr<class ov_msckf::State>(ov_msckf::VioManager::*)()) & ov_msckf::VioManager::get_state,
           "Accessor to get the current state\n\nC++: ov_msckf::VioManager::get_state() --> class std::shared_ptr<class ov_msckf::State>");
    cl.def("get_propagator",
           (class std::shared_ptr<class ov_msckf::Propagator>(ov_msckf::VioManager::*)()) & ov_msckf::VioManager::get_propagator,
           "Accessor to get the current propagator\n\nC++: ov_msckf::VioManager::get_propagator() --> class std::shared_ptr<class "
           "ov_msckf::Propagator>");
//    cl.def_readwrite("trackFEATS",&ov_msckf::VioManager::trackFEATS);
//    cl.def_readwrite("trackDATABASE",&ov_msckf::VioManager::trackDATABASE);
    cl.def("set_feature_tracker",&ov_msckf::VioManager::set_feature_tracker);
  }

  { // ov_eval::AlignTrajectory file: line:44
    M.def("align_trajectory",[](const std::vector<Eigen::Matrix<double, 7, 1>> &traj_es,
                                 const std::vector<Eigen::Matrix<double, 7, 1>> &traj_gt,
                                 std::string method, int n_aligned){
      Eigen::Matrix3d output_R;
      Eigen::Vector3d output_t;
      double output_s;

      ov_eval::AlignTrajectory::align_trajectory(traj_es,traj_gt,output_R,output_t,output_s,method,n_aligned);
      return py::make_tuple(output_R,output_t,output_s);
    });
  }
  { // ov_eval::ResultTrajectory file:ov_eval/src/calc/ResultTrajectory.h line:57
    pybind11::class_<ov_eval::ResultTrajectory, std::shared_ptr<ov_eval::ResultTrajectory>> cl(M, "ResultTrajectory", "A single run for a given dataset.\n\n This class has all the error function which can be calculated for the loaded trajectory.\n Given a groundtruth and trajectory we first align the two so that they are in the same frame.\n From there the following errors could be computed:\n - Absolute trajectory error\n - Relative pose Error\n - Normalized estimation error squared\n - Error and bound at each timestep\n\n Please see the \n\n Visual(-Inertial) Odometry](http://rpg.ifi.uzh.ch/docs/IROS18_Zhang.pdf) paper for implementation specific details.");
    cl.def( pybind11::init<std::string, std::string, std::string>(), pybind11::arg("path_est"), pybind11::arg("path_gt"), pybind11::arg("alignment_method") );

    cl.def( pybind11::init( [](ov_eval::ResultTrajectory const &o){ return new ov_eval::ResultTrajectory(o); } ) );
    cl.def("calculate_ate", (void (ov_eval::ResultTrajectory::*)(struct ov_eval::Statistics &, struct ov_eval::Statistics &)) &ov_eval::ResultTrajectory::calculate_ate, "Computes the Absolute Trajectory Error (ATE) for this trajectory.\n\n This will first do our alignment of the two trajectories.\n Then at each point the error will be calculated and normed as follows:\n \n\n\n\n \n Error values for the orientation\n \n\n Error values for the position\n\nC++: ov_eval::ResultTrajectory::calculate_ate(struct ov_eval::Statistics &, struct ov_eval::Statistics &) --> void", pybind11::arg("error_ori"), pybind11::arg("error_pos"));
    cl.def("calculate_ate_2d", (void (ov_eval::ResultTrajectory::*)(struct ov_eval::Statistics &, struct ov_eval::Statistics &)) &ov_eval::ResultTrajectory::calculate_ate_2d, "Computes the Absolute Trajectory Error (ATE) for this trajectory in the 2d x-y plane.\n\n This will first do our alignment of the two trajectories.\n We just grab the yaw component of the orientation and the xy plane error.\n Then at each point the error will be calculated and normed as follows:\n \n\n\n\n \n Error values for the orientation (yaw error)\n \n\n Error values for the position (xy error)\n\nC++: ov_eval::ResultTrajectory::calculate_ate_2d(struct ov_eval::Statistics &, struct ov_eval::Statistics &) --> void", pybind11::arg("error_ori"), pybind11::arg("error_pos"));
    cl.def("calculate_nees", (void (ov_eval::ResultTrajectory::*)(struct ov_eval::Statistics &, struct ov_eval::Statistics &)) &ov_eval::ResultTrajectory::calculate_nees, "Computes the Normalized Estimation Error Squared (NEES) for this trajectory.\n\n If we have a covariance in addition to our pose estimate we can compute the NEES values.\n At each timestep we compute this for both orientation and position.\n \n\n\n\n\n \n NEES values for the orientation\n \n\n NEES values for the position\n\nC++: ov_eval::ResultTrajectory::calculate_nees(struct ov_eval::Statistics &, struct ov_eval::Statistics &) --> void", pybind11::arg("nees_ori"), pybind11::arg("nees_pos"));
    cl.def("calculate_error", (void (ov_eval::ResultTrajectory::*)(struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &)) &ov_eval::ResultTrajectory::calculate_error, "Computes the error at each timestamp for this trajectory.\n\n As compared to ATE error (see \n\n This is normally used if you just want to look at a single run on a single dataset.\n \n\n\n\n \n Position x-axis error and bound if we have it in our file\n \n\n Position y-axis error and bound if we have it in our file\n \n\n Position z-axis error and bound if we have it in our file\n \n\n Orientation x-axis error and bound if we have it in our file\n \n\n Orientation y-axis error and bound if we have it in our file\n \n\n Orientation z-axis error and bound if we have it in our file\n \n\n Orientation roll error and bound if we have it in our file\n \n\n Orientation pitch error and bound if we have it in our file\n \n\n Orientation yaw error and bound if we have it in our file\n\nC++: ov_eval::ResultTrajectory::calculate_error(struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &, struct ov_eval::Statistics &) --> void", pybind11::arg("posx"), pybind11::arg("posy"), pybind11::arg("posz"), pybind11::arg("orix"), pybind11::arg("oriy"), pybind11::arg("oriz"), pybind11::arg("roll"), pybind11::arg("pitch"), pybind11::arg("yaw"));
  }

  { // ov_eval::AlignUtils file:ov_eval/src/alignment/AlignUtils.h line:44
    M.def("get_best_yaw",&ov_eval::AlignUtils::get_best_yaw);
    M.def("get_mean",&ov_eval::AlignUtils::get_mean);
    M.def("align_umeyama",[](const std::vector<Eigen::Matrix<double, 3, 1>> &data, const std::vector<Eigen::Matrix<double, 3, 1>> &model,bool known_scale, bool yaw_only){
      Eigen::Matrix3d output_R;
      Eigen::Vector3d output_t;
      double output_s;

      ov_eval::AlignUtils::align_umeyama(data,model,output_R,output_t,output_s,known_scale,yaw_only);
      return py::make_tuple(output_R,output_t,output_s);
    });

    M.def("align_umeyama",[](const std::vector<Eigen::Matrix<double, 3, 1>> &data, const std::vector<Eigen::Matrix<double, 3, 1>> &model,bool known_scale, bool yaw_only){
      Eigen::Matrix3d output_R;
      Eigen::Vector3d output_t;
      double output_s;

      ov_eval::AlignUtils::align_umeyama(data,model,output_R,output_t,output_s,known_scale,yaw_only);
      return py::make_tuple(output_R,output_t,output_s);
    });

  }
  { // ov_eval::Statistics file:ov_eval/src/utils/Statistics.h line:38
    pybind11::class_<ov_eval::Statistics, std::shared_ptr<ov_eval::Statistics>> cl(M, "Statistics", "Statistics object for a given set scalar time series values.\n\n Ensure that you call the calculate() function to update the values before using them.\n This will compute all the final results from the values in \n\n\n ");
    cl.def( pybind11::init( [](){ return new ov_eval::Statistics(); } ) );
    cl.def( pybind11::init( [](ov_eval::Statistics const &o){ return new ov_eval::Statistics(o); } ) );
    cl.def_readwrite("rmse", &ov_eval::Statistics::rmse);
    cl.def_readwrite("mean", &ov_eval::Statistics::mean);
    cl.def_readwrite("median", &ov_eval::Statistics::median);
    cl.def_readwrite("std", &ov_eval::Statistics::std);
    cl.def_readwrite("max", &ov_eval::Statistics::max);
    cl.def_readwrite("min", &ov_eval::Statistics::min);
    cl.def_readwrite("ninetynine", &ov_eval::Statistics::ninetynine);
    cl.def_readwrite("timestamps", &ov_eval::Statistics::timestamps);
    cl.def_readwrite("values", &ov_eval::Statistics::values);
    cl.def_readwrite("values_bound", &ov_eval::Statistics::values_bound);
    cl.def("calculate", (void (ov_eval::Statistics::*)()) &ov_eval::Statistics::calculate, "Will calculate all values from our vectors of information\n\nC++: ov_eval::Statistics::calculate() --> void");
    cl.def("clear", (void (ov_eval::Statistics::*)()) &ov_eval::Statistics::clear, "Will clear any old values\n\nC++: ov_eval::Statistics::clear() --> void");
  }
  { // ov_eval::Loader file:ov_eval/src/utils/Loader.h line:39
    M.def("load_data",&ov_eval::Loader::load_data);
    M.def("load_data_csv",&ov_eval::Loader::load_data_csv);
    M.def("load_simulation",&ov_eval::Loader::load_simulation);
    M.def("load_timing_flamegraph",&ov_eval::Loader::load_timing_flamegraph);
    M.def("load_timing_percent",&ov_eval::Loader::load_timing_percent);
    M.def("get_total_length",&ov_eval::Loader::get_total_length);
  }

  {
    M.def("virtual_test",[](CameraData cam_data,std::shared_ptr<TrackBase>& base){
      base->feed_new_camera(cam_data);
    });
  }

}
