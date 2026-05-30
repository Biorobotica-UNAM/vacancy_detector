#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

using namespace std::chrono_literals;

class ShelfDropTfNode : public rclcpp::Node {
public:
  ShelfDropTfNode() : Node("shelf_drop_tf_node") {
    declare_parameter<std::string>("pointcloud_topic", "/hardware/camera/points");
    declare_parameter<std::string>("fixed_frame_id", "map");
    declare_parameter<std::string>("validation_frame_id", "base_link");
    declare_parameter<std::string>("shelf_frame_id", "shelf_frame");
    declare_parameter<std::string>("drop_start_frame_id", "shelf_drop_start");
    declare_parameter<std::string>("drop_points_file",
                                   "src/Vision/vacancy_detector/src/"
                                   "shelf_drop_points.txt");

    declare_parameter<double>("roi_x_min_m", 0.20);
    declare_parameter<double>("roi_x_max_m", 1.60);
    declare_parameter<double>("roi_y_min_m", -0.85);
    declare_parameter<double>("roi_y_max_m", 0.85);
    declare_parameter<double>("roi_z_min_m", 0.25);
    declare_parameter<double>("roi_z_max_m", 1.75);
    declare_parameter<double>("plane_distance_threshold_m", 0.025);
    declare_parameter<double>("plane_eps_angle_deg", 20.0);
    declare_parameter<int>("min_plane_points", 250);
    declare_parameter<int>("max_horizontal_planes", 8);
    declare_parameter<double>("min_plane_separation_m", 0.10);

    declare_parameter<double>("surface_min_width_m", 0.25);
    declare_parameter<double>("surface_min_depth_m", 0.18);
    declare_parameter<double>("min_section_height_m", 0.18);
    declare_parameter<double>("open_section_height_m", 0.45);
    declare_parameter<double>("preferred_section_z_m", -1.0);
    declare_parameter<double>("preferred_section_z_weight", 1.0);

    declare_parameter<double>("start_height_m", 0.10);
    declare_parameter<double>("front_margin_m", 0.07);
    declare_parameter<double>("side_margin_m", 0.07);
    declare_parameter<double>("back_margin_m", 0.08);
    declare_parameter<double>("ceiling_margin_m", 0.10);
    declare_parameter<double>("obstacle_margin_m", 0.12);
    declare_parameter<double>("obstacle_min_height_m", 0.035);
    declare_parameter<double>("obstacle_max_height_m", 0.40);
    declare_parameter<double>("preferred_y_m", 0.0);

    declare_parameter<double>("safe_x_min_m", 0.15);
    declare_parameter<double>("safe_x_max_m", 1.35);
    declare_parameter<double>("safe_abs_y_max_m", 0.85);
    declare_parameter<double>("safe_z_min_m", 0.30);
    declare_parameter<double>("safe_z_max_m", 1.70);
    declare_parameter<double>("safe_dist_xy_max_m", 1.55);
    declare_parameter<bool>("active_on_start", false);

    pointcloud_topic_ = get_parameter("pointcloud_topic").as_string();
    fixed_frame_id_ = get_parameter("fixed_frame_id").as_string();
    validation_frame_id_ = get_parameter("validation_frame_id").as_string();
    shelf_frame_id_ = get_parameter("shelf_frame_id").as_string();
    drop_start_frame_id_ = get_parameter("drop_start_frame_id").as_string();
    drop_points_file_ = get_parameter("drop_points_file").as_string();
    active_ = get_parameter("active_on_start").as_bool();

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    static_tf_broadcaster_ =
        std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&ShelfDropTfNode::cloudCallback, this, std::placeholders::_1));

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "camera/drop_markers",
        rclcpp::QoS(10).transient_local().reliable());

    resetDropPointsFile();

    enable_srv_ = create_service<std_srvs::srv::SetBool>(
        "shelf_drop_tf/enable",
        std::bind(&ShelfDropTfNode::enableCallback, this,
                  std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(),
                "[shelf_drop_tf] Nodo iniciado. pointcloud_topic=%s "
                "fixed_frame=%s validation_frame=%s active=%s",
                pointcloud_topic_.c_str(), fixed_frame_id_.c_str(),
                validation_frame_id_.c_str(), active_ ? "true" : "false");
  }

private:
  struct PlaneSurface {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud{
        new pcl::PointCloud<pcl::PointXYZ>};
    Eigen::Vector3f normal{0.0f, 0.0f, 1.0f};
    Eigen::Vector3f centroid{0.0f, 0.0f, 0.0f};
    float x_min{0.0f};
    float x_max{0.0f};
    float y_min{0.0f};
    float y_max{0.0f};
    float z{0.0f};
    int inliers{0};
  };

  struct Section {
    PlaneSurface floor;
    float ceiling_z{0.0f};
    float width{0.0f};
    float depth{0.0f};
    float height{0.0f};
    float score{-std::numeric_limits<float>::max()};
  };

  struct Candidate {
    Eigen::Vector3f surface_point{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f start_point{0.0f, 0.0f, 0.0f};
    float score{-std::numeric_limits<float>::max()};
  };

  struct SavedDropPoint {
    int id{0};
    Eigen::Vector3f point{0.0f, 0.0f, 0.0f};
  };

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!active_) {
      RCLCPP_DEBUG_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "[shelf_drop_tf] inactivo; ignorando nubes hasta enable=true");
      return;
    }

    if (drop_point_saved_this_activation_) {
      RCLCPP_DEBUG_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "[shelf_drop_tf] punto ya guardado en esta activacion");
      return;
    }

    std::string error;
    if (!processCloud(*msg, error)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "[shelf_drop_tf] No se pudo calcular TF: %s",
                           error.c_str());
    }
  }

  void enableCallback(
      const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
      std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
    active_ = request->data;
    if (active_) {
      drop_point_saved_this_activation_ = false;
      publishMarkers();
    }

    response->success = true;
    response->message =
        active_ ? "shelf_drop_tf_enabled" : "shelf_drop_tf_disabled";
    RCLCPP_INFO(get_logger(), "[shelf_drop_tf] %s",
                response->message.c_str());
  }

  bool processCloud(const sensor_msgs::msg::PointCloud2 &input_msg,
                    std::string &error_out) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(input_msg, *cloud);
    if (cloud->empty()) {
      error_out = "empty_cloud";
      return false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr validation_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (!transformCloudToValidation(input_msg.header.frame_id, cloud,
                                    validation_cloud, error_out)) {
      return false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    applyRoi(validation_cloud, roi_cloud);

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "[shelf_drop_tf] ROI puntos: entrada=%zu validation=%zu roi=%zu",
        cloud->size(), validation_cloud->size(), roi_cloud->size());

    if (roi_cloud->size() < 500) {
      error_out = "insufficient_points_after_roi count=" +
                  std::to_string(roi_cloud->size());
      return false;
    }

    std::vector<PlaneSurface> planes;
    detectHorizontalPlanes(roi_cloud, planes);
    if (planes.empty()) {
      error_out = "no_horizontal_shelf_planes";
      return false;
    }

    Section section;
    if (!selectNearestSection(planes, section, error_out)) {
      return false;
    }

    std::vector<Eigen::Vector2f> obstacles =
        obstacleProjections(roi_cloud, section);

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "[shelf_drop_tf] seccion elegida: z=%.3f front=%.3f "
        "width=%.3f depth=%.3f height=%.3f obstaculos=%zu score=%.2f",
        section.floor.z, section.floor.x_min, section.width, section.depth,
        section.height, obstacles.size(), section.score);

    Candidate candidate;
    if (!selectCandidate(section, obstacles, candidate, error_out)) {
      return false;
    }

    if (!validateDropStartPose(candidate.start_point, error_out)) {
      return false;
    }

    Eigen::Vector3f fixed_origin;
    Eigen::Vector3f fixed_start_point;
    Eigen::Vector3f fixed_axis_x;
    Eigen::Vector3f fixed_axis_y;
    Eigen::Vector3f fixed_axis_z;
    if (!transformGeometryToFixed(section, candidate, fixed_origin,
                                  fixed_start_point, fixed_axis_x,
                                  fixed_axis_y, fixed_axis_z, error_out)) {
      return false;
    }

    publishTransforms(section, candidate, fixed_origin, fixed_axis_x,
                      fixed_axis_y, fixed_axis_z);
    saveDropPoint(fixed_start_point);
    publishMarkers();
    drop_point_saved_this_activation_ = true;

    RCLCPP_INFO(
        get_logger(),
        "[shelf_drop_tf] TF %s publicada: base=(%.3f %.3f %.3f) "
        "surface=(%.3f %.3f %.3f) score=%.2f",
        drop_start_frame_id_.c_str(), candidate.start_point.x(),
        candidate.start_point.y(), candidate.start_point.z(),
        candidate.surface_point.x(), candidate.surface_point.y(),
        candidate.surface_point.z(), candidate.score);

    error_out.clear();
    return true;
  }

  bool transformCloudToValidation(
      const std::string &source_frame,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
      pcl::PointCloud<pcl::PointXYZ>::Ptr &validation_cloud,
      std::string &error_out) {
    if (source_frame.empty()) {
      error_out = "input_cloud_frame_empty";
      return false;
    }
    if (validation_frame_id_.empty()) {
      error_out = "validation_frame_id_empty";
      return false;
    }

    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
    Eigen::Vector3f translation(0.0f, 0.0f, 0.0f);
    if (source_frame != validation_frame_id_) {
      try {
        const auto tf = tf_buffer_->lookupTransform(
            validation_frame_id_, source_frame, tf2::TimePointZero, 300ms);
        Eigen::Quaternionf q(tf.transform.rotation.w, tf.transform.rotation.x,
                             tf.transform.rotation.y, tf.transform.rotation.z);
        q.normalize();
        rotation = q.toRotationMatrix();
        translation = Eigen::Vector3f(tf.transform.translation.x,
                                      tf.transform.translation.y,
                                      tf.transform.translation.z);
      } catch (const tf2::TransformException &ex) {
        error_out = "validation_transform_failed target=" +
                    validation_frame_id_ + " source=" + source_frame +
                    " error=" + ex.what();
        return false;
      }
    }

    validation_cloud->points.reserve(cloud->size());
    for (const auto &pt : cloud->points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) ||
          !std::isfinite(pt.z)) {
        continue;
      }
      const Eigen::Vector3f p =
          rotation * Eigen::Vector3f(pt.x, pt.y, pt.z) + translation;
      validation_cloud->points.emplace_back(p.x(), p.y(), p.z());
    }
    validation_cloud->width =
        static_cast<std::uint32_t>(validation_cloud->points.size());
    validation_cloud->height = 1;
    validation_cloud->is_dense = false;
    return true;
  }

  void applyRoi(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                pcl::PointCloud<pcl::PointXYZ>::Ptr &roi_cloud) const {
    const float x_min =
        static_cast<float>(get_parameter("roi_x_min_m").as_double());
    const float x_max =
        static_cast<float>(get_parameter("roi_x_max_m").as_double());
    const float y_min =
        static_cast<float>(get_parameter("roi_y_min_m").as_double());
    const float y_max =
        static_cast<float>(get_parameter("roi_y_max_m").as_double());
    const float z_min =
        static_cast<float>(get_parameter("roi_z_min_m").as_double());
    const float z_max =
        static_cast<float>(get_parameter("roi_z_max_m").as_double());

    roi_cloud->points.reserve(cloud->size());
    for (const auto &pt : cloud->points) {
      if (pt.x >= x_min && pt.x <= x_max && pt.y >= y_min && pt.y <= y_max &&
          pt.z >= z_min && pt.z <= z_max) {
        roi_cloud->points.push_back(pt);
      }
    }
    roi_cloud->width = static_cast<std::uint32_t>(roi_cloud->points.size());
    roi_cloud->height = 1;
    roi_cloud->is_dense = false;
  }

  void detectHorizontalPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                              std::vector<PlaneSurface> &planes) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr remaining(
        new pcl::PointCloud<pcl::PointXYZ>);
    *remaining = *cloud;

    const int max_planes = get_parameter("max_horizontal_planes").as_int();
    const auto min_points =
        static_cast<std::size_t>(get_parameter("min_plane_points").as_int());

    for (int i = 0; i < max_planes && remaining->size() >= min_points; ++i) {
      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
      pcl::SACSegmentation<pcl::PointXYZ> seg;
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setAxis(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
      seg.setEpsAngle(get_parameter("plane_eps_angle_deg").as_double() *
                      (M_PI / 180.0));
      seg.setDistanceThreshold(
          get_parameter("plane_distance_threshold_m").as_double());
      seg.setMaxIterations(250);
      seg.setInputCloud(remaining);
      seg.segment(*inliers, *coefficients);

      if (inliers->indices.size() < min_points ||
          coefficients->values.size() < 4) {
        break;
      }

      PlaneSurface plane;
      plane.inliers = static_cast<int>(inliers->indices.size());
      plane.normal = Eigen::Vector3f(coefficients->values[0],
                                     coefficients->values[1],
                                     coefficients->values[2])
                         .normalized();
      if (plane.normal.dot(Eigen::Vector3f::UnitZ()) < 0.0f) {
        plane.normal = -plane.normal;
      }

      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(remaining);
      extract.setIndices(inliers);
      extract.setNegative(false);
      extract.filter(*plane.cloud);
      populatePlaneBounds(plane);
      planes.push_back(plane);

      extract.setNegative(true);
      pcl::PointCloud<pcl::PointXYZ>::Ptr next_remaining(
          new pcl::PointCloud<pcl::PointXYZ>);
      extract.filter(*next_remaining);
      remaining = next_remaining;
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1500,
                         "[shelf_drop_tf] planos horizontales=%zu",
                         planes.size());
  }

  void populatePlaneBounds(PlaneSurface &plane) const {
    plane.x_min = plane.y_min = std::numeric_limits<float>::max();
    plane.x_max = plane.y_max = -std::numeric_limits<float>::max();
    Eigen::Vector3f sum(0.0f, 0.0f, 0.0f);

    for (const auto &pt : plane.cloud->points) {
      plane.x_min = std::min(plane.x_min, pt.x);
      plane.x_max = std::max(plane.x_max, pt.x);
      plane.y_min = std::min(plane.y_min, pt.y);
      plane.y_max = std::max(plane.y_max, pt.y);
      sum += Eigen::Vector3f(pt.x, pt.y, pt.z);
    }

    if (!plane.cloud->empty()) {
      plane.centroid = sum / static_cast<float>(plane.cloud->size());
      plane.z = plane.centroid.z();
    }
  }

  bool selectNearestSection(const std::vector<PlaneSurface> &planes,
                            Section &best, std::string &error_out) const {
    const float min_width =
        static_cast<float>(get_parameter("surface_min_width_m").as_double());
    const float min_depth =
        static_cast<float>(get_parameter("surface_min_depth_m").as_double());
    const float min_height =
        static_cast<float>(get_parameter("min_section_height_m").as_double());
    const float open_height =
        static_cast<float>(get_parameter("open_section_height_m").as_double());
    const float min_sep =
        static_cast<float>(get_parameter("min_plane_separation_m").as_double());
    const float preferred_z = static_cast<float>(
        get_parameter("preferred_section_z_m").as_double());
    const float preferred_z_weight = static_cast<float>(
        get_parameter("preferred_section_z_weight").as_double());

    bool found = false;
    std::string first_reject;

    for (const auto &floor : planes) {
      Section section;
      section.floor = floor;
      section.width = floor.y_max - floor.y_min;
      section.depth = floor.x_max - floor.x_min;

      float ceiling_z = std::numeric_limits<float>::max();
      for (const auto &candidate_ceiling : planes) {
        const float dz = candidate_ceiling.z - floor.z;
        if (dz > min_sep && candidate_ceiling.z < ceiling_z) {
          ceiling_z = candidate_ceiling.z;
        }
      }
      if (ceiling_z == std::numeric_limits<float>::max()) {
        ceiling_z = floor.z + open_height;
      }

      section.ceiling_z = ceiling_z;
      section.height = section.ceiling_z - floor.z;

      if (section.width < min_width || section.depth < min_depth ||
          section.height < min_height) {
        if (first_reject.empty()) {
          first_reject = "size width=" + std::to_string(section.width) +
                         " depth=" + std::to_string(section.depth) +
                         " height=" + std::to_string(section.height);
        }
        continue;
      }

      const float center_y = 0.5f * (floor.y_min + floor.y_max);
      const float area = section.width * section.depth;
      section.score = -2.0f * floor.x_min - 0.35f * std::fabs(center_y) +
                      0.20f * area;
      if (preferred_z >= 0.0f) {
        section.score -= preferred_z_weight * std::fabs(floor.z - preferred_z);
      }

      if (!found || section.score > best.score) {
        best = section;
        found = true;
      }
    }

    if (!found) {
      error_out = "no_valid_shelf_section planes=" +
                  std::to_string(planes.size()) +
                  " first_reject=" + first_reject;
      return false;
    }
    return true;
  }

  std::vector<Eigen::Vector2f>
  obstacleProjections(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                      const Section &section) const {
    std::vector<Eigen::Vector2f> obstacles;
    const float min_h =
        static_cast<float>(get_parameter("obstacle_min_height_m").as_double());
    const float max_h =
        static_cast<float>(get_parameter("obstacle_max_height_m").as_double());
    const float ceiling_margin =
        static_cast<float>(get_parameter("ceiling_margin_m").as_double());

    const float obstacle_top =
        std::min(section.floor.z + max_h, section.ceiling_z - ceiling_margin);

    for (const auto &pt : cloud->points) {
      const float h = pt.z - section.floor.z;
      if (pt.x >= section.floor.x_min && pt.x <= section.floor.x_max &&
          pt.y >= section.floor.y_min && pt.y <= section.floor.y_max &&
          h > min_h && pt.z < obstacle_top) {
        obstacles.emplace_back(pt.x, pt.y);
      }
    }
    return obstacles;
  }

  bool selectCandidate(const Section &section,
                       const std::vector<Eigen::Vector2f> &obstacles,
                       Candidate &best, std::string &error_out) const {
    const float front_margin =
        static_cast<float>(get_parameter("front_margin_m").as_double());
    const float side_margin =
        static_cast<float>(get_parameter("side_margin_m").as_double());
    const float back_margin =
        static_cast<float>(get_parameter("back_margin_m").as_double());
    const float obstacle_margin =
        static_cast<float>(get_parameter("obstacle_margin_m").as_double());
    const float start_height =
        static_cast<float>(get_parameter("start_height_m").as_double());
    const float preferred_y =
        static_cast<float>(get_parameter("preferred_y_m").as_double());
    const float ceiling_margin =
        static_cast<float>(get_parameter("ceiling_margin_m").as_double());

    const float ux_min = section.floor.x_min + front_margin;
    const float ux_max = section.floor.x_max - back_margin;
    const float uy_min = section.floor.y_min + side_margin;
    const float uy_max = section.floor.y_max - side_margin;

    if (ux_min >= ux_max || uy_min >= uy_max ||
        section.floor.z + start_height >= section.ceiling_z - ceiling_margin) {
      error_out = "usable_shelf_area_too_small ux=[" +
                  std::to_string(ux_min) + "," + std::to_string(ux_max) +
                  "] uy=[" + std::to_string(uy_min) + "," +
                  std::to_string(uy_max) + "] height=" +
                  std::to_string(section.height);
      return false;
    }

    const float preferred_x = 0.5f * (ux_min + ux_max);
    constexpr int grid_x = 17;
    constexpr int grid_y = 17;
    bool found = false;

    for (int ix = 0; ix < grid_x; ++ix) {
      const float x = ux_min + (ux_max - ux_min) *
                                   (static_cast<float>(ix) /
                                    static_cast<float>(grid_x - 1));
      for (int iy = 0; iy < grid_y; ++iy) {
        const float y = uy_min + (uy_max - uy_min) *
                                     (static_cast<float>(iy) /
                                      static_cast<float>(grid_y - 1));

        float min_obstacle_distance = std::numeric_limits<float>::max();
        for (const auto &obs : obstacles) {
          const float d = std::hypot(x - obs.x(), y - obs.y());
          min_obstacle_distance = std::min(min_obstacle_distance, d);
        }
        if (min_obstacle_distance < obstacle_margin) {
          continue;
        }

        const float border_clearance =
            std::min({x - ux_min, ux_max - x, y - uy_min, uy_max - y});
        const float obstacle_score =
            std::min(min_obstacle_distance, obstacle_margin * 2.0f);
        const float center_cost =
            std::hypot(x - preferred_x, y - preferred_y);

        Candidate candidate;
        candidate.surface_point = Eigen::Vector3f(x, y, section.floor.z);
        candidate.start_point =
            candidate.surface_point + Eigen::Vector3f(0.0f, 0.0f, start_height);
        candidate.score = border_clearance + obstacle_score -
                          0.25f * center_cost;

        if (!found || candidate.score > best.score) {
          best = candidate;
          found = true;
        }
      }
    }

    if (!found) {
      error_out = "no_free_shelf_candidate obstacles=" +
                  std::to_string(obstacles.size()) +
                  " obstacle_margin=" + std::to_string(obstacle_margin);
      return false;
    }
    return true;
  }

  bool validateDropStartPose(const Eigen::Vector3f &start_point,
                             std::string &error_out) {
    const double x = start_point.x();
    const double y = start_point.y();
    const double z = start_point.z();
    const double distance_xy = std::hypot(x, y);
    const double x_min = get_parameter("safe_x_min_m").as_double();
    const double x_max = get_parameter("safe_x_max_m").as_double();
    const double abs_y_max = get_parameter("safe_abs_y_max_m").as_double();
    const double z_min = get_parameter("safe_z_min_m").as_double();
    const double z_max = get_parameter("safe_z_max_m").as_double();
    const double dist_max = get_parameter("safe_dist_xy_max_m").as_double();

    if (x < x_min || x > x_max || std::fabs(y) > abs_y_max || z < z_min ||
        z > z_max || distance_xy > dist_max) {
      error_out = "unsafe_shelf_drop_start frame=" + validation_frame_id_ +
                  " x=" + std::to_string(x) + " y=" + std::to_string(y) +
                  " z=" + std::to_string(z) +
                  " dist_xy=" + std::to_string(distance_xy);
      return false;
    }
    return true;
  }

  bool transformGeometryToFixed(const Section &section,
                                const Candidate &candidate,
                                Eigen::Vector3f &fixed_origin,
                                Eigen::Vector3f &fixed_start_point,
                                Eigen::Vector3f &fixed_axis_x,
                                Eigen::Vector3f &fixed_axis_y,
                                Eigen::Vector3f &fixed_axis_z,
                                std::string &error_out) {
    if (fixed_frame_id_.empty()) {
      error_out = "fixed_frame_id_empty";
      return false;
    }

    const Eigen::Vector3f origin(section.floor.x_min, section.floor.y_max,
                                 section.floor.z);
    const Eigen::Vector3f axis_x(0.0f, -1.0f, 0.0f);
    const Eigen::Vector3f axis_y(1.0f, 0.0f, 0.0f);
    const Eigen::Vector3f axis_z(0.0f, 0.0f, 1.0f);

    if (fixed_frame_id_ == validation_frame_id_) {
      fixed_origin = origin;
      fixed_start_point = candidate.start_point;
      fixed_axis_x = axis_x;
      fixed_axis_y = axis_y;
      fixed_axis_z = axis_z;
      return true;
    }

    try {
      const auto tf = tf_buffer_->lookupTransform(
          fixed_frame_id_, validation_frame_id_, tf2::TimePointZero, 300ms);
      Eigen::Quaternionf q(tf.transform.rotation.w, tf.transform.rotation.x,
                           tf.transform.rotation.y, tf.transform.rotation.z);
      q.normalize();
      const Eigen::Matrix3f rotation = q.toRotationMatrix();
      const Eigen::Vector3f translation(tf.transform.translation.x,
                                        tf.transform.translation.y,
                                        tf.transform.translation.z);

      fixed_origin = rotation * origin + translation;
      fixed_start_point = rotation * candidate.start_point + translation;
      fixed_axis_x = (rotation * axis_x).normalized();
      fixed_axis_y = (rotation * axis_y).normalized();
      fixed_axis_z = (rotation * axis_z).normalized();
      return true;
    } catch (const tf2::TransformException &ex) {
      error_out = "fixed_frame_transform_failed target=" + fixed_frame_id_ +
                  " source=" + validation_frame_id_ + " error=" + ex.what();
      return false;
    }
  }

  void publishTransforms(const Section &section, const Candidate &candidate,
                         const Eigen::Vector3f &fixed_origin,
                         const Eigen::Vector3f &fixed_axis_x,
                         const Eigen::Vector3f &fixed_axis_y,
                         const Eigen::Vector3f &fixed_axis_z) {
    Eigen::Matrix3f rotation;
    rotation.col(0) = fixed_axis_x;
    rotation.col(1) = fixed_axis_y;
    rotation.col(2) = fixed_axis_z;
    Eigen::Quaternionf q(rotation);
    q.normalize();

    geometry_msgs::msg::TransformStamped shelf_tf;
    shelf_tf.header.frame_id = fixed_frame_id_;
    shelf_tf.header.stamp = now();
    shelf_tf.child_frame_id = shelf_frame_id_;
    shelf_tf.transform.translation.x = fixed_origin.x();
    shelf_tf.transform.translation.y = fixed_origin.y();
    shelf_tf.transform.translation.z = fixed_origin.z();
    shelf_tf.transform.rotation.x = q.x();
    shelf_tf.transform.rotation.y = q.y();
    shelf_tf.transform.rotation.z = q.z();
    shelf_tf.transform.rotation.w = q.w();

    const Eigen::Vector3f origin(section.floor.x_min, section.floor.y_max,
                                 section.floor.z);
    const Eigen::Vector3f rel = candidate.start_point - origin;

    geometry_msgs::msg::TransformStamped drop_tf;
    drop_tf.header.stamp = shelf_tf.header.stamp;
    drop_tf.header.frame_id = shelf_frame_id_;
    drop_tf.child_frame_id = drop_start_frame_id_;
    drop_tf.transform.translation.x = rel.dot(Eigen::Vector3f(0.0f, -1.0f, 0.0f));
    drop_tf.transform.translation.y = rel.dot(Eigen::Vector3f(1.0f, 0.0f, 0.0f));
    drop_tf.transform.translation.z = rel.z();
    drop_tf.transform.rotation.w = 1.0;

    static_tf_broadcaster_->sendTransform(shelf_tf);
    static_tf_broadcaster_->sendTransform(drop_tf);
  }

  void resetDropPointsFile() {
    std::ofstream file(drop_points_file_, std::ios::trunc);
    if (!file.is_open()) {
      RCLCPP_WARN(get_logger(),
                  "[shelf_drop_tf] no se pudo reiniciar archivo: %s",
                  drop_points_file_.c_str());
      return;
    }
    file << "id,frame,x,y,z\n";
  }

  void saveDropPoint(const Eigen::Vector3f &start_point) {
    SavedDropPoint saved;
    saved.id = next_drop_point_id_++;
    saved.point = start_point;
    saved_drop_points_.push_back(saved);

    std::ofstream file(drop_points_file_, std::ios::app);
    if (!file.is_open()) {
      RCLCPP_WARN(get_logger(),
                  "[shelf_drop_tf] no se pudo escribir archivo: %s",
                  drop_points_file_.c_str());
      return;
    }
    file << saved.id << "," << fixed_frame_id_ << "," << std::fixed
         << std::setprecision(6) << saved.point.x() << "," << saved.point.y()
         << "," << saved.point.z() << "\n";
  }

  void publishMarkers() {
    visualization_msgs::msg::MarkerArray markers;
    std_msgs::msg::Header marker_header;
    marker_header.frame_id = fixed_frame_id_;
    marker_header.stamp = rclcpp::Time(0);

    visualization_msgs::msg::Marker clear;
    clear.header = marker_header;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear);

    for (const auto &saved : saved_drop_points_) {
      visualization_msgs::msg::Marker marker;
      marker.header = marker_header;
      marker.ns = "shelf_drop";
      marker.id = saved.id + 1;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = saved.point.x();
      marker.pose.position.y = saved.point.y();
      marker.pose.position.z = saved.point.z();
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.055;
      marker.scale.y = 0.055;
      marker.scale.z = 0.055;
      marker.color.r = 1.0f;
      marker.color.g = 0.65f;
      marker.color.b = 0.0f;
      marker.color.a = 0.9f;
      marker.lifetime = rclcpp::Duration(0, 0);
      markers.markers.push_back(marker);
    }

    marker_pub_->publish(markers);
    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[shelf_drop_tf] esferas seguras publicadas: count=%zu frame=%s "
        "topic=/vision/camera/drop_markers",
        saved_drop_points_.size(), fixed_frame_id_.c_str());
  }

  std::string pointcloud_topic_;
  std::string fixed_frame_id_;
  std::string validation_frame_id_;
  std::string shelf_frame_id_;
  std::string drop_start_frame_id_;
  std::string drop_points_file_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_pub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_srv_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
  std::vector<SavedDropPoint> saved_drop_points_;
  int next_drop_point_id_{0};
  bool drop_point_saved_this_activation_{false};
  bool active_{false};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ShelfDropTfNode>());
  rclcpp::shutdown();
  return 0;
}
