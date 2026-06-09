#include "rclcpp/rclcpp.hpp"

#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "Eigen/Geometry"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

class VacancyCloudFilterNode : public rclcpp::Node
{
public:
  VacancyCloudFilterNode()
  : Node("vacancy_cloud_filter_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    declare_parameter<std::string>("profile", "table");
    declare_parameter<std::string>("input_topic", "/hardware/camera/points");
    declare_parameter<std::string>("output_topic", "/vision/vacancy/points_filtered");
    declare_parameter<std::string>("input_qos_mode", "dual");
    declare_parameter<std::string>("base_link_name", "base_link");
    declare_parameter<bool>("active_on_start", false);
    declare_parameter<bool>("preserve_organized", false);

    declare_parameter<double>("roi_x_min_m", 0.20);
    declare_parameter<double>("roi_x_max_m", 1.60);
    declare_parameter<double>("roi_y_min_m", -0.85);
    declare_parameter<double>("roi_y_max_m", 0.85);
    declare_parameter<double>("roi_z_min_m", 0.0);
    declare_parameter<double>("roi_z_max_m", 1.50);

    declare_parameter<double>("voxel_leaf_size", 0.015);
    declare_parameter<int>("input_downsampling", 1);
    declare_parameter<int>("min_points_per_voxel", 1);
    declare_parameter<int>("min_neighbor_voxels", 0);
    declare_parameter<double>("processing_rate_hz", 10.0);

    declare_parameter<bool>("pause_navigation_on_enable", true);
    declare_parameter<std::string>("nav_cloud_filter_enable_service", "/navigation/cloud_filter/enable");
    declare_parameter<std::string>("nav_obstacle_detector_node", "/navigation/obstacle_detector");
    declare_parameter<std::string>(
      "nav_potential_fields_enable_topic",
      "/navigation/navigation/potential_fields/enable");
    declare_parameter<std::string>(
      "nav_potential_fields_enable_cloud_topic",
      "/navigation/navigation/potential_fields/enable_cloud");
    declare_parameter<bool>("pause_nav_lidar", true);
    declare_parameter<bool>("restore_use_point_cloud_default", true);
    declare_parameter<bool>("restore_use_lidar_default", true);
    declare_parameter<bool>("restore_potential_fields_default", true);
    declare_parameter<bool>("restore_potential_fields_cloud_default", true);
    declare_parameter<double>("nav_service_timeout_sec", 0.20);

    load_parameters();
    param_callback_handle_ = add_on_set_parameters_callback(
      std::bind(&VacancyCloudFilterNode::on_parameter_change, this, std::placeholders::_1));

    pub_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_, rclcpp::QoS(5).reliable());

    enable_srv_ = create_service<std_srvs::srv::SetBool>(
      "vacancy_cloud_filter/enable",
      std::bind(
        &VacancyCloudFilterNode::enable_callback,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    nav_cloud_filter_client_ = create_client<std_srvs::srv::SetBool>(
      nav_cloud_filter_enable_service_);
    obstacle_params_client_ =
      std::make_shared<rclcpp::AsyncParametersClient>(this, nav_obstacle_detector_node_);
    pot_fields_enable_pub_ = create_publisher<std_msgs::msg::Bool>(
      nav_potential_fields_enable_topic_, rclcpp::QoS(1).transient_local());
    pot_fields_enable_cloud_pub_ = create_publisher<std_msgs::msg::Bool>(
      nav_potential_fields_enable_cloud_topic_, rclcpp::QoS(1).transient_local());

    const auto processing_period = std::chrono::duration<double>(1.0 / processing_rate_hz_);
    processing_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(processing_period),
      std::bind(&VacancyCloudFilterNode::process_latest_cloud, this));

    set_enabled(active_on_start_, false);

    RCLCPP_INFO(
      get_logger(),
      "[vacancy_cloud_filter] profile=%s input=%s output=%s active=%s",
      profile_.c_str(), input_topic_.c_str(), output_topic_.c_str(),
      enabled_ ? "true" : "false");
  }

  ~VacancyCloudFilterNode() override
  {
    if (navigation_paused_) {
      restore_navigation();
    }
  }

private:
  struct VoxelKey
  {
    int x;
    int y;
    int z;

    bool operator==(const VoxelKey & other) const
    {
      return x == other.x && y == other.y && z == other.z;
    }
  };

  struct VoxelKeyHash
  {
    std::size_t operator()(const VoxelKey & key) const
    {
      const auto hx = std::hash<int>{}(key.x * 73856093);
      const auto hy = std::hash<int>{}(key.y * 19349663);
      const auto hz = std::hash<int>{}(key.z * 83492791);
      return hx ^ (hy << 1) ^ (hz << 2);
    }
  };

  struct VoxelAccumulator
  {
    int count = 0;
    double input_x = 0.0;
    double input_y = 0.0;
    double input_z = 0.0;
  };

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_best_effort_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_reliable_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_srv_;
  rclcpp::TimerBase::SharedPtr processing_timer_;
  OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr nav_cloud_filter_client_;
  std::shared_ptr<rclcpp::AsyncParametersClient> obstacle_params_client_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pot_fields_enable_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pot_fields_enable_cloud_pub_;

  sensor_msgs::msg::PointCloud2::SharedPtr latest_cloud_;
  std::mutex latest_cloud_mutex_;
  uint64_t latest_cloud_sequence_{0};
  uint64_t processed_cloud_sequence_{0};

  std::string profile_;
  std::string input_topic_;
  std::string output_topic_;
  std::string input_qos_mode_;
  std::string base_link_name_;
  std::string nav_cloud_filter_enable_service_;
  std::string nav_obstacle_detector_node_;
  std::string nav_potential_fields_enable_topic_;
  std::string nav_potential_fields_enable_cloud_topic_;

  double roi_x_min_;
  double roi_x_max_;
  double roi_y_min_;
  double roi_y_max_;
  double roi_z_min_;
  double roi_z_max_;
  double voxel_leaf_size_;
  double processing_rate_hz_;
  double nav_service_timeout_sec_;
  int input_downsampling_;
  int min_points_per_voxel_;
  int min_neighbor_voxels_;
  bool active_on_start_;
  bool enabled_{false};
  bool preserve_organized_;
  bool pause_navigation_on_enable_;
  bool pause_nav_lidar_;
  bool restore_use_point_cloud_default_;
  bool restore_use_lidar_default_;
  bool restore_potential_fields_default_;
  bool restore_potential_fields_cloud_default_;
  bool navigation_paused_{false};
  bool saved_obstacle_params_valid_{false};
  bool saved_use_point_cloud_{true};
  bool saved_use_lidar_{true};

  void load_parameters()
  {
    get_parameter("profile", profile_);
    get_parameter("input_topic", input_topic_);
    get_parameter("output_topic", output_topic_);
    get_parameter("input_qos_mode", input_qos_mode_);
    get_parameter("base_link_name", base_link_name_);
    get_parameter("active_on_start", active_on_start_);
    get_parameter("preserve_organized", preserve_organized_);
    get_parameter("roi_x_min_m", roi_x_min_);
    get_parameter("roi_x_max_m", roi_x_max_);
    get_parameter("roi_y_min_m", roi_y_min_);
    get_parameter("roi_y_max_m", roi_y_max_);
    get_parameter("roi_z_min_m", roi_z_min_);
    get_parameter("roi_z_max_m", roi_z_max_);
    get_parameter("voxel_leaf_size", voxel_leaf_size_);
    get_parameter("input_downsampling", input_downsampling_);
    get_parameter("min_points_per_voxel", min_points_per_voxel_);
    get_parameter("min_neighbor_voxels", min_neighbor_voxels_);
    get_parameter("processing_rate_hz", processing_rate_hz_);
    get_parameter("pause_navigation_on_enable", pause_navigation_on_enable_);
    get_parameter("nav_cloud_filter_enable_service", nav_cloud_filter_enable_service_);
    get_parameter("nav_obstacle_detector_node", nav_obstacle_detector_node_);
    get_parameter("nav_potential_fields_enable_topic", nav_potential_fields_enable_topic_);
    get_parameter(
      "nav_potential_fields_enable_cloud_topic",
      nav_potential_fields_enable_cloud_topic_);
    get_parameter("pause_nav_lidar", pause_nav_lidar_);
    get_parameter("restore_use_point_cloud_default", restore_use_point_cloud_default_);
    get_parameter("restore_use_lidar_default", restore_use_lidar_default_);
    get_parameter("restore_potential_fields_default", restore_potential_fields_default_);
    get_parameter(
      "restore_potential_fields_cloud_default",
      restore_potential_fields_cloud_default_);
    get_parameter("nav_service_timeout_sec", nav_service_timeout_sec_);

    if (roi_x_min_ > roi_x_max_) std::swap(roi_x_min_, roi_x_max_);
    if (roi_y_min_ > roi_y_max_) std::swap(roi_y_min_, roi_y_max_);
    if (roi_z_min_ > roi_z_max_) std::swap(roi_z_min_, roi_z_max_);
    voxel_leaf_size_ = std::max(0.0, voxel_leaf_size_);
    input_downsampling_ = std::max(1, input_downsampling_);
    min_points_per_voxel_ = std::max(1, min_points_per_voxel_);
    min_neighbor_voxels_ = std::max(0, min_neighbor_voxels_);
    processing_rate_hz_ = std::max(1.0, processing_rate_hz_);
    nav_service_timeout_sec_ = std::max(0.0, nav_service_timeout_sec_);
  }

  rcl_interfaces::msg::SetParametersResult on_parameter_change(
    const std::vector<rclcpp::Parameter> & params)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto & param : params) {
      const auto & name = param.get_name();
      if (name == "profile") profile_ = param.as_string();
      else if (name == "base_link_name") base_link_name_ = param.as_string();
      else if (name == "preserve_organized") preserve_organized_ = param.as_bool();
      else if (name == "roi_x_min_m") roi_x_min_ = param.as_double();
      else if (name == "roi_x_max_m") roi_x_max_ = param.as_double();
      else if (name == "roi_y_min_m") roi_y_min_ = param.as_double();
      else if (name == "roi_y_max_m") roi_y_max_ = param.as_double();
      else if (name == "roi_z_min_m") roi_z_min_ = param.as_double();
      else if (name == "roi_z_max_m") roi_z_max_ = param.as_double();
      else if (name == "voxel_leaf_size") voxel_leaf_size_ = std::max(0.0, param.as_double());
      else if (name == "input_downsampling") {
        input_downsampling_ = std::max(1, static_cast<int>(param.as_int()));
      } else if (name == "min_points_per_voxel") {
        min_points_per_voxel_ = std::max(1, static_cast<int>(param.as_int()));
      } else if (name == "min_neighbor_voxels") {
        min_neighbor_voxels_ = std::max(0, static_cast<int>(param.as_int()));
      } else if (name == "pause_navigation_on_enable") {
        pause_navigation_on_enable_ = param.as_bool();
      } else if (name == "pause_nav_lidar") {
        pause_nav_lidar_ = param.as_bool();
      } else if (name == "restore_use_point_cloud_default") {
        restore_use_point_cloud_default_ = param.as_bool();
      } else if (name == "restore_use_lidar_default") {
        restore_use_lidar_default_ = param.as_bool();
      } else if (name == "restore_potential_fields_default") {
        restore_potential_fields_default_ = param.as_bool();
      } else if (name == "restore_potential_fields_cloud_default") {
        restore_potential_fields_cloud_default_ = param.as_bool();
      } else if (name == "input_topic" || name == "output_topic" ||
                 name == "input_qos_mode" || name == "nav_cloud_filter_enable_service" ||
                 name == "nav_obstacle_detector_node" ||
                 name == "nav_potential_fields_enable_topic" ||
                 name == "nav_potential_fields_enable_cloud_topic") {
        result.successful = false;
        result.reason = "Topic/service parameters require node restart";
        break;
      }
    }

    if (roi_x_min_ > roi_x_max_) std::swap(roi_x_min_, roi_x_max_);
    if (roi_y_min_ > roi_y_max_) std::swap(roi_y_min_, roi_y_max_);
    if (roi_z_min_ > roi_z_max_) std::swap(roi_z_min_, roi_z_max_);
    return result;
  }

  void create_input_subscriptions()
  {
    const auto cb = std::bind(&VacancyCloudFilterNode::callback_cloud, this, std::placeholders::_1);
    if (input_qos_mode_ == "reliable") {
      sub_cloud_reliable_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_, rclcpp::QoS(5).reliable(), cb);
      sub_cloud_best_effort_.reset();
    } else if (input_qos_mode_ == "best_effort") {
      sub_cloud_best_effort_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_, rclcpp::SensorDataQoS(), cb);
      sub_cloud_reliable_.reset();
    } else {
      sub_cloud_best_effort_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_, rclcpp::SensorDataQoS(), cb);
      sub_cloud_reliable_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_, rclcpp::QoS(5).reliable(), cb);
    }
  }

  void clear_input_subscriptions()
  {
    sub_cloud_best_effort_.reset();
    sub_cloud_reliable_.reset();
    std::lock_guard<std::mutex> lock(latest_cloud_mutex_);
    latest_cloud_.reset();
    ++latest_cloud_sequence_;
  }

  void set_enabled(bool enabled, bool log_change)
  {
    if (enabled_ == enabled &&
        ((enabled && (sub_cloud_best_effort_ || sub_cloud_reliable_)) ||
         (!enabled && !sub_cloud_best_effort_ && !sub_cloud_reliable_))) {
      return;
    }

    enabled_ = enabled;
    if (enabled_) {
      pause_navigation();
      create_input_subscriptions();
    } else {
      clear_input_subscriptions();
      restore_navigation();
    }

    if (log_change) {
      RCLCPP_INFO(
        get_logger(),
        "[vacancy_cloud_filter] %s profile=%s",
        enabled_ ? "enabled" : "disabled",
        profile_.c_str());
    }
  }

  void enable_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response)
  {
    set_enabled(request->data, true);
    response->success = true;
    response->message =
      enabled_ ? "vacancy_cloud_filter_enabled" : "vacancy_cloud_filter_disabled";
  }

  void send_nav_cloud_filter_enable(bool enabled)
  {
    if (!nav_cloud_filter_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "[vacancy_cloud_filter] nav cloud_filter service not ready: %s",
        nav_cloud_filter_enable_service_.c_str());
      return;
    }
    auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
    request->data = enabled;
    nav_cloud_filter_client_->async_send_request(request);
  }

  void publish_potential_fields_enable(bool enabled, bool cloud_enabled)
  {
    std_msgs::msg::Bool msg;
    msg.data = enabled;
    pot_fields_enable_pub_->publish(msg);
    msg.data = cloud_enabled;
    pot_fields_enable_cloud_pub_->publish(msg);
  }

  void pause_obstacle_detector()
  {
    if (!obstacle_params_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "[vacancy_cloud_filter] obstacle detector parameter service not ready: %s",
        nav_obstacle_detector_node_.c_str());
      return;
    }

    obstacle_params_client_->get_parameters(
      {"use_point_cloud", "use_lidar"},
      [this](std::shared_future<std::vector<rclcpp::Parameter>> future) {
        try {
          const auto params = future.get();
          if (params.size() >= 2) {
            saved_use_point_cloud_ = params[0].as_bool();
            saved_use_lidar_ = params[1].as_bool();
            saved_obstacle_params_valid_ = true;
          }
        } catch (const std::exception & error) {
          RCLCPP_WARN(
            get_logger(),
            "[vacancy_cloud_filter] could not read obstacle_detector params: %s",
            error.what());
        }
      });

    std::vector<rclcpp::Parameter> params;
    params.emplace_back("use_point_cloud", false);
    if (pause_nav_lidar_) {
      params.emplace_back("use_lidar", false);
    }
    obstacle_params_client_->set_parameters(params);
  }

  void restore_obstacle_detector()
  {
    if (!obstacle_params_client_->service_is_ready()) {
      return;
    }

    const bool restore_cloud =
      saved_obstacle_params_valid_ ? saved_use_point_cloud_ : restore_use_point_cloud_default_;
    const bool restore_lidar =
      saved_obstacle_params_valid_ ? saved_use_lidar_ : restore_use_lidar_default_;

    std::vector<rclcpp::Parameter> params;
    params.emplace_back("use_point_cloud", restore_cloud);
    if (pause_nav_lidar_) {
      params.emplace_back("use_lidar", restore_lidar);
    }
    obstacle_params_client_->set_parameters(params);
  }

  void pause_navigation()
  {
    if (!pause_navigation_on_enable_ || navigation_paused_) {
      return;
    }
    navigation_paused_ = true;
    saved_obstacle_params_valid_ = false;
    send_nav_cloud_filter_enable(false);
    pause_obstacle_detector();
    publish_potential_fields_enable(false, false);
  }

  void restore_navigation()
  {
    if (!pause_navigation_on_enable_ || !navigation_paused_) {
      return;
    }
    restore_obstacle_detector();
    send_nav_cloud_filter_enable(true);
    publish_potential_fields_enable(
      restore_potential_fields_default_,
      restore_potential_fields_cloud_default_);
    navigation_paused_ = false;
  }

  void callback_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!enabled_ || !msg) {
      return;
    }
    std::lock_guard<std::mutex> lock(latest_cloud_mutex_);
    latest_cloud_ = msg;
    ++latest_cloud_sequence_;
  }

  void process_latest_cloud()
  {
    if (!enabled_) {
      return;
    }

    sensor_msgs::msg::PointCloud2::SharedPtr msg;
    uint64_t sequence = 0;
    {
      std::lock_guard<std::mutex> lock(latest_cloud_mutex_);
      if (!latest_cloud_ || latest_cloud_sequence_ == processed_cloud_sequence_) {
        return;
      }
      msg = latest_cloud_;
      sequence = latest_cloud_sequence_;
    }

    process_cloud(msg);
    processed_cloud_sequence_ = sequence;
  }

  static bool get_float32_field_offset(
    const sensor_msgs::msg::PointCloud2 & cloud,
    const std::string & field_name,
    std::size_t & offset)
  {
    for (const auto & field : cloud.fields) {
      if (field.name == field_name &&
          field.datatype == sensor_msgs::msg::PointField::FLOAT32) {
        offset = field.offset;
        return true;
      }
    }
    return false;
  }

  bool lookup_cloud_transform(
    const std_msgs::msg::Header & header,
    Eigen::Quaternionf & rotation,
    Eigen::Vector3f & translation)
  {
    try {
      const auto tf = tf_buffer_.lookupTransform(
        base_link_name_, header.frame_id, header.stamp,
        rclcpp::Duration::from_seconds(0.03));
      rotation = Eigen::Quaternionf(
        tf.transform.rotation.w,
        tf.transform.rotation.x,
        tf.transform.rotation.y,
        tf.transform.rotation.z);
      translation = Eigen::Vector3f(
        tf.transform.translation.x,
        tf.transform.translation.y,
        tf.transform.translation.z);
      rotation.normalize();
      return true;
    } catch (const std::exception &) {
      try {
        const auto tf = tf_buffer_.lookupTransform(
          base_link_name_, header.frame_id, rclcpp::Time(0),
          rclcpp::Duration::from_seconds(0.03));
        rotation = Eigen::Quaternionf(
          tf.transform.rotation.w,
          tf.transform.rotation.x,
          tf.transform.rotation.y,
          tf.transform.rotation.z);
        translation = Eigen::Vector3f(
          tf.transform.translation.x,
          tf.transform.translation.y,
          tf.transform.translation.z);
        rotation.normalize();
        return true;
      } catch (const std::exception & error) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[vacancy_cloud_filter] TF lookup failed from '%s' to '%s': %s",
          header.frame_id.c_str(), base_link_name_.c_str(), error.what());
        return false;
      }
    }
  }

  bool point_in_roi(const Eigen::Vector3f & point) const
  {
    return point.x() >= roi_x_min_ && point.x() <= roi_x_max_ &&
           point.y() >= roi_y_min_ && point.y() <= roi_y_max_ &&
           point.z() >= roi_z_min_ && point.z() <= roi_z_max_;
  }

  void publish_organized_cloud(
    const sensor_msgs::msg::PointCloud2 & input,
    std::size_t x_offset,
    std::size_t y_offset,
    std::size_t z_offset,
    const Eigen::Quaternionf & rotation,
    const Eigen::Vector3f & translation)
  {
    sensor_msgs::msg::PointCloud2 output = input;
    const auto total_points =
      static_cast<std::size_t>(input.width) * static_cast<std::size_t>(input.height);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto * base_ptr = output.data.data();

    for (std::size_t i = 0; i < total_points; ++i) {
      auto * point_ptr = base_ptr + i * output.point_step;
      float x;
      float y;
      float z;
      std::memcpy(&x, point_ptr + x_offset, sizeof(float));
      std::memcpy(&y, point_ptr + y_offset, sizeof(float));
      std::memcpy(&z, point_ptr + z_offset, sizeof(float));

      bool accepted = std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
      if (accepted) {
        const Eigen::Vector3f point_base = rotation * Eigen::Vector3f(x, y, z) + translation;
        accepted = point_in_roi(point_base);
      }

      if (!accepted) {
        std::memcpy(point_ptr + x_offset, &nan, sizeof(float));
        std::memcpy(point_ptr + y_offset, &nan, sizeof(float));
        std::memcpy(point_ptr + z_offset, &nan, sizeof(float));
      }
    }

    output.is_dense = false;
    pub_cloud_->publish(output);
  }

  int count_neighbors(
    const std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> & voxels,
    const VoxelKey & key) const
  {
    int neighbors = 0;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          VoxelKey neighbor{key.x + dx, key.y + dy, key.z + dz};
          if (voxels.find(neighbor) != voxels.end()) {
            ++neighbors;
          }
        }
      }
    }
    return neighbors;
  }

  void publish_compact_cloud(
    const sensor_msgs::msg::PointCloud2 & input,
    const std::vector<VoxelAccumulator> & accepted)
  {
    sensor_msgs::msg::PointCloud2 output;
    output.header = input.header;
    output.height = 1;
    output.width = static_cast<uint32_t>(accepted.size());
    output.is_bigendian = false;
    output.is_dense = true;

    sensor_msgs::PointCloud2Modifier modifier(output);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(accepted.size());

    sensor_msgs::PointCloud2Iterator<float> out_x(output, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(output, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(output, "z");

    for (const auto & voxel : accepted) {
      const auto inv_count = 1.0 / static_cast<double>(voxel.count);
      *out_x = static_cast<float>(voxel.input_x * inv_count);
      *out_y = static_cast<float>(voxel.input_y * inv_count);
      *out_z = static_cast<float>(voxel.input_z * inv_count);
      ++out_x;
      ++out_y;
      ++out_z;
    }

    pub_cloud_->publish(output);
  }

  void process_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!msg || msg->data.empty() || msg->point_step == 0) {
      return;
    }

    std::size_t x_offset;
    std::size_t y_offset;
    std::size_t z_offset;
    if (!get_float32_field_offset(*msg, "x", x_offset) ||
        !get_float32_field_offset(*msg, "y", y_offset) ||
        !get_float32_field_offset(*msg, "z", z_offset)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "[vacancy_cloud_filter] input cloud does not expose float32 x/y/z fields.");
      return;
    }

    Eigen::Quaternionf rotation = Eigen::Quaternionf::Identity();
    Eigen::Vector3f translation(0.0f, 0.0f, 0.0f);
    if (!lookup_cloud_transform(msg->header, rotation, translation)) {
      return;
    }

    if (preserve_organized_) {
      publish_organized_cloud(*msg, x_offset, y_offset, z_offset, rotation, translation);
      return;
    }

    std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash> voxels;
    const auto total_points =
      static_cast<std::size_t>(msg->width) * static_cast<std::size_t>(msg->height);
    voxels.reserve(total_points / 8 + 1);

    const auto * base_ptr = msg->data.data();
    const auto point_skip = static_cast<std::size_t>(std::max(1, input_downsampling_));
    const double inv_leaf = voxel_leaf_size_ > 0.0 ? 1.0 / voxel_leaf_size_ : 0.0;

    for (std::size_t i = 0; i < total_points; i += point_skip) {
      const auto * point_ptr = base_ptr + i * msg->point_step;
      float x;
      float y;
      float z;
      std::memcpy(&x, point_ptr + x_offset, sizeof(float));
      std::memcpy(&y, point_ptr + y_offset, sizeof(float));
      std::memcpy(&z, point_ptr + z_offset, sizeof(float));

      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }

      const Eigen::Vector3f point_base = rotation * Eigen::Vector3f(x, y, z) + translation;
      if (!point_in_roi(point_base)) {
        continue;
      }

      VoxelKey key;
      if (voxel_leaf_size_ > 0.0) {
        key = VoxelKey{
          static_cast<int>(std::floor(point_base.x() * inv_leaf)),
          static_cast<int>(std::floor(point_base.y() * inv_leaf)),
          static_cast<int>(std::floor(point_base.z() * inv_leaf))};
      } else {
        key = VoxelKey{
          static_cast<int>(i),
          0,
          0};
      }

      auto & voxel = voxels[key];
      voxel.count += 1;
      voxel.input_x += x;
      voxel.input_y += y;
      voxel.input_z += z;
    }

    std::vector<VoxelAccumulator> accepted;
    accepted.reserve(voxels.size());
    for (const auto & entry : voxels) {
      const auto & key = entry.first;
      const auto & voxel = entry.second;
      if (voxel.count < min_points_per_voxel_) {
        continue;
      }
      if (min_neighbor_voxels_ > 0 && count_neighbors(voxels, key) < min_neighbor_voxels_) {
        continue;
      }
      accepted.push_back(voxel);
    }

    publish_compact_cloud(*msg, accepted);
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VacancyCloudFilterNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
