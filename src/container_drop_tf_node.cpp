#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <vision_interfaces/srv/get_detections.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

using namespace std::chrono_literals;

class ContainerDropTfNode : public rclcpp::Node {
public:
  ContainerDropTfNode() : Node("container_drop_tf_node") {
    declare_parameter<std::string>("pointcloud_topic", "/hardware/camera/points");
    declare_parameter<std::string>("fixed_frame_id", "map");
    declare_parameter<std::string>("validation_frame_id", "base_link");
    declare_parameter<std::string>("container_frame_id", "container_frame");
    declare_parameter<std::string>("drop_start_frame_id",
                                   "container_drop_start");
    declare_parameter<std::string>("drop_points_file",
                                   "src/Vision/vacancy_detector/src/"
                                   "container_drop_points.txt");
    declare_parameter<std::string>("cloud_filter_enable_service",
                                   "/vision/container_cloud_filter/enable");

    declare_parameter<double>("roi_x_min_m", 0.20);
    declare_parameter<double>("roi_x_max_m", 1.60);
    declare_parameter<double>("roi_y_min_m", -0.85);
    declare_parameter<double>("roi_y_max_m", 0.85);
    declare_parameter<double>("roi_z_min_m", -0.20);
    declare_parameter<double>("roi_z_max_m", 1.05);
    declare_parameter<double>("floor_distance_threshold_m", 0.025);
    declare_parameter<double>("floor_eps_angle_deg", 20.0);
    declare_parameter<double>("object_min_height_m", 0.12);
    declare_parameter<double>("object_max_height_m", 0.85);
    declare_parameter<double>("container_min_width_m", 0.18);
    declare_parameter<double>("container_max_width_m", 0.80);
    declare_parameter<double>("container_min_depth_m", 0.18);
    declare_parameter<double>("container_max_depth_m", 0.90);
    declare_parameter<double>("partial_rim_min_depth_m", 0.05);
    declare_parameter<int>("partial_rim_min_points", 500);
    // Si solo se ve una franja del borde frontal, +X mueve el drop hacia el
    // interior estimado de la cubeta, alejandolo del robot.
    declare_parameter<double>("partial_rim_drop_x_offset_m", 0.12);
    declare_parameter<double>("rim_band_m", 0.055);
    declare_parameter<double>("min_wall_coverage", 0.45);
    declare_parameter<double>("max_interior_fill_ratio", 0.35);
    declare_parameter<bool>("reject_full_containers", true);
    declare_parameter<double>("full_container_fill_ratio", 0.85);
    declare_parameter<double>("full_container_min_coverage", 0.30);
    declare_parameter<std::string>("last_result", "idle");
    declare_parameter<std::string>("last_error", "");
    // Altura extra sobre la superficie/borde detectado donde se dibuja y
    // publica container_drop_start. Subirlo ayuda a evitar colisiones.
    declare_parameter<double>("start_clearance_m", 0.10);
    // Desplazamiento frontal del punto de esfera/drop. En base_link, +X mueve
    // la marca hacia el interior/lejos del robot; -X la acerca al robot.
    declare_parameter<double>("drop_x_offset_m", 0.0);
    // Desplazamiento lateral relativo del punto de esfera/drop. En base_link,
    // +Y mueve la marca a la izquierda del robot; -Y a la derecha.
    declare_parameter<double>("drop_y_offset_m", 0.0);
    // Si es true, ignora el Y detectado y fuerza la esfera/drop a un Y fijo.
    // Es util cuando la cubeta esta centrada pero la nube detecta solo un borde.
    declare_parameter<bool>("override_drop_y", false);
    // Valor lateral absoluto usado cuando override_drop_y=true.
    declare_parameter<double>("drop_y_override_m", 0.0);
    // Si es true, ignora la altura detectada y fuerza Z absoluta en base_link.
    declare_parameter<bool>("override_drop_z", false);
    // Valor Z absoluto usado cuando override_drop_z=true.
    declare_parameter<double>("drop_z_override_m", 1.0);
    declare_parameter<double>("cluster_tolerance_m", 0.045);
    declare_parameter<int>("cluster_min_points", 80);
    declare_parameter<int>("cluster_max_points", 12000);
    declare_parameter<double>("safe_x_min_m", 0.15);
    declare_parameter<double>("safe_x_max_m", 1.35);
    declare_parameter<double>("safe_abs_y_max_m", 0.85);
    declare_parameter<double>("safe_z_min_m", 0.08);
    declare_parameter<double>("safe_z_max_m", 1.10);
    declare_parameter<double>("safe_dist_xy_max_m", 1.50);
    declare_parameter<bool>("use_yolo_roi", true);
    declare_parameter<std::string>("yolo_service", "/vision/detection");
    declare_parameter<std::vector<std::string>>(
        "yolo_queries",
        std::vector<std::string>{"trash can", "bin", "bucket", "container",
                                 "bowl", "vase"});
    declare_parameter<double>("yolo_min_confidence", 0.30);
    declare_parameter<int>("yolo_bbox_padding_px", 20);
    declare_parameter<double>("yolo_wait_timeout_sec", 2.5);
    declare_parameter<int>("yolo_max_roi_failures", 3);
    declare_parameter<int>("yolo_roi_expand_step_px", 80);
    // Cuando la ROI de YOLO falla, seguimos usando su perimetro como pista:
    // este margen 3D permite aceptar candidatos un poco fuera de la caja
    // proyectada por ruido, calibracion o bbox demasiado ajustado.
    declare_parameter<double>("yolo_gate_margin_m", 0.12);
    declare_parameter<bool>("fallback_without_yolo", true);
    declare_parameter<bool>("active_on_start", false);

    pointcloud_topic_ = get_parameter("pointcloud_topic").as_string();
    fixed_frame_id_ = get_parameter("fixed_frame_id").as_string();
    validation_frame_id_ = get_parameter("validation_frame_id").as_string();
    container_frame_id_ = get_parameter("container_frame_id").as_string();
    drop_start_frame_id_ = get_parameter("drop_start_frame_id").as_string();
    drop_points_file_ = get_parameter("drop_points_file").as_string();
    cloud_filter_enable_service_ =
        get_parameter("cloud_filter_enable_service").as_string();
    active_ = get_parameter("active_on_start").as_bool();

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    yolo_client_ = create_client<vision_interfaces::srv::GetDetections>(
        get_parameter("yolo_service").as_string());

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&ContainerDropTfNode::cloudCallback, this,
                  std::placeholders::_1));

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "camera/drop_markers",
        rclcpp::QoS(10).transient_local().reliable());
    roi_debug_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "container_drop_tf/owl_roi_points", rclcpp::SensorDataQoS());

    resetDropPointsFile();

    enable_srv_ = create_service<std_srvs::srv::SetBool>(
        "container_drop_tf/enable",
        std::bind(&ContainerDropTfNode::enableCallback, this,
                  std::placeholders::_1, std::placeholders::_2));
    cloud_filter_enable_client_ =
        create_client<std_srvs::srv::SetBool>(cloud_filter_enable_service_);

    RCLCPP_INFO(get_logger(),
                "[container_drop_tf] Nodo iniciado. pointcloud_topic=%s "
                "fixed_frame=%s validation_frame=%s active=%s",
                pointcloud_topic_.c_str(), fixed_frame_id_.c_str(),
                validation_frame_id_.c_str(), active_ ? "true" : "false");
  }

private:
  struct Candidate {
    Eigen::Vector3f center_floor{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f drop_start{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f axis_x{1.0f, 0.0f, 0.0f};
    Eigen::Vector3f axis_y{0.0f, 1.0f, 0.0f};
    Eigen::Vector3f axis_z{0.0f, 0.0f, 1.0f};
    float width{0.0f};
    float depth{0.0f};
    float height{0.0f};
    float coverage{0.0f};
    float interior_fill{1.0f};
    float score{-std::numeric_limits<float>::max()};
    bool partial_rim{false};
  };

  struct ShapeScore {
    float coverage{0.0f};
    float interior_fill{1.0f};
  };

  struct SavedDropPoint {
    int id{0};
    Eigen::Vector3f point{0.0f, 0.0f, 0.0f};
  };

  struct ImageRoi {
    int base_x_min{0};
    int base_y_min{0};
    int base_x_max{0};
    int base_y_max{0};
    int x_min{0};
    int y_min{0};
    int x_max{0};
    int y_max{0};
    double confidence{0.0};
    std::string class_name;
  };

  struct YoloGateBounds {
    bool valid{false};
    int points{0};
    float min_x{std::numeric_limits<float>::max()};
    float max_x{-std::numeric_limits<float>::max()};
    float min_y{std::numeric_limits<float>::max()};
    float max_y{-std::numeric_limits<float>::max()};
    float min_z{std::numeric_limits<float>::max()};
    float max_z{-std::numeric_limits<float>::max()};
    std::string class_name;
  };

  struct SearchBounds {
    float x_min{0.0f};
    float x_max{0.0f};
    float y_min{0.0f};
    float y_max{0.0f};
    float z_min{0.0f};
    float z_max{0.0f};
  };

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!active_) {
      RCLCPP_DEBUG_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "[container_drop_tf] inactivo; ignorando nubes hasta enable=true");
      return;
    }

    if (drop_point_saved_this_activation_) {
      RCLCPP_DEBUG_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "[container_drop_tf] punto ya guardado en esta activacion");
      return;
    }

    if (container_rejected_this_activation_) {
      RCLCPP_DEBUG_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "[container_drop_tf] contenedor lleno ya rechazado en esta "
          "activacion; ignorando nubes restantes");
      return;
    }

    std::string error;
    if (!prepareYoloRoiForCloud()) {
      return;
    }
    const bool used_yolo_roi = has_selected_yolo_roi_;
    if (!processCloud(*msg, error)) {
      if (error.find("container_full") != std::string::npos) {
        container_rejected_this_activation_ = true;
        active_ = false;
        yolo_request_failed_ = true;
        has_selected_yolo_roi_ = false;
        has_yolo_gate_roi_ = false;
        set_parameter(rclcpp::Parameter("last_result", "container_full"));
        set_parameter(rclcpp::Parameter("last_error", error));
        setCloudFilterEnabled(false);
        RCLCPP_WARN(get_logger(),
                    "[container_drop_tf] contenedor lleno rechazado; se omiten "
                    "expansiones ROI y procesamiento adicional: %s",
                    error.c_str());
        return;
      }

      if (used_yolo_roi && get_parameter("fallback_without_yolo").as_bool()) {
        ++yolo_roi_failure_count_;
        const int max_failures =
            get_parameter("yolo_max_roi_failures").as_int();
        RCLCPP_WARN(get_logger(),
                    "[container_drop_tf] YOLO ROI fallo %d/%d: %s",
                    yolo_roi_failure_count_, max_failures, error.c_str());
        if (yolo_roi_failure_count_ < max_failures) {
          expandYoloRoi(yolo_roi_failure_count_);
        } else {
          yolo_gate_roi_ = selected_yolo_roi_;
          has_yolo_gate_roi_ = true;
          has_selected_yolo_roi_ = false;
          yolo_request_failed_ = true;
          RCLCPP_WARN(get_logger(),
                      "[container_drop_tf] YOLO ROI descartado; usando "
                      "nube completa filtrada por perimetro YOLO");
        }
      }
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "[container_drop_tf] No se pudo calcular TF: %s",
                           error.c_str());
    }
  }

  void enableCallback(
      const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
      std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
    active_ = request->data;
    if (active_) {
      drop_point_saved_this_activation_ = false;
      container_rejected_this_activation_ = false;
      set_parameter(rclcpp::Parameter("last_result", "running"));
      set_parameter(rclcpp::Parameter("last_error", ""));
      // Cada escaneo debe iniciar sin esferas previas; si no, RViz muestra
      // puntos viejos mientras OWLv2 y la nube nueva todavia estan procesando.
      saved_drop_points_.clear();
      publishMarkers();
      publishDebugCloud(pcl::PointCloud<pcl::PointXYZ>(), validation_frame_id_);
      setCloudFilterEnabled(true);
      resetYoloState();
      requestYoloDetections();
    } else {
      set_parameter(rclcpp::Parameter("last_result", "disabled"));
      setCloudFilterEnabled(false);
    }

    response->success = true;
    response->message = active_ ? "container_drop_tf_enabled"
                                : "container_drop_tf_disabled";
    RCLCPP_INFO(get_logger(), "[container_drop_tf] %s",
                response->message.c_str());
  }

  void resetYoloState() {
    has_selected_yolo_roi_ = false;
    yolo_request_pending_ = false;
    yolo_request_failed_ = false;
    yolo_roi_failure_count_ = 0;
    yolo_request_stamp_ = now();
    selected_yolo_roi_ = ImageRoi{};
    yolo_gate_roi_ = ImageRoi{};
    has_yolo_gate_roi_ = false;
    yolo_future_ = {};
  }

  void requestYoloDetections() {
    if (!get_parameter("use_yolo_roi").as_bool()) {
      RCLCPP_INFO(get_logger(),
                  "[container_drop_tf] YOLO ROI deshabilitado por parametro");
      return;
    }

    if (!yolo_client_->wait_for_service(100ms)) {
      yolo_request_failed_ = true;
      RCLCPP_WARN(get_logger(),
                  "[container_drop_tf] YOLO no disponible en %s; usando "
                  "fallback geometrico si esta permitido",
                  get_parameter("yolo_service").as_string().c_str());
      return;
    }

    auto request = std::make_shared<vision_interfaces::srv::GetDetections::Request>();
    request->queries = get_parameter("yolo_queries").as_string_array();
    auto future_and_request_id = yolo_client_->async_send_request(request);
    yolo_future_ = future_and_request_id.future.share();
    yolo_request_pending_ = true;
    yolo_request_stamp_ = now();

    std::string query_list;
    for (const auto &query : request->queries) {
      query_list += query_list.empty() ? query : "," + query;
    }
    RCLCPP_INFO(get_logger(),
                "[container_drop_tf] YOLO request enviado: service=%s "
                "queries=[%s]",
                get_parameter("yolo_service").as_string().c_str(),
                query_list.c_str());
  }

  bool prepareYoloRoiForCloud() {
    if (!get_parameter("use_yolo_roi").as_bool() || has_selected_yolo_roi_) {
      return true;
    }

    if (yolo_request_pending_) {
      if (yolo_future_.valid() &&
          yolo_future_.wait_for(0ms) == std::future_status::ready) {
        yolo_request_pending_ = false;
        const auto response = yolo_future_.get();
        handleYoloResponse(response);
        return true;
      }

      const double elapsed = (now() - yolo_request_stamp_).seconds();
      const double timeout =
          get_parameter("yolo_wait_timeout_sec").as_double();
      if (elapsed < timeout) {
        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 500,
            "[container_drop_tf] esperando YOLO ROI elapsed=%.2f/%.2f",
            elapsed, timeout);
        return false;
      }

      yolo_request_pending_ = false;
      yolo_request_failed_ = true;
      RCLCPP_WARN(get_logger(),
                  "[container_drop_tf] YOLO timeout despues de %.2fs",
                  elapsed);
    }

    if (yolo_request_failed_ &&
        !get_parameter("fallback_without_yolo").as_bool()) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "[container_drop_tf] sin YOLO ROI y fallback deshabilitado");
      return false;
    }

    if (yolo_request_failed_) {
      if (has_yolo_gate_roi_) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "[container_drop_tf] procesando nube completa con gate YOLO");
        return true;
      }
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[container_drop_tf] procesando sin YOLO ROI (fallback geometrico)");
    }
    return true;
  }

  void handleYoloResponse(
      const vision_interfaces::srv::GetDetections::Response::SharedPtr
          &response) {
    if (!response || !response->success || response->detections.empty()) {
      yolo_request_failed_ = true;
      RCLCPP_WARN(get_logger(),
                  "[container_drop_tf] YOLO sin detecciones utiles: %s",
                  response ? response->message.c_str() : "null_response");
      return;
    }

    const double min_conf = get_parameter("yolo_min_confidence").as_double();
    const int padding = get_parameter("yolo_bbox_padding_px").as_int();
    bool found = false;
    ImageRoi best;
    double best_score = -std::numeric_limits<double>::max();

    RCLCPP_INFO(get_logger(),
                "[container_drop_tf] YOLO detections=%zu message='%s'",
                response->detections.size(), response->message.c_str());

    std::unordered_map<std::string, int> class_counts;
    for (const auto &det : response->detections) {
      const double width = std::max(0.0, det.x_max - det.x_min);
      const double height = std::max(0.0, det.y_max - det.y_min);
      const double area = width * height;
      const double center_x = 0.5 * (det.x_min + det.x_max);
      const double center_offset = std::fabs(center_x - 320.0) / 320.0;

      std::string sanitized = det.class_name;
      std::replace(sanitized.begin(), sanitized.end(), ' ', '_');
      const int instance = class_counts[sanitized]++;
      const std::string frame = "OWL_" + sanitized + "_" +
                                std::to_string(instance);

      bool has_tf = false;
      double base_x = 0.0;
      double base_y = 0.0;
      try {
        const auto tf = tf_buffer_->lookupTransform(
            validation_frame_id_, frame, tf2::TimePointZero, 200ms);
        has_tf = true;
        base_x = tf.transform.translation.x;
        base_y = tf.transform.translation.y;
      } catch (const tf2::TransformException &ex) {
        RCLCPP_DEBUG(get_logger(),
                     "[container_drop_tf] YOLO det sin TF frame=%s error=%s",
                     frame.c_str(), ex.what());
      }

      const double area_score = area / 10000.0;
      const double distance_score =
          has_tf ? (get_parameter("roi_x_max_m").as_double() - base_x) /
                       std::max(0.001, get_parameter("roi_x_max_m").as_double() -
                                           get_parameter("roi_x_min_m").as_double())
                 : 0.0;
      const double lateral_score =
          has_tf ? -0.25 * std::fabs(base_y) : 0.0;
      const double score = 0.35 * det.confidence + 1.35 * area_score +
                           1.00 * distance_score + lateral_score -
                           0.15 * center_offset;

      RCLCPP_INFO(
          get_logger(),
          "[container_drop_tf] YOLO det class='%s' conf=%.2f bbox=(%.0f %.0f "
          "%.0f %.0f) area=%.0f frame=%s base=(%.3f %.3f) tf=%s score=%.3f",
          det.class_name.c_str(), det.confidence, det.x_min, det.y_min,
          det.x_max, det.y_max, area, frame.c_str(), base_x, base_y,
          has_tf ? "true" : "false", score);

      if (det.confidence < min_conf || width < 5.0 || height < 5.0) {
        continue;
      }

      if (!found || score > best_score) {
        best_score = score;
        best.class_name = det.class_name;
        best.confidence = det.confidence;
        best.base_x_min = static_cast<int>(std::floor(det.x_min)) - padding;
        best.base_y_min = static_cast<int>(std::floor(det.y_min)) - padding;
        best.base_x_max = static_cast<int>(std::ceil(det.x_max)) + padding;
        best.base_y_max = static_cast<int>(std::ceil(det.y_max)) + padding;
        best.x_min = best.base_x_min;
        best.y_min = best.base_y_min;
        best.x_max = best.base_x_max;
        best.y_max = best.base_y_max;
        found = true;
      }
    }

    if (!found) {
      yolo_request_failed_ = true;
      RCLCPP_WARN(get_logger(),
                  "[container_drop_tf] YOLO detecciones descartadas por "
                  "confianza/tamano min_conf=%.2f",
                  min_conf);
      return;
    }

    selected_yolo_roi_ = best;
    yolo_gate_roi_ = best;
    has_selected_yolo_roi_ = true;
    has_yolo_gate_roi_ = true;
    RCLCPP_INFO(get_logger(),
                "[container_drop_tf] YOLO ROI elegido class='%s' conf=%.2f "
                "bbox_padded=(%d %d %d %d)",
                selected_yolo_roi_.class_name.c_str(),
                selected_yolo_roi_.confidence, selected_yolo_roi_.x_min,
                selected_yolo_roi_.y_min, selected_yolo_roi_.x_max,
                selected_yolo_roi_.y_max);
  }

  void expandYoloRoi(int failure_count) {
    const int step = get_parameter("yolo_roi_expand_step_px").as_int();
    const int extra = std::max(0, failure_count) * step;
    selected_yolo_roi_.x_min = selected_yolo_roi_.base_x_min - extra;
    selected_yolo_roi_.y_min = selected_yolo_roi_.base_y_min - extra;
    selected_yolo_roi_.x_max = selected_yolo_roi_.base_x_max + extra;
    selected_yolo_roi_.y_max = selected_yolo_roi_.base_y_max + extra;
    yolo_gate_roi_ = selected_yolo_roi_;
    has_yolo_gate_roi_ = true;

    RCLCPP_WARN(get_logger(),
                "[container_drop_tf] expandiendo YOLO ROI por falla: "
                "failure=%d extra=%d bbox=(%d %d %d %d)",
                failure_count, extra, selected_yolo_roi_.x_min,
                selected_yolo_roi_.y_min, selected_yolo_roi_.x_max,
                selected_yolo_roi_.y_max);
  }

  bool processCloud(const sensor_msgs::msg::PointCloud2 &input_msg,
                    std::string &error_out) {
    // Secuencia principal:
    // 1) pasar la nube al frame de validacion (normalmente base_link),
    // 2) recortar la zona frente al robot,
    // 3) separar piso y objetos,
    // 4) buscar clusters con forma de contenedor,
    // 5) publicar una TF segura sobre el centro del hueco.
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(input_msg, *cloud);
    if (cloud->empty()) {
      error_out = "empty_cloud";
      return false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr validation_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    // Trabajar en base_link hace que x/y/z tengan significado fisico claro:
    // x frente al robot, y lateral, z altura sobre el piso.
    const ImageRoi *image_roi =
        has_selected_yolo_roi_ ? &selected_yolo_roi_ : nullptr;
    const ImageRoi *gate_roi =
        (!has_selected_yolo_roi_ && has_yolo_gate_roi_) ? &yolo_gate_roi_
                                                        : nullptr;
    YoloGateBounds yolo_gate_bounds;
    if (!transformCloudToValidation(input_msg.header.frame_id, cloud,
                                    validation_cloud, error_out, image_roi,
                                    gate_roi, yolo_gate_bounds)) {
      return false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    // La ROI reduce falsos positivos: buscamos contenedores bajos y cercanos,
    // no paredes, personas ni muebles fuera del alcance de drop.
    applyRoi(validation_cloud, roi_cloud);
    publishDebugCloud(*roi_cloud, validation_frame_id_);

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "[container_drop_tf] ROI puntos: entrada=%zu validation=%zu roi=%zu "
        "yolo_roi=%s",
        cloud->size(), validation_cloud->size(), roi_cloud->size(),
        image_roi ? selected_yolo_roi_.class_name.c_str()
                  : (yolo_gate_bounds.valid ? "gate_only" : "none"));

    if (roi_cloud->size() < 500) {
      error_out = "insufficient_points_after_roi count=" +
                  std::to_string(roi_cloud->size());
      return false;
    }

    pcl::ModelCoefficients::Ptr floor_coefficients(
        new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr floor_inliers(new pcl::PointIndices);
    // El piso es la referencia para medir altura real del contenedor.
    if (!segmentFloor(roi_cloud, floor_coefficients, floor_inliers,
                      error_out)) {
      return false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr floor_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr object_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(roi_cloud);
    extract.setIndices(floor_inliers);
    extract.setNegative(false);
    extract.filter(*floor_cloud);
    extract.setNegative(true);
    extract.filter(*object_cloud);

    const float floor_z = centroid(floor_cloud).z();
    pcl::PointCloud<pcl::PointXYZ>::Ptr above_floor_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    // Quitamos puntos demasiado pegados al piso y objetos demasiado altos.
    // Esto deja candidatos tipo bote/caja y reduce ruido de suelo.
    filterObjectsAboveFloor(object_cloud, floor_z, above_floor_cloud);

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "[container_drop_tf] piso inliers=%zu objetos=%zu sobre_piso=%zu "
        "floor_z=%.3f",
        floor_inliers->indices.size(), object_cloud->size(),
        above_floor_cloud->size(), floor_z);

    if (above_floor_cloud->size() < 100) {
      error_out = "insufficient_object_points count=" +
                  std::to_string(above_floor_cloud->size());
      return false;
    }

    std::vector<pcl::PointIndices> clusters;
    // Cada cluster es un objeto fisico candidato: bote, caja, pata, mochila, etc.
    extractClusters(above_floor_cloud, clusters);
    if (clusters.empty()) {
      error_out = "no_clusters_above_floor";
      return false;
    }

    Candidate best;
    // Filtramos clusters por tamano, cobertura de borde e interior libre.
    if (!selectBestContainer(above_floor_cloud, clusters, floor_z, best,
                             error_out,
                             yolo_gate_bounds.valid ? &yolo_gate_bounds
                                                    : nullptr)) {
      return false;
    }

    if (!validateDropStartPose(best.drop_start, error_out)) {
      return false;
    }

    Candidate fixed_best = best;
    // Publicamos las TF en fixed_frame como estaticas para que permanezcan
    // visibles/disponibles hasta que una nueva busqueda las reemplace.
    if (!transformCandidateToFixed(fixed_best, error_out)) {
      return false;
    }

    publishTransforms(fixed_best);
    saveDropPoint(fixed_best.drop_start);
    publishMarkers();
    drop_point_saved_this_activation_ = true;
    set_parameter(rclcpp::Parameter("last_result", "success"));
    set_parameter(rclcpp::Parameter("last_error", ""));
    setCloudFilterEnabled(false);

    RCLCPP_INFO(
        get_logger(),
        "[container_drop_tf] TF %s publicada: base=(%.3f %.3f %.3f) "
        "size=(%.3f %.3f %.3f) coverage=%.2f interior=%.2f score=%.2f "
        "mode=%s",
        drop_start_frame_id_.c_str(), best.drop_start.x(),
        best.drop_start.y(), best.drop_start.z(), best.width, best.depth,
        best.height, best.coverage, best.interior_fill, best.score,
        best.partial_rim ? "partial_rim" : "full");

    error_out.clear();
    return true;
  }

  void setCloudFilterEnabled(bool enabled) {
    if (!cloud_filter_enable_client_->wait_for_service(1s)) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "[container_drop_tf] filtro vacancy no disponible: %s",
          cloud_filter_enable_service_.c_str());
      return;
    }
    auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
    request->data = enabled;
    cloud_filter_enable_client_->async_send_request(
        request,
        [this, enabled](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture future) {
          const auto response = future.get();
          RCLCPP_INFO(get_logger(),
                      "[container_drop_tf] filtro vacancy %s: success=%s msg='%s'",
                      enabled ? "enable" : "disable",
                      response->success ? "true" : "false",
                      response->message.c_str());
        });
  }

  bool transformCloudToValidation(
      const std::string &source_frame,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
      pcl::PointCloud<pcl::PointXYZ>::Ptr &validation_cloud,
      std::string &error_out, const ImageRoi *image_roi,
      const ImageRoi *gate_roi, YoloGateBounds &gate_bounds) {
    // Convierte cada punto de la nube al frame de validacion. No usamos una
    // transformacion PCL completa para mantener este nodo autocontenido.
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

    const bool can_use_image_roi =
        image_roi != nullptr && cloud->height > 1 && cloud->width > 1 &&
        cloud->points.size() ==
            static_cast<std::size_t>(cloud->width) *
                static_cast<std::size_t>(cloud->height);
    const bool can_use_gate_roi =
        !can_use_image_roi && gate_roi != nullptr && cloud->height > 1 &&
        cloud->width > 1 &&
        cloud->points.size() ==
            static_cast<std::size_t>(cloud->width) *
                static_cast<std::size_t>(cloud->height);

    int u_min = 0;
    int v_min = 0;
    int u_max = static_cast<int>(cloud->width) - 1;
    int v_max = static_cast<int>(cloud->height) - 1;
    int gate_u_min = 0;
    int gate_v_min = 0;
    int gate_u_max = static_cast<int>(cloud->width) - 1;
    int gate_v_max = static_cast<int>(cloud->height) - 1;
    if (can_use_image_roi) {
      u_min = std::clamp(image_roi->x_min, 0, static_cast<int>(cloud->width) - 1);
      v_min = std::clamp(image_roi->y_min, 0, static_cast<int>(cloud->height) - 1);
      u_max = std::clamp(image_roi->x_max, 0, static_cast<int>(cloud->width) - 1);
      v_max = std::clamp(image_roi->y_max, 0, static_cast<int>(cloud->height) - 1);
      if (u_max < u_min) {
        std::swap(u_min, u_max);
      }
      if (v_max < v_min) {
        std::swap(v_min, v_max);
      }
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1500,
          "[container_drop_tf] aplicando YOLO bbox en nube organizada: "
          "class='%s' conf=%.2f pixel=[%d,%d -> %d,%d] cloud=%ux%u",
          image_roi->class_name.c_str(), image_roi->confidence, u_min, v_min,
          u_max, v_max, cloud->width, cloud->height);
    } else if (can_use_gate_roi) {
      gate_u_min =
          std::clamp(gate_roi->x_min, 0, static_cast<int>(cloud->width) - 1);
      gate_v_min =
          std::clamp(gate_roi->y_min, 0, static_cast<int>(cloud->height) - 1);
      gate_u_max =
          std::clamp(gate_roi->x_max, 0, static_cast<int>(cloud->width) - 1);
      gate_v_max =
          std::clamp(gate_roi->y_max, 0, static_cast<int>(cloud->height) - 1);
      if (gate_u_max < gate_u_min) {
        std::swap(gate_u_min, gate_u_max);
      }
      if (gate_v_max < gate_v_min) {
        std::swap(gate_v_min, gate_v_max);
      }
      gate_bounds.class_name = gate_roi->class_name;
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1500,
          "[container_drop_tf] usando nube completa con gate YOLO: "
          "class='%s' conf=%.2f pixel=[%d,%d -> %d,%d] cloud=%ux%u",
          gate_roi->class_name.c_str(), gate_roi->confidence, gate_u_min,
          gate_v_min, gate_u_max, gate_v_max, cloud->width, cloud->height);
    } else if (image_roi != nullptr) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[container_drop_tf] YOLO ROI disponible pero la nube no esta "
          "organizada; se usara nube completa cloud=%ux%u points=%zu",
          cloud->width, cloud->height, cloud->points.size());
    } else if (gate_roi != nullptr) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "[container_drop_tf] gate YOLO disponible pero la nube no esta "
          "organizada; no se podra filtrar candidatos por perimetro "
          "cloud=%ux%u points=%zu",
          cloud->width, cloud->height, cloud->points.size());
    }

    const std::size_t reserve_size =
        can_use_image_roi
            ? static_cast<std::size_t>(u_max - u_min + 1) *
                  static_cast<std::size_t>(v_max - v_min + 1)
            : cloud->size();
    validation_cloud->points.reserve(reserve_size);
    const SearchBounds search_bounds = readSearchBounds();

    const auto update_gate_bounds = [&](const Eigen::Vector3f &p) {
      if (!isInsideSearchBounds(p, search_bounds)) {
        return;
      }
      ++gate_bounds.points;
      gate_bounds.min_x = std::min(gate_bounds.min_x, p.x());
      gate_bounds.max_x = std::max(gate_bounds.max_x, p.x());
      gate_bounds.min_y = std::min(gate_bounds.min_y, p.y());
      gate_bounds.max_y = std::max(gate_bounds.max_y, p.y());
      gate_bounds.min_z = std::min(gate_bounds.min_z, p.z());
      gate_bounds.max_z = std::max(gate_bounds.max_z, p.z());
    };

    const auto append_transformed = [&](const pcl::PointXYZ &pt, int u,
                                        int v) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) ||
          !std::isfinite(pt.z)) {
        return;
      }
      const Eigen::Vector3f p =
          rotation * Eigen::Vector3f(pt.x, pt.y, pt.z) + translation;
      validation_cloud->points.emplace_back(p.x(), p.y(), p.z());
      if (can_use_gate_roi && u >= gate_u_min && u <= gate_u_max &&
          v >= gate_v_min && v <= gate_v_max) {
        update_gate_bounds(p);
      }
    };

    if (can_use_image_roi) {
      for (int v = v_min; v <= v_max; ++v) {
        for (int u = u_min; u <= u_max; ++u) {
          const std::size_t idx =
              static_cast<std::size_t>(v) * cloud->width +
              static_cast<std::size_t>(u);
          append_transformed(cloud->points[idx], u, v);
        }
      }
    } else {
      for (int v = 0; v < static_cast<int>(cloud->height); ++v) {
        for (int u = 0; u < static_cast<int>(cloud->width); ++u) {
          const std::size_t idx =
              static_cast<std::size_t>(v) * cloud->width +
              static_cast<std::size_t>(u);
          append_transformed(cloud->points[idx], u, v);
        }
      }
    }
    validation_cloud->width =
        static_cast<std::uint32_t>(validation_cloud->points.size());
    validation_cloud->height = 1;
    validation_cloud->is_dense = false;

    if (can_use_image_roi || can_use_gate_roi) {
      logCloudWindow(validation_cloud,
                     can_use_image_roi ? "owl_roi_3d" : "owl_gate_full_cloud");
    }

    if (can_use_gate_roi) {
      gate_bounds.valid = gate_bounds.points >= 30;
      if (gate_bounds.valid) {
        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1500,
            "[container_drop_tf] gate YOLO 3D: class='%s' puntos=%d "
            "x=[%.3f %.3f] y=[%.3f %.3f] z=[%.3f %.3f]",
            gate_bounds.class_name.c_str(), gate_bounds.points,
            gate_bounds.min_x, gate_bounds.max_x, gate_bounds.min_y,
            gate_bounds.max_y, gate_bounds.min_z, gate_bounds.max_z);
      } else {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 1500,
            "[container_drop_tf] gate YOLO sin suficientes puntos validos: "
            "class='%s' puntos=%d",
            gate_bounds.class_name.c_str(), gate_bounds.points);
      }
    }
    return true;
  }

  SearchBounds readSearchBounds() const {
    SearchBounds bounds;
    bounds.x_min =
        static_cast<float>(get_parameter("roi_x_min_m").as_double());
    bounds.x_max =
        static_cast<float>(get_parameter("roi_x_max_m").as_double());
    bounds.y_min =
        static_cast<float>(get_parameter("roi_y_min_m").as_double());
    bounds.y_max =
        static_cast<float>(get_parameter("roi_y_max_m").as_double());
    bounds.z_min =
        static_cast<float>(get_parameter("roi_z_min_m").as_double());
    bounds.z_max =
        static_cast<float>(get_parameter("roi_z_max_m").as_double());
    return bounds;
  }

  static bool isInsideSearchBounds(const Eigen::Vector3f &p,
                                   const SearchBounds &bounds) {
    return p.x() >= bounds.x_min && p.x() <= bounds.x_max &&
           p.y() >= bounds.y_min && p.y() <= bounds.y_max &&
           p.z() >= bounds.z_min && p.z() <= bounds.z_max;
  }

  void applyRoi(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                pcl::PointCloud<pcl::PointXYZ>::Ptr &roi_cloud) const {
    // Filtro rectangular en base_link. Los parametros se pueden ajustar desde
    // launch si el robot debe buscar mas cerca/lejos o con otro campo lateral.
    const SearchBounds search_bounds = readSearchBounds();
    roi_cloud->points.reserve(cloud->size());
    for (const auto &pt : cloud->points) {
      const Eigen::Vector3f p(pt.x, pt.y, pt.z);
      if (isInsideSearchBounds(p, search_bounds)) {
        roi_cloud->points.push_back(pt);
      }
    }
    roi_cloud->width = static_cast<std::uint32_t>(roi_cloud->points.size());
    roi_cloud->height = 1;
    roi_cloud->is_dense = false;
  }

  bool segmentFloor(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                    pcl::ModelCoefficients::Ptr &coefficients,
                    pcl::PointIndices::Ptr &inliers,
                    std::string &error_out) {
    // RANSAC busca un plano casi horizontal. En base_link el eje Z apunta hacia
    // arriba, por eso el plano del piso debe ser perpendicular a Z.
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setAxis(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
    seg.setEpsAngle(get_parameter("floor_eps_angle_deg").as_double() *
                    (M_PI / 180.0));
    seg.setDistanceThreshold(
        get_parameter("floor_distance_threshold_m").as_double());
    seg.setMaxIterations(250);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "[container_drop_tf] RANSAC piso: inliers=%zu coef=[%.4f %.4f %.4f "
        "%.4f]",
        inliers->indices.size(),
        coefficients->values.empty() ? 0.0f : coefficients->values[0],
        coefficients->values.size() < 2 ? 0.0f : coefficients->values[1],
        coefficients->values.size() < 3 ? 0.0f : coefficients->values[2],
        coefficients->values.size() < 4 ? 0.0f : coefficients->values[3]);

    if (inliers->indices.size() < 300) {
      error_out = "floor_not_found inliers=" +
                  std::to_string(inliers->indices.size());
      return false;
    }
    return true;
  }

  void filterObjectsAboveFloor(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, float floor_z,
      pcl::PointCloud<pcl::PointXYZ>::Ptr &object_cloud) const {
    // Solo dejamos puntos dentro de una banda de altura compatible con
    // contenedores en piso; por default entre 12 cm y 85 cm.
    const float min_h =
        static_cast<float>(get_parameter("object_min_height_m").as_double());
    const float max_h =
        static_cast<float>(get_parameter("object_max_height_m").as_double());

    object_cloud->points.reserve(cloud->size());
    for (const auto &pt : cloud->points) {
      const float h = pt.z - floor_z;
      if (h >= min_h && h <= max_h) {
        object_cloud->points.push_back(pt);
      }
    }
    object_cloud->width =
        static_cast<std::uint32_t>(object_cloud->points.size());
    object_cloud->height = 1;
    object_cloud->is_dense = false;
  }

  void extractClusters(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                       std::vector<pcl::PointIndices> &clusters){
    // Cluster euclidiano: agrupa puntos cercanos en objetos independientes.
    // cluster_tolerance_m controla que tan separados pueden estar los puntos.
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(get_parameter("cluster_tolerance_m").as_double());
    ec.setMinClusterSize(get_parameter("cluster_min_points").as_int());
    ec.setMaxClusterSize(get_parameter("cluster_max_points").as_int());
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(clusters);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1500,
                         "[container_drop_tf] clusters=%zu",
                         clusters.size());
  }

  bool selectBestContainer(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                           const std::vector<pcl::PointIndices> &clusters,
                           float floor_z, Candidate &best,
                           std::string &error_out,
                           const YoloGateBounds *yolo_gate_bounds = nullptr) {
    // Evaluamos todos los objetos y conservamos el que parezca mas contenedor:
    // buena cobertura de paredes, interior poco lleno y posicion alcanzable.
    bool found = false;
    std::string reject_summary;

    int cluster_id = 0;
    for (const auto &cluster : clusters) {
      Candidate candidate;
      std::string reject_reason;
      if (!evaluateCluster(cloud, cluster, floor_z, candidate,
                           reject_reason)) {
        RCLCPP_INFO(
            get_logger(),
            "[container_drop_tf] cluster[%d] rechazado: puntos=%zu reason=%s",
            cluster_id, cluster.indices.size(), reject_reason.c_str());
        if (reject_summary.empty()) {
          reject_summary = reject_reason;
        }
        ++cluster_id;
        continue;
      }

      if (yolo_gate_bounds != nullptr) {
        std::string gate_reject;
        if (!isCandidateInsideYoloGate(candidate, *yolo_gate_bounds,
                                       gate_reject)) {
          RCLCPP_INFO(
              get_logger(),
              "[container_drop_tf] cluster[%d] rechazado: puntos=%zu "
              "center=(%.3f %.3f %.3f) drop=(%.3f %.3f %.3f) "
              "size=(%.3f %.3f %.3f) reason=%s",
              cluster_id, cluster.indices.size(), candidate.center_floor.x(),
              candidate.center_floor.y(), candidate.center_floor.z(),
              candidate.drop_start.x(), candidate.drop_start.y(),
              candidate.drop_start.z(), candidate.width, candidate.depth,
              candidate.height, gate_reject.c_str());
          if (reject_summary.empty()) {
            reject_summary = gate_reject;
          }
          ++cluster_id;
          continue;
        }
      }

      std::string safety_reject;
      if (!validateDropStartPose(candidate.drop_start, safety_reject)) {
        RCLCPP_INFO(
            get_logger(),
            "[container_drop_tf] cluster[%d] inseguro: puntos=%zu "
            "center=(%.3f %.3f %.3f) drop=(%.3f %.3f %.3f) "
            "size=(%.3f %.3f %.3f) coverage=%.2f interior=%.2f score=%.2f "
            "mode=%s reason=%s",
            cluster_id, cluster.indices.size(), candidate.center_floor.x(),
            candidate.center_floor.y(), candidate.center_floor.z(),
            candidate.drop_start.x(), candidate.drop_start.y(),
            candidate.drop_start.z(), candidate.width, candidate.depth,
            candidate.height, candidate.coverage, candidate.interior_fill,
            candidate.score,
            candidate.partial_rim ? "partial_rim" : "full",
            safety_reject.c_str());
        if (reject_summary.empty()) {
          reject_summary = safety_reject;
        }
        ++cluster_id;
        continue;
      }

      RCLCPP_INFO(
          get_logger(),
          "[container_drop_tf] cluster[%d] candidato: puntos=%zu "
          "center=(%.3f %.3f %.3f) drop=(%.3f %.3f %.3f) "
          "size=(%.3f %.3f %.3f) coverage=%.2f interior=%.2f score=%.2f "
          "mode=%s",
          cluster_id, cluster.indices.size(), candidate.center_floor.x(),
          candidate.center_floor.y(), candidate.center_floor.z(),
          candidate.drop_start.x(), candidate.drop_start.y(),
          candidate.drop_start.z(), candidate.width, candidate.depth,
          candidate.height, candidate.coverage, candidate.interior_fill,
          candidate.score,
          candidate.partial_rim ? "partial_rim" : "full");

      if (!found || candidate.score > best.score) {
        best = candidate;
        found = true;
      }
      ++cluster_id;
    }

    if (!found) {
      error_out = "no_container_candidate clusters=" +
                  std::to_string(clusters.size()) +
                  " first_reject=" + reject_summary;
      return false;
    }

    return true;
  }

  bool isCandidateInsideYoloGate(const Candidate &candidate,
                                 const YoloGateBounds &gate,
                                 std::string &reject_reason) const {
    const float margin =
        static_cast<float>(get_parameter("yolo_gate_margin_m").as_double());
    const float x = candidate.center_floor.x();
    const float y = candidate.center_floor.y();
    if (x < gate.min_x - margin || x > gate.max_x + margin ||
        y < gate.min_y - margin || y > gate.max_y + margin) {
      reject_reason = "outside_yolo_gate class=" + gate.class_name +
                      " center=(" + std::to_string(x) + "," +
                      std::to_string(y) + ") gate_x=[" +
                      std::to_string(gate.min_x) + "," +
                      std::to_string(gate.max_x) + "] gate_y=[" +
                      std::to_string(gate.min_y) + "," +
                      std::to_string(gate.max_y) + "] margin=" +
                      std::to_string(margin);
      return false;
    }
    return true;
  }

  bool evaluateCluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                       const pcl::PointIndices &indices, float floor_z,
                       Candidate &candidate,
                       std::string &reject_reason) const {
    // Primero calculamos un bounding box orientado en XY usando PCA. Asi una
    // caja rotada frente al robot sigue teniendo ancho/profundidad correctos.
    std::vector<Eigen::Vector3f> points;
    points.reserve(indices.indices.size());
    for (const int idx : indices.indices) {
      const auto &pt = cloud->points[idx];
      points.emplace_back(pt.x, pt.y, pt.z);
    }

    Eigen::Vector2f mean(0.0f, 0.0f);
    float min_z = std::numeric_limits<float>::max();
    float max_z = -std::numeric_limits<float>::max();
    for (const auto &p : points) {
      mean += Eigen::Vector2f(p.x(), p.y());
      min_z = std::min(min_z, p.z());
      max_z = std::max(max_z, p.z());
    }
    mean /= static_cast<float>(points.size());

    Eigen::Matrix2f cov = Eigen::Matrix2f::Zero();
    for (const auto &p : points) {
      const Eigen::Vector2f d = Eigen::Vector2f(p.x(), p.y()) - mean;
      cov += d * d.transpose();
    }
    cov /= static_cast<float>(points.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(cov);
    Eigen::Vector2f axis_x = solver.eigenvectors().col(1).normalized();
    if (axis_x.x() < 0.0f) {
      axis_x = -axis_x;
    }
    Eigen::Vector2f axis_y(-axis_x.y(), axis_x.x());

    float min_lx = std::numeric_limits<float>::max();
    float max_lx = -std::numeric_limits<float>::max();
    float min_ly = std::numeric_limits<float>::max();
    float max_ly = -std::numeric_limits<float>::max();
    for (const auto &p : points) {
      const Eigen::Vector2f rel = Eigen::Vector2f(p.x(), p.y()) - mean;
      const float lx = rel.dot(axis_x);
      const float ly = rel.dot(axis_y);
      min_lx = std::min(min_lx, lx);
      max_lx = std::max(max_lx, lx);
      min_ly = std::min(min_ly, ly);
      max_ly = std::max(max_ly, ly);
    }

    const float width = max_lx - min_lx;
    const float depth = max_ly - min_ly;
    const float height = max_z - floor_z;
    const float min_width = static_cast<float>(
        get_parameter("container_min_width_m").as_double());
    const float max_width = static_cast<float>(
        get_parameter("container_max_width_m").as_double());
    const float min_depth = static_cast<float>(
        get_parameter("container_min_depth_m").as_double());
    const float max_depth = static_cast<float>(
        get_parameter("container_max_depth_m").as_double());
    const float min_height =
        static_cast<float>(get_parameter("object_min_height_m").as_double());
    const float max_height =
        static_cast<float>(get_parameter("object_max_height_m").as_double());

    const float partial_min_depth = static_cast<float>(
        get_parameter("partial_rim_min_depth_m").as_double());
    const auto partial_min_points =
        static_cast<std::size_t>(get_parameter("partial_rim_min_points").as_int());
    const bool partial_rim_candidate =
        width >= min_width && width <= max_width && depth >= partial_min_depth &&
        depth < min_depth && height >= min_height && height <= max_height &&
        indices.indices.size() >= partial_min_points;

    if ((width < min_width || width > max_width || depth < min_depth ||
         depth > max_depth || height < min_height || height > max_height) &&
        !partial_rim_candidate) {
      reject_reason = "size_out_of_range width=" + std::to_string(width) +
                      " depth=" + std::to_string(depth) +
                      " height=" + std::to_string(height);
      return false;
    }

    ShapeScore shape;
    if (partial_rim_candidate) {
      shape.coverage = 0.65f;
      shape.interior_fill = 0.0f;
    } else {
      shape = scoreClosedFootprint(points, mean, axis_x, axis_y, min_lx, max_lx,
                                   min_ly, max_ly, floor_z, max_z);
    }

    const float min_coverage =
        static_cast<float>(get_parameter("min_wall_coverage").as_double());
    const float max_fill = static_cast<float>(
        get_parameter("max_interior_fill_ratio").as_double());
    const bool reject_full_containers =
        get_parameter("reject_full_containers").as_bool();
    const float full_fill = static_cast<float>(
        get_parameter("full_container_fill_ratio").as_double());
    const float full_min_coverage = static_cast<float>(
        get_parameter("full_container_min_coverage").as_double());
    if (!partial_rim_candidate && reject_full_containers &&
        shape.coverage >= full_min_coverage &&
        shape.interior_fill >= full_fill) {
      reject_reason = "container_full coverage=" +
                      std::to_string(shape.coverage) +
                      " interior_fill=" +
                      std::to_string(shape.interior_fill);
      return false;
    }

    if (!partial_rim_candidate &&
        (shape.coverage < min_coverage || shape.interior_fill > max_fill)) {
      reject_reason = "shape_rejected coverage=" +
                      std::to_string(shape.coverage) +
                      " interior_fill=" +
                      std::to_string(shape.interior_fill);
      return false;
    }

    const Eigen::Vector2f center_xy =
        mean + axis_x * ((min_lx + max_lx) * 0.5f) +
        axis_y * ((min_ly + max_ly) * 0.5f);

    const float clearance =
        static_cast<float>(get_parameter("start_clearance_m").as_double());
    const float drop_x_offset =
        static_cast<float>(get_parameter("drop_x_offset_m").as_double());
    const float drop_y_offset =
        static_cast<float>(get_parameter("drop_y_offset_m").as_double());
    const float partial_drop_x_offset = static_cast<float>(
        get_parameter("partial_rim_drop_x_offset_m").as_double());
    const bool override_drop_y = get_parameter("override_drop_y").as_bool();
    const float drop_y_override =
        static_cast<float>(get_parameter("drop_y_override_m").as_double());
    const bool override_drop_z = get_parameter("override_drop_z").as_bool();
    const float drop_z_override =
        static_cast<float>(get_parameter("drop_z_override_m").as_double());
    // La esfera se calcula desde el centro detectado del cluster, luego se
    // corrige con offsets calibrables para colocarla dentro de la cubeta real.
    float drop_y = center_xy.y() + drop_y_offset;
    if (override_drop_y) {
      drop_y = drop_y_override;
    }
    // El punto de drop queda sobre el centro del hueco, un poco arriba del
    // borde mas alto para iniciar la maniobra sin chocar con el contenedor.
    const float final_drop_x =
        center_xy.x() + drop_x_offset +
        (partial_rim_candidate ? partial_drop_x_offset : 0.0f);

    candidate.center_floor = Eigen::Vector3f(center_xy.x(), center_xy.y(),
                                             floor_z);
    const float final_drop_z =
        override_drop_z ? drop_z_override : max_z + clearance;
    candidate.drop_start = Eigen::Vector3f(final_drop_x, drop_y, final_drop_z);
    candidate.axis_x = Eigen::Vector3f(axis_x.x(), axis_x.y(), 0.0f);
    candidate.axis_y = Eigen::Vector3f(axis_y.x(), axis_y.y(), 0.0f);
    candidate.axis_z = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    candidate.width = width;
    candidate.depth = depth;
    candidate.height = height;
    candidate.coverage = shape.coverage;
    candidate.interior_fill = shape.interior_fill;
    candidate.partial_rim = partial_rim_candidate;
    candidate.score = 2.0f * shape.coverage - shape.interior_fill +
                      0.4f * width + 0.3f * depth - 0.35f * height -
                      0.35f * std::fabs(center_xy.y()) -
                      0.15f * std::hypot(center_xy.x(), center_xy.y());
    if (partial_rim_candidate) {
      candidate.score -= 0.25f;
    }
    return true;
  }

  ShapeScore scoreClosedFootprint(
      const std::vector<Eigen::Vector3f> &points, const Eigen::Vector2f &mean,
      const Eigen::Vector2f &axis_x, const Eigen::Vector2f &axis_y,
      float min_lx, float max_lx, float min_ly, float max_ly, float floor_z,
      float max_z) const {
    // Medimos si el cluster parece un perimetro cerrado:
    // - coverage: cuantos bins del borde tienen puntos altos,
    // - interior_fill: cuantos puntos altos hay dentro del centro.
    // Un contenedor bueno tiene coverage alto e interior_fill bajo.
    const float width = max_lx - min_lx;
    const float depth = max_ly - min_ly;
    const float rim_band =
        static_cast<float>(get_parameter("rim_band_m").as_double());
    const float rim_min_z = floor_z + std::max(0.07f, 0.45f * (max_z - floor_z));

    constexpr int side_bins = 12;
    std::vector<bool> left(side_bins, false);
    std::vector<bool> right(side_bins, false);
    std::vector<bool> front(side_bins, false);
    std::vector<bool> back(side_bins, false);
    int interior_high_points = 0;
    int interior_points = 0;

    for (const auto &p : points) {
      const Eigen::Vector2f rel = Eigen::Vector2f(p.x(), p.y()) - mean;
      const float lx = rel.dot(axis_x);
      const float ly = rel.dot(axis_y);
      const bool high_enough = p.z() >= rim_min_z;

      const bool in_interior = lx > min_lx + 0.30f * width &&
                               lx < max_lx - 0.30f * width &&
                               ly > min_ly + 0.30f * depth &&
                               ly < max_ly - 0.30f * depth;
      if (in_interior) {
        ++interior_points;
        if (high_enough) {
          ++interior_high_points;
        }
      }

      if (!high_enough) {
        continue;
      }

      const int bx = std::clamp(
          static_cast<int>(((lx - min_lx) / std::max(width, 0.001f)) *
                           static_cast<float>(side_bins)),
          0, side_bins - 1);
      const int by = std::clamp(
          static_cast<int>(((ly - min_ly) / std::max(depth, 0.001f)) *
                           static_cast<float>(side_bins)),
          0, side_bins - 1);

      if (lx < min_lx + rim_band) {
        left[by] = true;
      }
      if (lx > max_lx - rim_band) {
        right[by] = true;
      }
      if (ly < min_ly + rim_band) {
        front[bx] = true;
      }
      if (ly > max_ly - rim_band) {
        back[bx] = true;
      }
    }

    const auto count_true = [](const std::vector<bool> &bins) {
      return std::count(bins.begin(), bins.end(), true);
    };

    const float total_coverage =
        static_cast<float>(count_true(left) + count_true(right) +
                           count_true(front) + count_true(back)) /
        static_cast<float>(side_bins * 4);
    const float interior_fill =
        interior_points == 0
            ? 0.0f
            : static_cast<float>(interior_high_points) /
                  static_cast<float>(interior_points);
    return {total_coverage, interior_fill};
  }

  Eigen::Vector3f centroid(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) const {
    Eigen::Vector4f c4;
    pcl::compute3DCentroid(*cloud, c4);
    return Eigen::Vector3f(c4.x(), c4.y(), c4.z());
  }

  bool validateDropStartPose(const Eigen::Vector3f &start_point,
                             std::string &error_out) {
    // Seguridad geometrica en base_link. Evita publicar puntos detras del robot,
    // demasiado laterales, demasiado altos o fuera del alcance esperado.
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
      error_out = "unsafe_container_drop_start frame=" +
                  validation_frame_id_ + " x=" + std::to_string(x) +
                  " y=" + std::to_string(y) + " z=" + std::to_string(z) +
                  " dist_xy=" + std::to_string(distance_xy);
      return false;
    }
    return true;
  }

  bool transformCandidateToFixed(Candidate &candidate,
                                 std::string &error_out) {
    // El detector trabaja en base_link, pero las TF/markers se publican en
    // fixed_frame_id para quedar disponibles en el arbol global de TF.
    if (fixed_frame_id_.empty()) {
      error_out = "fixed_frame_id_empty";
      return false;
    }

    if (fixed_frame_id_ == validation_frame_id_) {
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

      candidate.center_floor = rotation * candidate.center_floor + translation;
      candidate.drop_start = rotation * candidate.drop_start + translation;
      candidate.axis_x = (rotation * candidate.axis_x).normalized();
      candidate.axis_y = (rotation * candidate.axis_y).normalized();
      candidate.axis_z = (rotation * candidate.axis_z).normalized();
      return true;
    } catch (const tf2::TransformException &ex) {
      error_out = "fixed_frame_transform_failed target=" + fixed_frame_id_ +
                  " source=" + validation_frame_id_ + " error=" + ex.what();
      return false;
    }
  }

  void publishTransforms(const Candidate &candidate) {
    // Publicamos dos frames:
    // container_frame: centro del contenedor en el piso.
    // container_drop_start: punto sobre el hueco donde iniciara el drop.
    Eigen::Matrix3f rotation;
    rotation.col(0) = candidate.axis_x;
    rotation.col(1) = candidate.axis_y;
    rotation.col(2) = candidate.axis_z;
    Eigen::Quaternionf q(rotation);
    q.normalize();

    geometry_msgs::msg::TransformStamped container_tf;
    container_tf.header.frame_id = fixed_frame_id_;
    container_tf.header.stamp = now();
    container_tf.child_frame_id = container_frame_id_;
    container_tf.transform.translation.x = candidate.center_floor.x();
    container_tf.transform.translation.y = candidate.center_floor.y();
    container_tf.transform.translation.z = candidate.center_floor.z();
    container_tf.transform.rotation.x = q.x();
    container_tf.transform.rotation.y = q.y();
    container_tf.transform.rotation.z = q.z();
    container_tf.transform.rotation.w = q.w();

    const Eigen::Vector3f rel =
        candidate.drop_start - candidate.center_floor;
    geometry_msgs::msg::TransformStamped drop_tf;
    drop_tf.header.stamp = container_tf.header.stamp;
    drop_tf.header.frame_id = container_frame_id_;
    drop_tf.child_frame_id = drop_start_frame_id_;
    drop_tf.transform.translation.x = rel.dot(candidate.axis_x);
    drop_tf.transform.translation.y = rel.dot(candidate.axis_y);
    drop_tf.transform.translation.z = rel.dot(candidate.axis_z);
    drop_tf.transform.rotation.w = 1.0;

    tf_broadcaster_->sendTransform(container_tf);
    tf_broadcaster_->sendTransform(drop_tf);
  }

  void logCloudWindow(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                      const std::string &label) const {
    if (!cloud || cloud->empty()) {
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1500,
          "[container_drop_tf] ventana %s sin puntos validos", label.c_str());
      return;
    }

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();
    float max_z = -std::numeric_limits<float>::max();
    for (const auto &pt : cloud->points) {
      min_x = std::min(min_x, pt.x);
      min_y = std::min(min_y, pt.y);
      min_z = std::min(min_z, pt.z);
      max_x = std::max(max_x, pt.x);
      max_y = std::max(max_y, pt.y);
      max_z = std::max(max_z, pt.z);
    }

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "[container_drop_tf] ventana %s puntos=%zu x=[%.3f %.3f] "
        "y=[%.3f %.3f] z=[%.3f %.3f]",
        label.c_str(), cloud->size(), min_x, max_x, min_y, max_y, min_z,
        max_z);
  }

  void resetDropPointsFile() {
    std::ofstream file(drop_points_file_, std::ios::trunc);
    if (!file.is_open()) {
      RCLCPP_WARN(get_logger(),
                  "[container_drop_tf] no se pudo reiniciar archivo: %s",
                  drop_points_file_.c_str());
      return;
    }
    file << "id,frame,x,y,z\n";
  }

  void saveDropPoint(const Eigen::Vector3f &start_point) {
    saved_drop_points_.clear();
    SavedDropPoint saved;
    saved.id = next_drop_point_id_++;
    saved.point = start_point;
    saved_drop_points_.push_back(saved);

    std::ofstream file(drop_points_file_, std::ios::app);
    if (!file.is_open()) {
      RCLCPP_WARN(get_logger(),
                  "[container_drop_tf] no se pudo escribir archivo: %s",
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
      marker.ns = "container_drop";
      marker.id = saved.id + 1;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = saved.point.x();
      marker.pose.position.y = saved.point.y();
      marker.pose.position.z = saved.point.z();
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.07;
      marker.scale.y = 0.07;
      marker.scale.z = 0.07;
      marker.color.r = 1.0f;
      marker.color.g = 0.2f;
      marker.color.b = 0.0f;
      marker.color.a = 0.9f;
      marker.lifetime = rclcpp::Duration(0, 0);
      markers.markers.push_back(marker);
    }

    marker_pub_->publish(markers);
    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[container_drop_tf] esferas seguras publicadas: count=%zu frame=%s "
        "topic=/vision/camera/drop_markers",
        saved_drop_points_.size(), fixed_frame_id_.c_str());
  }

  void publishDebugCloud(const pcl::PointCloud<pcl::PointXYZ> &cloud,
                         const std::string &frame_id) const {
    if (!roi_debug_cloud_pub_) {
      return;
    }
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = frame_id;
    msg.header.stamp = now();
    roi_debug_cloud_pub_->publish(msg);
  }

  std::string pointcloud_topic_;
  std::string fixed_frame_id_;
  std::string validation_frame_id_;
  std::string container_frame_id_;
  std::string drop_start_frame_id_;
  std::string drop_points_file_;
  std::string cloud_filter_enable_service_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      roi_debug_cloud_pub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_srv_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr cloud_filter_enable_client_;
  rclcpp::Client<vision_interfaces::srv::GetDetections>::SharedPtr
      yolo_client_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::vector<SavedDropPoint> saved_drop_points_;
  rclcpp::Client<vision_interfaces::srv::GetDetections>::SharedFuture
      yolo_future_;
  rclcpp::Time yolo_request_stamp_;
  ImageRoi selected_yolo_roi_;
  ImageRoi yolo_gate_roi_;
  int next_drop_point_id_{0};
  int yolo_roi_failure_count_{0};
  bool drop_point_saved_this_activation_{false};
  bool yolo_request_pending_{false};
  bool yolo_request_failed_{false};
  bool has_selected_yolo_roi_{false};
  bool has_yolo_gate_roi_{false};
  bool container_rejected_this_activation_{false};
  bool active_{false};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ContainerDropTfNode>());
  rclcpp::shutdown();
  return 0;
}
