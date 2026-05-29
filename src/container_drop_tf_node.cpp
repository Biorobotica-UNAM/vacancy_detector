#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
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
    declare_parameter<double>("rim_band_m", 0.055);
    declare_parameter<double>("min_wall_coverage", 0.45);
    declare_parameter<double>("max_interior_fill_ratio", 0.35);
    declare_parameter<double>("start_clearance_m", 0.10);
    declare_parameter<double>("drop_x_offset_m", 0.0);
    declare_parameter<double>("drop_y_offset_m", 0.0);
    declare_parameter<bool>("override_drop_y", false);
    declare_parameter<double>("drop_y_override_m", 0.0);
    declare_parameter<double>("cluster_tolerance_m", 0.045);
    declare_parameter<int>("cluster_min_points", 80);
    declare_parameter<int>("cluster_max_points", 12000);
    declare_parameter<double>("safe_x_min_m", 0.15);
    declare_parameter<double>("safe_x_max_m", 1.35);
    declare_parameter<double>("safe_abs_y_max_m", 0.85);
    declare_parameter<double>("safe_z_min_m", 0.08);
    declare_parameter<double>("safe_z_max_m", 1.10);
    declare_parameter<double>("safe_dist_xy_max_m", 1.50);
    declare_parameter<bool>("active_on_start", false);

    pointcloud_topic_ = get_parameter("pointcloud_topic").as_string();
    fixed_frame_id_ = get_parameter("fixed_frame_id").as_string();
    validation_frame_id_ = get_parameter("validation_frame_id").as_string();
    container_frame_id_ = get_parameter("container_frame_id").as_string();
    drop_start_frame_id_ = get_parameter("drop_start_frame_id").as_string();
    drop_points_file_ = get_parameter("drop_points_file").as_string();
    active_ = get_parameter("active_on_start").as_bool();

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&ContainerDropTfNode::cloudCallback, this,
                  std::placeholders::_1));

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "camera/container_drop_markers",
        rclcpp::QoS(10).transient_local().reliable());

    resetDropPointsFile();

    enable_srv_ = create_service<std_srvs::srv::SetBool>(
        "container_drop_tf/enable",
        std::bind(&ContainerDropTfNode::enableCallback, this,
                  std::placeholders::_1, std::placeholders::_2));

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
  };

  struct ShapeScore {
    float coverage{0.0f};
    float interior_fill{1.0f};
  };

  struct SavedDropPoint {
    int id{0};
    Eigen::Vector3f point{0.0f, 0.0f, 0.0f};
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

    std::string error;
    if (!processCloud(*msg, error)) {
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
      publishMarkers();
    }

    response->success = true;
    response->message = active_ ? "container_drop_tf_enabled"
                                : "container_drop_tf_disabled";
    RCLCPP_INFO(get_logger(), "[container_drop_tf] %s",
                response->message.c_str());
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
    if (!transformCloudToValidation(input_msg.header.frame_id, cloud,
                                    validation_cloud, error_out)) {
      return false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    // La ROI reduce falsos positivos: buscamos contenedores bajos y cercanos,
    // no paredes, personas ni muebles fuera del alcance de drop.
    applyRoi(validation_cloud, roi_cloud);

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "[container_drop_tf] ROI puntos: entrada=%zu validation=%zu roi=%zu",
        cloud->size(), validation_cloud->size(), roi_cloud->size());

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
                             error_out)) {
      return false;
    }

    if (!validateDropStartPose(best.drop_start, error_out)) {
      return false;
    }

    Candidate fixed_best = best;
    // Publicamos las TF en fixed_frame con timestamp real para que el executor
    // pueda distinguir una deteccion fresca de una deteccion anterior.
    if (!transformCandidateToFixed(fixed_best, error_out)) {
      return false;
    }

    publishTransforms(fixed_best);
    saveDropPoint(fixed_best.drop_start);
    publishMarkers();
    drop_point_saved_this_activation_ = true;

    RCLCPP_INFO(
        get_logger(),
        "[container_drop_tf] TF %s publicada: base=(%.3f %.3f %.3f) "
        "size=(%.3f %.3f %.3f) coverage=%.2f interior=%.2f score=%.2f",
        drop_start_frame_id_.c_str(), best.drop_start.x(),
        best.drop_start.y(), best.drop_start.z(), best.width, best.depth,
        best.height, best.coverage, best.interior_fill, best.score);

    error_out.clear();
    return true;
  }

  bool transformCloudToValidation(
      const std::string &source_frame,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
      pcl::PointCloud<pcl::PointXYZ>::Ptr &validation_cloud,
      std::string &error_out) {
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
    // Filtro rectangular en base_link. Los parametros se pueden ajustar desde
    // launch si el robot debe buscar mas cerca/lejos o con otro campo lateral.
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
                       std::vector<pcl::PointIndices> &clusters) const {
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
                           std::string &error_out) {
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

      std::string safety_reject;
      if (!validateDropStartPose(candidate.drop_start, safety_reject)) {
        RCLCPP_INFO(
            get_logger(),
            "[container_drop_tf] cluster[%d] inseguro: puntos=%zu "
            "center=(%.3f %.3f %.3f) drop=(%.3f %.3f %.3f) "
            "size=(%.3f %.3f %.3f) coverage=%.2f interior=%.2f score=%.2f "
            "reason=%s",
            cluster_id, cluster.indices.size(), candidate.center_floor.x(),
            candidate.center_floor.y(), candidate.center_floor.z(),
            candidate.drop_start.x(), candidate.drop_start.y(),
            candidate.drop_start.z(), candidate.width, candidate.depth,
            candidate.height, candidate.coverage, candidate.interior_fill,
            candidate.score, safety_reject.c_str());
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
          "size=(%.3f %.3f %.3f) coverage=%.2f interior=%.2f score=%.2f",
          cluster_id, cluster.indices.size(), candidate.center_floor.x(),
          candidate.center_floor.y(), candidate.center_floor.z(),
          candidate.drop_start.x(), candidate.drop_start.y(),
          candidate.drop_start.z(), candidate.width, candidate.depth,
          candidate.height, candidate.coverage, candidate.interior_fill,
          candidate.score);

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

    if (width < min_width || width > max_width || depth < min_depth ||
        depth > max_depth || height < min_height || height > max_height) {
      reject_reason = "size_out_of_range width=" + std::to_string(width) +
                      " depth=" + std::to_string(depth) +
                      " height=" + std::to_string(height);
      return false;
    }

    const ShapeScore shape =
        scoreClosedFootprint(points, mean, axis_x, axis_y, min_lx, max_lx,
                             min_ly, max_ly, floor_z, max_z);

    const float min_coverage =
        static_cast<float>(get_parameter("min_wall_coverage").as_double());
    const float max_fill = static_cast<float>(
        get_parameter("max_interior_fill_ratio").as_double());
    if (shape.coverage < min_coverage || shape.interior_fill > max_fill) {
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
    const bool override_drop_y = get_parameter("override_drop_y").as_bool();
    const float drop_y_override =
        static_cast<float>(get_parameter("drop_y_override_m").as_double());
    float drop_y = center_xy.y() + drop_y_offset;
    if (override_drop_y) {
      drop_y = drop_y_override;
    }
    // El punto de drop queda sobre el centro del hueco, un poco arriba del
    // borde mas alto para iniciar la maniobra sin chocar con el contenedor.
    candidate.center_floor = Eigen::Vector3f(center_xy.x(), center_xy.y(),
                                             floor_z);
    candidate.drop_start =
        Eigen::Vector3f(center_xy.x() + drop_x_offset, drop_y,
                        max_z + clearance);
    candidate.axis_x = Eigen::Vector3f(axis_x.x(), axis_x.y(), 0.0f);
    candidate.axis_y = Eigen::Vector3f(axis_y.x(), axis_y.y(), 0.0f);
    candidate.axis_z = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    candidate.width = width;
    candidate.depth = depth;
    candidate.height = height;
    candidate.coverage = shape.coverage;
    candidate.interior_fill = shape.interior_fill;
    candidate.score = 2.0f * shape.coverage - shape.interior_fill +
                      0.4f * width + 0.3f * depth - 0.35f * height -
                      0.35f * std::fabs(center_xy.y()) -
                      0.15f * std::hypot(center_xy.x(), center_xy.y());
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
        "topic=/vision/camera/container_drop_markers",
        saved_drop_points_.size(), fixed_frame_id_.c_str());
  }

  std::string pointcloud_topic_;
  std::string fixed_frame_id_;
  std::string validation_frame_id_;
  std::string container_frame_id_;
  std::string drop_start_frame_id_;
  std::string drop_points_file_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_pub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_srv_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::vector<SavedDropPoint> saved_drop_points_;
  int next_drop_point_id_{0};
  bool drop_point_saved_this_activation_{false};
  bool active_{false};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ContainerDropTfNode>());
  rclcpp::shutdown();
  return 0;
}
