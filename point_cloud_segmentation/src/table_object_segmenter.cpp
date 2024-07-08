#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/common/distances.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl/common/pca.h>
//include for the service
#include <point_cloud_segmentation/FilterPCL.h>
#include <point_cloud_segmentation/SegmentGraspObject.h>
#include <point_cloud_segmentation/GetVisualData.h>

//include for finding Surface Normals
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>

//include for pcl visualization
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/common.h>

//opencv includes
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/CameraInfo.h>
// ransac includes
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <math.h>

// extract from pointcloud stuff
#include <pcl/filters/extract_indices.h>
// for outlier removal
#include <pcl/filters/statistical_outlier_removal.h>
// pcl filter
#include <pcl/filters/passthrough.h>
// voxel grid filter
#include <pcl/filters/voxel_grid.h>
// for clustering
#include <pcl/segmentation/extract_clusters.h>
// kd tree
#include <pcl/search/kdtree.h>

#define BACKGROUND_DIST  500.0
#define BACKGROUND_COLOR 0.0
#define UNDEFINED        0.0

// #define TEST_OBJECT_BB 1
//
#include <ros/console.h>




typedef sensor_msgs::PointCloud2 MsgCloud;
typedef pcl::PointXYZRGB PtXYZRGB;
typedef pcl::PointCloud<PtXYZRGB> PCXYZRGB;
typedef pcl::Normal PtNormals;
typedef pcl::PointCloud<PtNormals> PCNormals;
typedef pcl::PointXYZ PtXYZ;
typedef pcl::PointCloud<PtXYZ> PCXYZ;
typedef pcl::search::KdTree<PtXYZRGB>::Ptr KdTreePtr;
using boost::shared_ptr;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
                                                  sensor_msgs::PointCloud2,
                                                  sensor_msgs::Image,
                                                  sensor_msgs::Image,
                                                  sensor_msgs::Image,
                                                  sensor_msgs::Image> SegmentSyncPolicy;


class ProtoObject
{
 public:
  PCXYZRGB cloud;
  PCNormals normals;
  Eigen::Vector4f centroid;
};

typedef std::deque<ProtoObject> ProtoObjects;

class TableObjectSegmenter
{
public:
  bool real_camera;
  bool pointcloud_from_client;
  std::string camera_frame_id;
  double filter_bias;
  double max_z;
  double min_z;
  double max_x;
  double min_x;
  double max_y;
  double min_y;
  double RANSAC_dist_thresh;
  double color_sim_min;
  bool create_srv;
  bool create_publisher;
  bool check_table_color;
  int table_r;
  int table_g;
  int table_b;
  int connected_componet_thresh;
  int close_size;
  bool debug_;
  bool segment_on_callback_;
  bool visualize_clusters_;
  bool visualize_bounding_box_;
  bool debug_srv_output_;
  bool keep_z_filter_organized;
  double ransac_prob;
  double cluster_tolerance_;
  int min_cluster_size_;
  int max_cluster_size_;
  PCXYZRGB cur_cloud_;
  PCXYZRGB cur_sd_cloud_;
  sensor_msgs::ImageConstPtr cur_depth_frame_;
  sensor_msgs::ImageConstPtr cur_rgb_frame_;
  sensor_msgs::ImageConstPtr cur_sd_depth_frame_;
  sensor_msgs::ImageConstPtr cur_sd_rgb_frame_;
  bool have_sensor_data_;
  int object_seg_seq_count_;
  double voxel_down_res_;
  shared_ptr<tf::TransformListener> tf_;
  std::string workspace_frame_;
  bool transform_cloud_;
  bool downsample_obj_cloud_;
  int num_normal_neighbors_;
  std::string pcd_camera_frame_;
public:
  TableObjectSegmenter(ros::NodeHandle n):n_(n), n_private_("~"), have_sensor_data_(false),
                                          object_seg_seq_count_(0),
                                          point_cloud_sub_(n, "point_cloud_topic", 1),
                                          point_cloud_sd_sub_(n, "sd_point_cloud_topic", 1),
                                          depth_img_sub_(n, "depth_img_topic", 1),
                                          rgb_img_sub_(n, "rgb_img_topic", 1),
                                          sd_depth_img_sub_(n, "sd_depth_img_topic", 1),
                                          sd_rgb_img_sub_(n, "sd_rgb_img_topic", 1),
                                          sync_(SegmentSyncPolicy(15), point_cloud_sub_,
                                                point_cloud_sd_sub_, depth_img_sub_, rgb_img_sub_,
                                                sd_depth_img_sub_, sd_rgb_img_sub_)
  {
    n_private_.param("real_camera", real_camera, true);
    n_private_.param("pointcloud_from_client", pointcloud_from_client, true);
    n_private_.param<std::string>("camera_frame_id", camera_frame_id, "");
    n_private_.param("filter_bias", filter_bias, 0.0);
    // positive filter_bias keeps table, negative removes more of the table and above
    //n_private_.param("keep_z_filter_organized",keep_z_filter_organized,true);
    n_private_.param("keep_z_filter_organized",keep_z_filter_organized,false);
    n_private_.param("debug",debug_,false);
    n_private_.param("debug_srv_output",debug_srv_output_,false);
    n_private_.param("segment_on_callback",segment_on_callback_,false);
    n_private_.param("visualize_clusters",visualize_clusters_,false);
    n_private_.param("visualize_bounding_box",visualize_bounding_box_,false);
    n_private_.param("min_x", min_x, -5.);
    n_private_.param("max_x", max_x, 5.);
    n_private_.param("min_y", min_y, -5.);
    n_private_.param("max_y", max_y, 5.);
    n_private_.param("min_z", min_z, -5.);
    n_private_.param("max_z", max_z, 5.);
    n_private_.param("voxel_downsample_res", voxel_down_res_, 0.005);
    n_private_.param("downsample_object_cloud", downsample_obj_cloud_, true);
    n_private_.param("num_normal_neighbors", num_normal_neighbors_, 8);
    n_private_.param("RANSAC_Distance_Threshold",RANSAC_dist_thresh,0.01);
    n_private_.param("create_srv",create_srv,true);
    n_private_.param("create_publisher",create_publisher,true);
    n_private_.param("check_table_color",check_table_color,true);
    n_private_.param("table_r",table_r,255);
    n_private_.param("table_g",table_g,255);
    n_private_.param("table_b",table_b,255);
    n_private_.param("color_sim_min",color_sim_min,.90);
    n_private_.param("connected_componet_thresh",connected_componet_thresh,100);
    n_private_.param("close_size",close_size,10);
    n_private_.param("RANSAC_prob", ransac_prob,1.0/3.0);
    n_private_.param("cluster_tolerance", cluster_tolerance_, 0.03);
    n_private_.param("min_cluster_size_", min_cluster_size_,100);
    n_private_.param("max_cluster_size_", max_cluster_size_,25000);
    n_private_.param("transform_cloud", transform_cloud_, false);
    n_private_.param<std::string>("workspace_frame", workspace_frame_, "world");
    tf_ = shared_ptr<tf::TransformListener>(new tf::TransformListener());

    if (!pointcloud_from_client)
    {
        sync_.registerCallback(&TableObjectSegmenter::segmentCallback, this);
    }

    if(create_publisher)
    {
      obj_cloud_pub_ = n_.advertise<MsgCloud>("/object_cloud",1);
      obj_pose_pub_ = n_.advertise<geometry_msgs::PoseStamped>("/object_pose",1);
      table_pub_ = n_.advertise<MsgCloud>("/table_cloud",1);
      filter_cloud_pub_ = n_.advertise<MsgCloud>("/filter_cloud",1);
    }

    if(create_srv)
    {
      filter_srv_ = n_.advertiseService("filter_PointCloud",
                                        &TableObjectSegmenter::filterSrv,this);
    }

    obj_seg_srv_ = n_.advertiseService("object_segmenter",
                                       &TableObjectSegmenter::objSegSrv, this);
    get_visual_data_srv_ = n_.advertiseService("get_visual_data",
                                       &TableObjectSegmenter::getVisualDataSrv, this);


    ROS_DEBUG("Filter_Bias=%lf",filter_bias);
    ROS_DEBUG("max_z=%lf",max_z);
    ROS_DEBUG("RANSAC_Distance_Threshold=%lf",RANSAC_dist_thresh);
    ROS_DEBUG("color_sim_min=%lf",color_sim_min);
    ROS_DEBUG("Finished initializing RANSAC_Filter");
  }
  bool filterSrv(point_cloud_segmentation::FilterPCL::Request &req,
                 point_cloud_segmentation::FilterPCL::Response &res)
  {
    /* This will use ransac to filter out anything that is not the table */
    // unpack the images
    ROS_INFO("RANSAC:Filter Srv Called");
    sensor_msgs::ImageConstPtr color_img_msg =
      boost::shared_ptr<sensor_msgs::Image const>
      (new sensor_msgs::Image(req.rgb_img));
    sensor_msgs::ImageConstPtr depth_img_msg =
     boost::shared_ptr<sensor_msgs::Image const>
      (new sensor_msgs::Image(req.depth_img));

    //define data structures
    PCXYZRGB::Ptr in_pcl(new PCXYZRGB);
    PCXYZRGB::Ptr filtered_pcl(new PCXYZRGB);
    PCNormals::Ptr normals_pcl(new PCNormals);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    MsgCloud send_pcl;
    MsgCloud send_normals;
    cv::Mat RGB_img;
    cv::Mat DPT_img;

    ROS_INFO("RANSAC:Unpacked Data");
    // get the input pointcloud
    pcl::fromROSMsg(req.pcl_input,*in_pcl);
    int width = in_pcl->width;
    int height = in_pcl->height;

    // convert the depth and RGB to cv::Mat
    DPT_img = cv_bridge::toCvShare(depth_img_msg,"mono16")->image;
    RGB_img = cv_bridge::toCvShare(color_img_msg,"bgr8")->image;

    // filter out things that are far away
    std::string fieldName = "z";
    filter(in_pcl,in_pcl,fieldName,0.0,max_z);

    // run the ransac algorithm
    findPlane(in_pcl,inliers,coefficients);

    // define the coefficients
    double a = coefficients->values[0];
    double b = coefficients->values[1];
    double c = coefficients->values[2];
    double d = coefficients->values[3];

    // find the normals
    findNormalsII(in_pcl,normals_pcl);

    // extract the indicies for inliers
    //remove_indices(in_pcl,inliers);

    // filter based on this plane
    filterPlane(in_pcl,filtered_pcl,normals_pcl,RGB_img,DPT_img,a,b,c,d);

    // set up and return the response
    pcl::toROSMsg(*filtered_pcl,send_pcl);
    pcl::toROSMsg(*normals_pcl,send_normals);
    sensor_msgs::ImagePtr RGBmsg = cv_bridge::CvImage(std_msgs::Header(),"rgb8",RGB_img).toImageMsg();
    sensor_msgs::ImagePtr DEPTHmsg = cv_bridge::CvImage(std_msgs::Header(),"mono16",DPT_img).toImageMsg();

    // set up the response
    res.pcl_filtered = send_pcl;
    res.pcl_normals = send_normals;
    res.rgb_filtered = *RGBmsg;
    res.depth_filtered = *DEPTHmsg;
    res.a = a;
    res.b = b;
    res.c = c;
    res.d = d;

    return(1);

  }

  void filterPlane(PCXYZRGB::Ptr in_pcl, PCXYZRGB::Ptr filtered_pcl,
		   PCNormals::Ptr pc_normals,
		   cv::Mat RGB_img, cv::Mat DPT_img,
		   double a, double b, double c, double d)
  {
    /* overloaded function to filter an dpt and rgb image, create a normal
       vector cv::Mat and filtered pcl */
    // set up the filtered pcl
    filtered_pcl->header = in_pcl->header;
    int width = in_pcl->width;
    int height = in_pcl->height;
    filtered_pcl->points.resize(width*height);
    filtered_pcl->height = height;
    filtered_pcl->width = width;
    filtered_pcl->is_dense = true;
    //*filtered_pcl = *tf_cloud;
    double x; double y; double z;int finitec = 0;
    int normalCount = 0;

    for(int u = 0; u<width; u++)
    {
      for(int v=0; v<height; v++)
      {
	x = in_pcl->at(u,v).x;
	y = in_pcl->at(u,v).y;
	z = in_pcl->at(u,v).z;
	if(a * x + b * y + c * z + d + filter_bias >= 0.0 && z < max_z)
	{
	  // set filtered image to in_pcl value
	  filtered_pcl->at(u,v) = in_pcl->at(u,v);
	  if(!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
	  {
	    finitec ++;
	    filtered_pcl->at(u,v).x = UNDEFINED;
	    filtered_pcl->at(u,v).y = UNDEFINED;
	    filtered_pcl->at(u,v).z = UNDEFINED;
	  }
	}
	else
	{
	  // set everything to background
	  filtered_pcl->at(u,v).x = BACKGROUND_DIST;
	  filtered_pcl->at(u,v).y = BACKGROUND_DIST;
	  filtered_pcl->at(u,v).z = BACKGROUND_DIST;
	  filtered_pcl->at(u,v).r = BACKGROUND_COLOR;
	  filtered_pcl->at(u,v).g = BACKGROUND_COLOR;
	  filtered_pcl->at(u,v).b = BACKGROUND_COLOR;
	  RGB_img.at<cv::Vec3b>(v,u) = cv::Vec3b(BACKGROUND_COLOR,BACKGROUND_COLOR,BACKGROUND_COLOR);
	  DPT_img.at<cv::Vec3b>(v,u) = cv::Vec3b(BACKGROUND_COLOR,BACKGROUND_COLOR,BACKGROUND_COLOR);
	  pc_normals->at(u,v).normal[0]=0.0;
	  pc_normals->at(u,v).normal[1]=0.0;
	  pc_normals->at(u,v).normal[2]=0.0;
	}
	if(!pcl_isfinite(pc_normals->at(u,v).normal[0]) ||
           !pcl_isfinite(pc_normals->at(u,v).normal[1]) ||
           !pcl_isfinite(pc_normals->at(u,v).normal[2]))
	{
	  pc_normals->at(u,v).normal[0] = 0.0;
	  pc_normals->at(u,v).normal[1] = 0.0;
	  pc_normals->at(u,v).normal[2] = 0.0;
	  normalCount++;
	}
      }
    }
    int total = (width * height);
    ROS_INFO("total found = %d",finitec);
    ROS_INFO("Normal count = %d",normalCount);
    ROS_INFO("percent=%d",total);

  }

  void findPlane(PCXYZRGB::Ptr in_pcl,
                 pcl::PointIndices::Ptr inliers,
                 pcl::ModelCoefficients::Ptr coefficients)
  {
    /* using RANSAC this will find the table plane */
    pcl::SACSegmentation<PtXYZRGB> seg;
    seg.setProbability(ransac_prob);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold(RANSAC_dist_thresh);
    seg.setInputCloud(in_pcl);
    seg.segment(*inliers,*coefficients);
  }

  /**
   * Use the plane to filter out points below and keep only the points above
   *
   * @param in_pcl Input cloud to remove table from
   * @param filtered_pcl Point remaining above the table after filtering (returned)
   * @param a plane parameter 1
   * @param b plane parameter 2
   * @param c plane parameter 3
   * @param d plane parameter 4
   */
  void filterPlane(PCXYZRGB::Ptr in_pcl, PCXYZRGB::Ptr filtered_pcl,
		   double a, double b, double c, double d)
  {
    // set up the filtered pcl
    filtered_pcl->header = in_pcl->header;
    filtered_pcl->is_dense = false;
    double x, y, z;
    for(int i=0; i < in_pcl->points.size(); ++i)
    {
      x = in_pcl->points[i].x;
      y = in_pcl->points[i].y;
      z = in_pcl->points[i].z;
      if(a * x + b * y + c * z + d + filter_bias >= 0.0 && z < max_z)
      {
        filtered_pcl->points.push_back(in_pcl->points[i]);
      }
    }
  }

  void findNormalsII(PCXYZRGB::Ptr in_pcl, PCNormals::Ptr normals)
  {
    /* Finds the normal vectors a pcl */
    pcl::IntegralImageNormalEstimation<PtXYZRGB, PtNormals> ne;
    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(in_pcl);
    ne.compute(*normals);
  }

  void filter(PCXYZRGB::Ptr pcl_in, PCXYZRGB::Ptr pcl_out,
	      std::string & fieldName, double fieldMin, double fieldMax)
  {
    //helper function for filtering on a PointCloud
    if (debug_)
      ROS_INFO("Doing single direction filter");
    pcl::PassThrough<PtXYZRGB> pass;
    pass.setInputCloud(pcl_in);
    pass.setKeepOrganized(keep_z_filter_organized);
    //pass.setUserFilterValue(BACKGROUND_DIST);
    pass.setFilterFieldName(fieldName);
    pass.setFilterLimits(fieldMin, fieldMax);
    //pass.setFilterLimitsNegative (true);
    pass.filter(*pcl_out);
    if (debug_)
    {
      std::ostringstream min_str_stream, max_str_stream;
      min_str_stream << fieldMin;
      max_str_stream << fieldMax;
      ROS_INFO("Ran %s direction filter, min: %s, max: %s", fieldName.c_str(), 
              min_str_stream.str().c_str(), max_str_stream.str().c_str());
    }
  }

  /**
   * Downsample a cloud using a voxel grid. Leaf size is controlled by voxel_down_res_
   *
   * @param cloud_in The cloud to downsample
   * @param cloud_down the downsampled cloud
   */
  void downsampleCloud(PCXYZRGB& cloud_in, PCXYZRGB& cloud_down)
  {
    pcl::VoxelGrid<PtXYZRGB> downsample_outliers;
    downsample_outliers.setInputCloud(cloud_in.makeShared());
    downsample_outliers.setLeafSize(voxel_down_res_, voxel_down_res_,
                                    voxel_down_res_);
    downsample_outliers.filter(cloud_down);
  }

  /**
   * Callback function for the point cloud topic
   *
   * @param msg_cloud incoming point cloud message
   */
  void segmentCallback(const sensor_msgs::PointCloud2ConstPtr& msg_cloud,
                       const sensor_msgs::PointCloud2ConstPtr& msg_sd_cloud,
                       const sensor_msgs::ImageConstPtr& depth_msg,
                       const sensor_msgs::ImageConstPtr& rgb_msg,
                       const sensor_msgs::ImageConstPtr& sd_depth_msg,
                       const sensor_msgs::ImageConstPtr& sd_rgb_msg)
  {
    MsgCloud send_cloud;
    PCXYZRGB::Ptr raw_cloud(new PCXYZRGB);
    PCXYZRGB::Ptr table_cloud(new PCXYZRGB);
    PCXYZRGB::Ptr obj_cloud(new PCXYZRGB);
    // define the needed data structures
    if(debug_)
    {
      ROS_DEBUG("Entered segmentCallback");
    }

    // Convert point cloud into PCL format
    pcl::fromROSMsg(*msg_cloud,*raw_cloud);
    pcd_camera_frame_ = msg_cloud->header.frame_id;
    // Store in a class variable for service callbacks
    pcl::copyPointCloud(*raw_cloud, cur_cloud_);
    cur_depth_frame_ = depth_msg;
    cur_rgb_frame_ = rgb_msg;

    if(real_camera)
    {
        cur_sd_depth_frame_ = sd_depth_msg;
        cur_sd_rgb_frame_ = sd_rgb_msg;
        // Convert sd point cloud into PCL format
        pcl::fromROSMsg(*msg_sd_cloud, cur_sd_cloud_);
    }

    if (debug_ && !have_sensor_data_)
      ROS_INFO_STREAM("Received sensor data");
    have_sensor_data_ = true;

    // Call segmentation
    if(segment_on_callback_)
    {
      ProtoObject main_obj;
      segmentTableAndSingleObject(raw_cloud, obj_cloud, table_cloud, main_obj);

      geometry_msgs::PoseStamped obj_pose;
      double bb_width, bb_height, bb_depth;
      findCloudBoundingBoxPCA(obj_cloud, obj_pose.pose,
                              bb_width, bb_height, bb_depth);

      // Publish object pose
      obj_pose.header.frame_id = obj_cloud->header.frame_id;
      obj_pose.header.stamp = ros::Time::now();
      obj_pose_pub_.publish(obj_pose);

      // Publish clouds over ros topics
      pcl::toROSMsg(*table_cloud, send_cloud);
      table_pub_.publish(send_cloud);
      pcl::toROSMsg(main_obj.cloud, send_cloud);
      obj_cloud_pub_.publish(send_cloud);
    }
  }

  /**
   * Service to perform table and object segmentation. It then computes a
   * few properties of the object to return as a GraspObject message type
   * in the service response
   *
   * @param req A request to segment. Currently contents are ignored.
   * @param res Object properties for use in grasping
   *
   * @return true if successful, false otherwise
   */
  bool objSegSrv(point_cloud_segmentation::SegmentGraspObject::Request &req,
                 point_cloud_segmentation::SegmentGraspObject::Response &res)
  {
   //compile this to output ROS DEBUG statements
   //if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug) ) {
   //ros::console::notifyLoggerLevelsChanged();
   //}

   ROS_DEBUG_STREAM("starting segmentation");
    //PCXYZRGB::Ptr raw_cloud = cur_cloud_.makeShared();
    PCXYZRGB::Ptr raw_cloud;
    pointcloud_from_client = true;
    if (!pointcloud_from_client)
    {
        // Ensure we have received a point cloud before processing
        if (!have_sensor_data_)
        {
          ROS_ERROR_STREAM("Requesting an object segmentation prior to receiving sensor data");
          return false;
        }
        raw_cloud = cur_cloud_.makeShared();
    } else {
        std::cout << req.client_cloud_path << std::endl;
        // PCXYZRGB::Ptr raw_cloud(new PCXYZRGB);
        raw_cloud = (new PCXYZRGB)->makeShared();
        pcl::io::loadPCDFile(req.client_cloud_path, *raw_cloud);
        // The client point cloud should be in camera frame
        pcd_camera_frame_ = camera_frame_id;
        raw_cloud->header.frame_id = pcd_camera_frame_;
        // raw_cloud = (new PCXYZRGB)->makeShared();
        // pcl::fromROSMsg(client_scene_cloud, *raw_cloud);
    }

    PCXYZRGB::Ptr table_cloud(new PCXYZRGB);
    PCXYZRGB::Ptr obj_cloud(new PCXYZRGB);

    // Transform point cloud to the world frame
    if (transform_cloud_)
    {
      tf_->waitForTransform(workspace_frame_, pcd_camera_frame_,
                            ros::Time::now(), ros::Duration(0.5));
      pcl_ros::transformPointCloud(workspace_frame_, *raw_cloud, *raw_cloud, *tf_);
    }

    // Call segmentation
    ProtoObject grasp_obj_seg;
    bool obj_found = false;
    if (real_camera) 
        obj_found = segmentTableAndSingleObject(raw_cloud, obj_cloud, table_cloud, grasp_obj_seg);
    else
        obj_found = segmentSingleObjectSim(raw_cloud, obj_cloud, grasp_obj_seg);
    if (!obj_found) {
        res.object_found = false;
        return true;
    }
    res.object_found = true;

    // Copy point clouds to send over
    pcl::toROSMsg(grasp_obj_seg.cloud, res.obj.cloud);
    pcl::toROSMsg(grasp_obj_seg.normals, res.obj.normals);

    // Get object pose details from PCA
    findCloudBoundingBoxPCA(obj_cloud, res.obj.pose,
                            res.obj.width, res.obj.height, res.obj.depth);

    // Copy details into response
    res.obj.header.frame_id = grasp_obj_seg.cloud.header.frame_id;
    res.obj.header.seq = object_seg_seq_count_++;
    res.obj.header.stamp = ros::Time::now();

    debug_srv_output_ = true;
    if (debug_srv_output_)
    {

        // HACK: For debugging service callback currently
        // Publish clouds over ros topics
        ROS_WARN_STREAM("Publishing table and object clouds");
        MsgCloud send_cloud, table_send_cloud;
        pcl::toROSMsg(*table_cloud, table_send_cloud);
        table_pub_.publish(table_send_cloud);
        pcl::toROSMsg(grasp_obj_seg.cloud, send_cloud);
        obj_cloud_pub_.publish(send_cloud);

        // HACK: Publishing object pose for debug purposes
        geometry_msgs::PoseStamped obj_pose;
        obj_pose.header = res.obj.header;
        obj_pose.header.stamp = send_cloud.header.stamp;
        obj_pose.pose = res.obj.pose;
        obj_pose_pub_.publish(obj_pose);
    }
    return true;
  }

  /**
   * Service to get visual data.
   *
   * @param req A request to get visual data.
   * @param res Respond with visual data.
   *
   * @return true if successful, false otherwise
   */
  bool getVisualDataSrv(point_cloud_segmentation::GetVisualData::Request &req,
                 point_cloud_segmentation::GetVisualData::Response &res)
  {
    // Ensure we have received a point cloud before processing
    if (!have_sensor_data_)
    {
      ROS_ERROR_STREAM("Request to get visual data prior to receiving sensor data");
      return false;
    }

    // Copy point clouds to send over
    pcl::toROSMsg(cur_cloud_, res.scene_cloud);
    
    res.scene_depth_img.header = cur_depth_frame_->header;
    res.scene_depth_img.height = cur_depth_frame_->height;
    res.scene_depth_img.width = cur_depth_frame_->width;
    res.scene_depth_img.encoding = cur_depth_frame_->encoding;
    res.scene_depth_img.is_bigendian = cur_depth_frame_->is_bigendian;
    res.scene_depth_img.step = cur_depth_frame_->step;
    res.scene_depth_img.data = cur_depth_frame_->data;

    res.scene_rgb_img.header = cur_rgb_frame_->header;
    res.scene_rgb_img.height = cur_rgb_frame_->height;
    res.scene_rgb_img.width = cur_rgb_frame_->width;
    res.scene_rgb_img.encoding = cur_rgb_frame_->encoding;
    res.scene_rgb_img.is_bigendian = cur_rgb_frame_->is_bigendian;
    res.scene_rgb_img.step = cur_rgb_frame_->step;
    res.scene_rgb_img.data = cur_rgb_frame_->data;

    if(real_camera)
    {
        pcl::toROSMsg(cur_sd_cloud_, res.scene_sd_cloud);

        res.scene_sd_depth_img.header = cur_sd_depth_frame_->header;
        res.scene_sd_depth_img.height = cur_sd_depth_frame_->height;
        res.scene_sd_depth_img.width = cur_sd_depth_frame_->width;
        res.scene_sd_depth_img.encoding = cur_sd_depth_frame_->encoding;
        res.scene_sd_depth_img.is_bigendian = cur_sd_depth_frame_->is_bigendian;
        res.scene_sd_depth_img.step = cur_sd_depth_frame_->step;
        res.scene_sd_depth_img.data = cur_sd_depth_frame_->data;

        res.scene_sd_rgb_img.header = cur_sd_rgb_frame_->header;
        res.scene_sd_rgb_img.height = cur_sd_rgb_frame_->height;
        res.scene_sd_rgb_img.width = cur_sd_rgb_frame_->width;
        res.scene_sd_rgb_img.encoding = cur_sd_rgb_frame_->encoding;
        res.scene_sd_rgb_img.is_bigendian = cur_sd_rgb_frame_->is_bigendian;
        res.scene_sd_rgb_img.step = cur_sd_rgb_frame_->step;
        res.scene_sd_rgb_img.data = cur_sd_rgb_frame_->data;
    }

    return true;
  }

  /**
   * Perform segmentation using RANSAC to find a table plane and single object on the table
   *
   * @param raw_cloud The input point cloud for filtering
   * @param obj_cloud (returned) The point cloud containing points from the object of interest
   * @param table_cloud (returned) The point cloud containing points from the table
   */
  bool segmentTableAndSingleObject(PCXYZRGB::Ptr& raw_cloud, PCXYZRGB::Ptr& obj_cloud,
                                   PCXYZRGB::Ptr& table_cloud, ProtoObject& main_obj)
  {
    PCXYZRGB::Ptr objs_cloud(new PCXYZRGB);
    PCXYZRGB::Ptr objs_cloud_down(new PCXYZRGB);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::PointIndices::Ptr keepers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);

    // filter the point cloud in world frame
    std::string fieldName = "z";
    filter(raw_cloud, raw_cloud, fieldName, min_z, max_z);
    fieldName = "x";
    filter(raw_cloud, raw_cloud, fieldName, min_x, max_x);
    fieldName = "y";
    filter(raw_cloud, raw_cloud, fieldName, min_y, max_y);

    if (debug_srv_output_)
    {
        MsgCloud raw_send_cloud;
        pcl::toROSMsg(*raw_cloud, raw_send_cloud);
        filter_cloud_pub_.publish(raw_send_cloud);
    }

    // run ransac and find a list of inliers indicies from tf_cloud
    findPlane(raw_cloud, inliers, coefficients);

    // Extract plane coefficients
    double a = coefficients->values[0];
    double b = coefficients->values[1];
    double c = coefficients->values[2];
    double d = coefficients->values[3];

    if(debug_)
    {
      ROS_INFO("Model Coefficients:%lfx+%lfy+%lfz+%lf=0", coefficients->values[0],
               coefficients->values[1], coefficients->values[2], coefficients->values[3]);
    }
    // if user wants to check color then check color duh?:)

    if(check_table_color)
    {
      check_color(raw_cloud,inliers,keepers);
    }

    pcl::copyPointCloud(*raw_cloud, inliers->indices, *table_cloud);

    // extract the indicies for inliers
    remove_indices(raw_cloud,inliers);

    // use the plane to filter the image

    filterPlane(raw_cloud,objs_cloud,a,b,c,d);
    if(keep_z_filter_organized)
    {
      addIndices(raw_cloud,objs_cloud,keepers);
    }
    else
    {
      addIndicesUnorganized(raw_cloud,objs_cloud,keepers);
    }
    if(debug_)
      ROS_INFO("Filtered Plane");

    ProtoObjects objs;
    bool obj_cluster_found = false;
    // Downsample remaining objects before clustering for speedup
    if (downsample_obj_cloud_)
    {
      downsampleCloud(*objs_cloud, *objs_cloud_down);
      obj_cluster_found = clusterProtoObjects(*objs_cloud_down,objs);
    }
    else
    {
      obj_cluster_found = clusterProtoObjects(*objs_cloud, objs);
    }

    if(debug_)
      ROS_INFO("Got Clusteres");

    if (!obj_cluster_found)
        return false;

     //if(visualize_clusters_)
     //{
     //  for(int i=0;i<objs.size();i++)
     //  {
     //    visPointCloud(objs[i].cloud);
     //    // visNormals(objs[i].cloud.makeShared(),
     //    //            objs[i].normals.makeShared());
     //  }
     //}

    // find the obj closests to the center of the table
    //Eigen::Vector4f table_centroid;
    //pcl::compute3DCentroid(*table_cloud,table_centroid);
    //int minIdx = 0;
    //double dist = 10000.0;
    //for(int i=0; i<objs.size();i++)
    //{
    //  double newDist = sqrDist(objs[i].centroid,table_centroid);
    //  if(newDist < dist)
    //  {
	//minIdx = i;
	//dist = newDist;
    //  }
    //}
    
    //find the largest object cluster
    //I assume the pcl Euclidean Clustering computes the clusters in the order of cluster sizes
    int minIdx = 0;
    // if(visualize_clusters_)
    // {
    //   visPointCloud(objs[minIdx].cloud);

    //   visNormals(objs[minIdx].cloud.makeShared(),
    //              objs[minIdx].normals.makeShared());
    // }

    // Select the cluster closest to the table centroid as the object of interest
    pcl::copyPointCloud(objs[minIdx].cloud, *obj_cloud);
    main_obj.centroid = objs[minIdx].centroid;
    pcl::copyPointCloud(objs[minIdx].cloud, main_obj.cloud);
    pcl::copyPointCloud(objs[minIdx].normals, main_obj.normals);
    if (debug_)
      ROS_INFO_STREAM("table_cloud Size:" << table_cloud->points.size());

    return true;
  }


  /**
   * Perform segmentation for single object on the table in simulation.
   *
   * @param raw_cloud The input point cloud for filtering
   * @param obj_cloud (returned) The point cloud containing points from the object of interest
   */
  bool segmentSingleObjectSim(PCXYZRGB::Ptr& raw_cloud, PCXYZRGB::Ptr& obj_cloud,
                              ProtoObject& main_obj)
  {
    PCXYZRGB::Ptr objs_cloud(new PCXYZRGB);
    PCXYZRGB::Ptr objs_cloud_down(new PCXYZRGB);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::PointIndices::Ptr keepers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);

    // filter the point cloud in world frame
    std::string fieldName = "z";
    filter(raw_cloud, obj_cloud, fieldName, min_z, max_z);
    fieldName = "x";
    filter(obj_cloud, obj_cloud, fieldName, min_x, max_x);
    fieldName = "y";
    filter(obj_cloud, obj_cloud, fieldName, min_y, max_y);

    if (debug_srv_output_)
    {
        MsgCloud obj_send_cloud;
        pcl::toROSMsg(*obj_cloud, obj_send_cloud);
        filter_cloud_pub_.publish(obj_send_cloud);
    }

    if (obj_cloud->empty())
    {
        ROS_INFO_STREAM("No object point cloud found!");
        return false;
    }
    
    pcl::copyPointCloud(*obj_cloud, main_obj.cloud);
    pcl::compute3DCentroid(*obj_cloud, main_obj.centroid);
    pcl::NormalEstimation<PtXYZRGB, PtNormals> ne;
    const KdTreePtr clusters_tree(new pcl::search::KdTree<PtXYZRGB>);
    clusters_tree->setInputCloud(obj_cloud->makeShared());
    PCNormals objects_normals;
    ne.setInputCloud(obj_cloud->makeShared());
    ne.setSearchMethod(clusters_tree);
    ne.setKSearch(num_normal_neighbors_);
    ne.compute(main_obj.normals);

    return true;
  }


  static inline double sqrDist(Eigen::Vector4f& a, Eigen::Vector4f& b)
  {
    const double dx = a[0]-b[0];
    const double dy = a[1]-b[1];
    const double dz = a[2]-b[2];
    return dx*dx+dy*dy+dz*dz;
  }

  bool clusterProtoObjects(PCXYZRGB& objects_cloud, ProtoObjects& objs)
  {
    std::vector<pcl::PointIndices> clusters;
    pcl::EuclideanClusterExtraction<PtXYZRGB> pcl_cluster;
    const KdTreePtr clusters_tree(new pcl::search::KdTree<PtXYZRGB>);
    clusters_tree->setInputCloud(objects_cloud.makeShared());
    std::vector<int> indx_holder;
    pcl::removeNaNFromPointCloud(objects_cloud,objects_cloud,indx_holder);
    if (debug_)
      ROS_INFO("Built Tree");
    pcl_cluster.setClusterTolerance(cluster_tolerance_);
    pcl_cluster.setMinClusterSize(min_cluster_size_);
    pcl_cluster.setMaxClusterSize(max_cluster_size_);
    pcl_cluster.setSearchMethod(clusters_tree);
    pcl_cluster.setInputCloud(objects_cloud.makeShared());
    pcl_cluster.extract(clusters);
    ROS_INFO_STREAM("Number of clusters found matching the given constraints: "
		     << clusters.size());

    if (clusters.empty())
    {
        ROS_ERROR_STREAM("No object cluster found!");
        return false;
    }

    // Find normals in the cluster tree
    pcl::NormalEstimation<PtXYZRGB, PtNormals> ne;
    PCNormals objects_normals;
    ne.setInputCloud(objects_cloud.makeShared());
    ne.setSearchMethod(clusters_tree);
    ne.setKSearch(num_normal_neighbors_);
    // ne.setRadiusSearch(0.03);
    ne.compute(objects_normals);

    for (unsigned int i = 0; i < clusters.size(); ++i)
    {
	// Create proto objects from the point cloud
	ProtoObject po;
	pcl::copyPointCloud(objects_cloud, clusters[i], po.cloud);
        // Extract associated normals
        pcl::copyPointCloud(objects_normals, clusters[i], po.normals);
	pcl::compute3DCentroid(po.cloud, po.centroid);
	objs.push_back(po);
    }

    return true;
  }

  void spin()
  { //ros::Rate loop_rate(1.0);
    while(n_.ok())
    {
      ros::spinOnce();
      //loop_rate.sleep();

    }
  }

  void check_color(PCXYZRGB::Ptr cloud,
		   pcl::PointIndices::Ptr& indices,
		   pcl::PointIndices::Ptr& keep_list)
  {
    /* checks the indices to ensure they match in table color before removal */
    int loc;
    PtXYZRGB temp;
    double color_dist;
    pcl::PointIndices::Ptr new_idxs(new pcl::PointIndices);
    for(int i=0;i<indices->indices.size();i++)
    {
      loc = indices->indices[i];
      temp = cloud->points[loc];
      // find the L2 distance in the color space
      color_dist = sqrt(static_cast<double>(
       	(static_cast<int>(temp.r) - table_r) * (static_cast<int>(temp.r) - table_r) +
	(static_cast<int>(temp.g) - table_g) * (static_cast<int>(temp.g) - table_g) +
	(static_cast<int>(temp.b) - table_b) * (static_cast<int>(temp.b) - table_b)));
      color_dist = exp(-color_dist/255.0); // take exp(-) to make a simularity measure
      //ROS_INFO("color:[%d,%d,%d],color_dist:%lf",static_cast<int>(temp.r),static_cast<int>(temp.g),static_cast<int>(temp.b),color_dist);
      if(color_dist >= color_sim_min)
      {
	// add pack if simular enough to the table color
	new_idxs->indices.push_back(loc);
      }
      else
      {
	// add that to the keep_list
	keep_list->indices.push_back(loc);
      }
    }
    indices = new_idxs;
  }

  void addIndices(PCXYZRGB::Ptr origin_cloud,
		     PCXYZRGB::Ptr add_cloud,
		     pcl::PointIndices::Ptr add_list)
  {
    /* adds the points from origin_cloud to add_cloud at the indicies */
    for(int i=0;i<add_list->indices.size();i++)
    {
      add_cloud->points[add_list->indices[i]] =
	origin_cloud->points[add_list->indices[i]];
    }
  }
  void addIndicesUnorganized(PCXYZRGB::Ptr origin_cloud,
			     PCXYZRGB::Ptr add_cloud,
			     pcl::PointIndices::Ptr add_list)
  {
    for(int i=0; i<add_list->indices.size();++i)
    {
      add_cloud->points.push_back(origin_cloud->points[add_list->indices[i]]);
    }
  }

  void remove_indices(PCXYZRGB::Ptr out_cloud,
		       pcl::PointIndices::Ptr inliers)
  {
    /* removes the list of indicies from the pointcloud
       this function does keep the cloud organized */
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    //extract.setInputCloud(tf_cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filterDirectly(out_cloud);
    if (debug_)
      ROS_INFO("Removed inlier indices");
  }

  void visNormals(PCXYZRGB::Ptr color, PCNormals::Ptr normals)
  {
    /* visualizes a pointcloud with normals */
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor(0.0,0.0,0.8);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(color);
    viewer.addPointCloud<PtXYZRGB>(color, rgb, "object_cloud");
    viewer.addPointCloudNormals<PtXYZRGB,PtNormals>(color,normals,2);

    while(!viewer.wasStopped ())
    {
      viewer.spinOnce();
    }
  }

  void visNormalsAsColors(PCXYZRGB::Ptr color, PCNormals::Ptr normals)
  {
    /* visualizes a pointcloud with normals */
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor(0.0,0.0,0.0);
    // Color points based on their normal, need to scale the normals
    PCXYZRGB norm_colors;
    pcl::copyPointCloud(*color, norm_colors);
    for (int i = 0; i < color->points.size(); ++i)
    {
      norm_colors.at(i).r = normals->at(i).normal[0]*255;
      norm_colors.at(i).g = normals->at(i).normal[1]*255;
      norm_colors.at(i).b = normals->at(i).normal[2]*255;
    }
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(norm_colors.makeShared());
    viewer.addPointCloud<PtXYZRGB>(color, rgb, "object_cloud");
    // viewer.addPointCloudNormals<PtXYZRGB,PtNormals>(color,normals,1);

    while(!viewer.wasStopped ())
    {
      viewer.spinOnce();
    }
  }

  void visPointCloud(PCXYZRGB::Ptr in_cloud)
  {
    /* visualizes the inputed pointcloud */
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor(0.0,0.0,0.5);
    // viewer.addPointCloud<PtXYZRGB>(in_cloud);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(in_cloud);
    viewer.addPointCloud<PtXYZRGB>(in_cloud, rgb, "object_cloud");

    while(!viewer.wasStopped())
    {
      viewer.spinOnce();
    }
  }

  void visPointCloud(PCXYZRGB in_cloud)
  {
    /* visualizes the inputed pointcloud
     overloaded to deal with pointclouds not in ptr format */
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor(0.0,0.0,0.5);
    // viewer.addPointCloud<PtXYZRGB>(in_cloud.makeShared());
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(in_cloud.makeShared());
    viewer.addPointCloud<PtXYZRGB>(in_cloud.makeShared(), rgb, "object_cloud");

    while(!viewer.wasStopped())
    {
      viewer.spinOnce();
    }
  }

  /**
   * Compute the bounding box associated with an object cloud using PCA
   *
   * @param obj_color_cloud The cloud to bound
   * @param pose [output] The object's pose
   * @param width [output] The bounding box width (1st principal component range)
   * @param height [output] The bounding box height (2nd principal component range)
   * @param depth [output] The bounding box depth (3rd principal component range)
   */
  void findCloudBoundingBoxPCA(PCXYZRGB::Ptr& obj_color_cloud, geometry_msgs::Pose& pose,
                               double& width, double& height, double& depth)
  {
    Eigen::Vector3f eigen_values;
    Eigen::Matrix3f eigen_vectors;
    Eigen::Vector4f centroid;
    // Remove the RGB components from the input cloud
    PCXYZ obj_cloud;
    pcl::copyPointCloud(*obj_color_cloud, obj_cloud);
    pcl::PCA<PtXYZ> pca;
    try{
      pca.setInputCloud(obj_cloud.makeShared());
      ROS_DEBUG_STREAM("Getting mean");
      centroid = pca.getMean();
      ROS_DEBUG_STREAM("Getting eigen values");
      eigen_values = pca.getEigenValues();
      ROS_DEBUG_STREAM("Getting eigen vectors");
      eigen_vectors = pca.getEigenVectors();
    } catch(pcl::InitFailedException ife)
    {
      ROS_WARN_STREAM("Failed to compute PCA");
      ROS_WARN_STREAM("ife: " << ife.what());
    }
    // Ensure the coordinate system is right handed
    eigen_vectors.col(2) = eigen_vectors.col(0).cross(eigen_vectors.col(1));
    Eigen::Matrix4f proj_transform;
    proj_transform.block<3,3>(0,0) = eigen_vectors.transpose();
    proj_transform.block<3,1>(0,3) = -1.f * (proj_transform.block<3,3>(0,0) * centroid.head<3>());
    // Transform points into eigen space to find extrema and compute ranges
    PCXYZ proj;
    // pca.project(obj_cloud, proj);
    pcl::transformPointCloud(obj_cloud, proj, proj_transform);

    PtXYZ proj_min;
    PtXYZ proj_max;
    pcl::getMinMax3D(proj, proj_min, proj_max);
    width = fabs(proj_max.x-proj_min.x);
    height = fabs(proj_max.y-proj_min.y);
    depth = fabs(proj_max.z-proj_min.z);

    Eigen::Vector3f box_mean_eigenspace = 0.5f*(proj_max.getVector3fMap() +
                                                proj_min.getVector3fMap());

    // Compute the transform into the object's space
    Eigen::Quaternionf orientation(eigen_vectors);
    orientation.normalize();
    Eigen::Vector3f rectified_centroid = eigen_vectors*box_mean_eigenspace + \
                                         centroid.head<3>();
    pose.position.x = rectified_centroid[0];
    pose.position.y = rectified_centroid[1];
    pose.position.z = rectified_centroid[2];
    pose.orientation.x = orientation.x();
    pose.orientation.y = orientation.y();
    pose.orientation.z = orientation.z();
    pose.orientation.w = orientation.w();

    if (debug_)
      ROS_INFO_STREAM("width = " << width << "\theight = " << height <<
                      "\tdepth = " << depth);

    // if (visualize_bounding_box_)
    // {
    //   ROS_INFO_STREAM("eigen values:\n" << eigen_values);
    //   ROS_INFO_STREAM("eigen vectors\n" << eigen_vectors);
    //   // ROS_INFO_STREAM("e1 X e2 = " << eig3);
    //   ROS_INFO_STREAM("pca centroid:\n" << centroid);
    //   ROS_INFO_STREAM("box_mean:\n" << box_mean_eigenspace);
    //   ROS_INFO_STREAM("Recovered centroid:\n" << rectified_centroid);
    //   ROS_INFO_STREAM("Recovered orientation:\n"<< pose.orientation);
    //   /* visualizes the inputed pointcloud */
    //   pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    //   viewer.setBackgroundColor(0.0,0.0,0.7);
    //   pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(obj_color_cloud);
    //   viewer.addPointCloud<PtXYZRGB>(obj_color_cloud, rgb, "object_cloud");
    //   viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
    //                                           "object_cloud");

    //   viewer.addCube(rectified_centroid, orientation, width, height, depth);
    //   viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
		//                          pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
		// 			 "cube");
    //   while(!viewer.wasStopped())
    //   {
    //     viewer.spinOnce();
    //   }
    // }
  }

  protected:
  ros::NodeHandle n_;
  ros::NodeHandle n_private_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> point_cloud_sub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> point_cloud_sd_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_img_sub_;
  message_filters::Subscriber<sensor_msgs::Image> rgb_img_sub_;
  message_filters::Subscriber<sensor_msgs::Image> sd_depth_img_sub_;
  message_filters::Subscriber<sensor_msgs::Image> sd_rgb_img_sub_;
  message_filters::Synchronizer<SegmentSyncPolicy> sync_;
  ros::Publisher table_pub_;
  ros::Publisher filter_cloud_pub_;
  ros::Publisher obj_cloud_pub_;
  ros::Publisher obj_pose_pub_;
  ros::ServiceServer filter_srv_;
  ros::ServiceServer obj_seg_srv_;
  ros::ServiceServer get_visual_data_srv_;
};

/**
 * Generate a random, normalized quaternion
 *
 * @return The random quaternion
 */
Eigen::Quaternionf randomQuaternion()
{
  Eigen::Quaternionf q((rand()/(float)(RAND_MAX)),
                       (rand()/(float)(RAND_MAX)),
                       (rand()/(float)(RAND_MAX)),
                       (rand()/(float)(RAND_MAX)));
  q.normalize();
  return q;
}


void generateObjectCloud(PCXYZRGB::Ptr& obj_color_cloud, int num_points = 500)
{
  // Generate random object dimensions
  double radius_a = rand()/(float)(RAND_MAX)*0.5+0.05;
  double radius_b = rand()/(float)(RAND_MAX)*0.5+0.05;
  double object_height = rand()/(float)(RAND_MAX)*1.0+0.05;

  // Object centroid
  Eigen::Vector3f centroid(0.6,0.3,0.1);
  // Object orientation
  // Generate random rotation matrix
  Eigen::Quaternionf q = randomQuaternion();
  Eigen::Matrix3f rotation;
  rotation = q;
  ROS_INFO_STREAM("Object center = \n" << centroid);
  ROS_INFO_STREAM("Object orientation (q) = \n" << q.x() << "\n" << q.y()  << "\n" << q.z()  << "\n" << q.w());
  ROS_INFO_STREAM("Object orientation = \n" << rotation);
  obj_color_cloud->resize(num_points);
  for (int i = 0; i < num_points; ++i)
  {
    // Generate a random point on an elongated ellipse (cylinder with ellipse base)
    Eigen::Vector3f point_obj;
    // Get random theta between 0 and 2PI
    float theta = rand() / (float)(RAND_MAX) * M_PI*2.0;
    // ROS_INFO_STREAM("Generated theta = " << theta);
    point_obj[0] = radius_a*std::cos(theta);
    point_obj[1] = radius_b*std::sin(theta);
    // Generate random height on cylinder between -height/2 and height/2
    point_obj[2] = rand() / (float)(RAND_MAX) * object_height - object_height*0.5;
    // ROS_INFO_STREAM("Generated z = " << point_obj[2]);
    Eigen::Vector3f point_world = rotation*point_obj + centroid;
    obj_color_cloud->points[i].x = point_world[0];
    obj_color_cloud->points[i].y = point_world[1];
    obj_color_cloud->points[i].z = point_world[2];
    obj_color_cloud->points[i].r = 255;
    obj_color_cloud->points[i].g = 0;
    obj_color_cloud->points[i].b = 0;
    // ROS_INFO_STREAM("Adding point \n" << point_world);
  }
}

void testObjectRotation()
{
  ros::NodeHandle nh;
  TableObjectSegmenter segmenter(nh);
  // Generate synthetnic object pointcloud for testing
  PCXYZRGB::Ptr obj_cloud(new PCXYZRGB);
  geometry_msgs::Pose obj_pose;
  double width, height, depth;
  int seed = 1;
  srand(seed);
  std::cout << "seed " << seed;
  segmenter.visualize_bounding_box_ = true;
  segmenter.debug_ = true;

  while (true)
  {
    generateObjectCloud(obj_cloud, 500);
    segmenter.findCloudBoundingBoxPCA(obj_cloud, obj_pose, width, height, depth);
  }
}

int main(int arg, char** argv)
{
  //This will initialize everything
  ros::init(arg,argv,"tabletop_object_segmenter");
#ifdef TEST_OBJECT_BB
  testObjectRotation();
#else // TEST_OBJECT_BB
  ros::NodeHandle nh;
  TableObjectSegmenter segmenter(nh);
  segmenter.spin();
#endif // TEST_OBJECT_BB
}
