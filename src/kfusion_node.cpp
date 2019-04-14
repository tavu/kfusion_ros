#include <ros/ros.h>

#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/ChannelFloat32.h>
#include <geometry_msgs/Point32.h>

#include <image_transport/image_transport.h>
// #include <message_filters/subscriber.h>
// #include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <string.h>
// #include <sensor_msgs/image_encodings.h>
#include <kernels.h>

#include <tf/LinearMath/Matrix3x3.h>
//#include <tf/btMatrix3x3.h>
//#include <tf/LinearMath/btMatrix3x3.h>
//#include<tf/tf.h>
//#include <tf/transform_datatypes.h>

// #define CAM_INFO_TOPIC "/zed/depth/camera_info"
// #define RGB_TOPIC "/zed/rgb/image_raw_color"
// #define DEPTH_TOPIC "/zed/depth/depth_registered"

//#define CAM_INFO_TOPIC "/camera/depth/camera_info"
//#define DEPTH_TOPIC "/camera/depth/image"
//#define RGB_TOPIC "/camera/rgb/image_color"


#define CAM_INFO_TOPIC "/camera/depth/camera_info"
#define RGB_TOPIC "/camera/rgb/image_rect_color"
#define DEPTH_TOPIC "/camera/depth/image_rect"

#define PUB_VOLUME_TOPIC "/kfusion/volume_rendered"
#define PUB_ODOM_TOPIC "/kfusion/odom"
#define PUB_POINTS_TOPIC "/kfusion/pointCloud"

#define PUBLISH_POINT_RATE 10

typedef unsigned char uchar;

//KFusion Params
typedef struct
{
    int compute_size_ratio=1;
    int integration_rate=1;
    int rendering_rate = 1;
    int tracking_rate=1;
    uint3 volume_resolution = make_uint3(256,256,256);
    float3 volume_direction = make_float3(4,4,4);
    float3 volume_size = make_float3(8,8,8);
    std::vector<int> pyramid = {10,5,4};
    float mu = 0.1;
    float icp_threshold = 1e-03;
    
    uint2 inputSize;
    uint2 computationSize;
    float4 camera;
    
} kparams_t;

//Buffers
uint16_t *inputDepth=0;
float *inputDepthFl=0;
uchar3 *inputRGB;
uchar4 *depthRender;
uchar4 *trackRender;
uchar4 *volumeRender;

kparams_t params;
Kfusion *kfusion=nullptr;

int frame = 0;
int odom_delay;
int odom_rec=0;
int odom_counter = 0;
bool publish_volume=true;
bool publish_points=true;
int publish_points_rate;

//ros pub/subs
ros::Subscriber cam_info_sub;
ros::Publisher volume_pub;
ros::Publisher odom_pub ;
ros::Publisher points_pub;
nav_msgs::Odometry odom_;
Matrix4 T_B_P;
//functions
void initKFusion();
void publishVolume();
void publishOdom();
void publishPoints();

void readKfusionParams(ros::NodeHandle &n_p)
{   
    kparams_t par;
    n_p.getParam("compute_size_ratio",par.compute_size_ratio);
    n_p.getParam("integration_rate",par.integration_rate);
    n_p.getParam("rendering_rate",par.rendering_rate);
    n_p.getParam("tracking_rate",par.tracking_rate);
    
    std::vector<int> vol_res;
    n_p.getParam("volume_resolution",vol_res);
    par.volume_resolution = make_uint3((uint)vol_res[0],(uint)vol_res[1],(uint)vol_res[2]);    
    
    std::vector<float> vol_direct;
    n_p.getParam("volume_direction",vol_direct);
    par.volume_direction = make_float3(vol_direct[0],vol_direct[1],vol_direct[2]);    
    
    std::vector<float> vol_size;
    n_p.getParam("volume_size",vol_size);
    par.volume_size = make_float3(vol_size[0],vol_size[1],vol_size[2]);    
    
    n_p.getParam("pyramid",par.pyramid);  
    
    n_p.getParam("mu",par.mu);
    n_p.getParam("icp_threshold",par.icp_threshold);
    /*
    kparams_t par;
    n_p.getParam("compute_size_ratio",par.compute_size_ratio);
    n_p.getParam("integration_rate",par.integration_rate);
    n_p.getParam("rendering_rate",par.rendering_rate);
    n_p.getParam("tracking_rate",par.tracking_rate);
    n_p.getParam("volume_resolution",par.volume_resolution);
    n_p.getParam("volume_direction",par.volume_direction);
//     std::vector<double> pose_list;
    
    n_p.getParam("volume_size",par.volume_size);
    n_p.getParam("pyramid",par.pyramid);
    n_p.getParam("mu",par.mu);
    n_p.getParam("icp_threshold",par.icp_threshold);
    */
    params=par;
}

void odomCallback(const nav_msgs::Odometry &odom)
{
    ROS_INFO("odom %d",odom_rec);

    if(kfusion==nullptr)
        return;
    
    if(odom_rec!=odom_delay)
    {
        odom_rec++; 
        return;
    }
    ROS_INFO("odomsdf");
         
    odom_rec++; 


    if(odom_counter==0)
    {
        odom_counter++;
        odom_ = odom;

        return;
    }


   
    Matrix4 pose, pose_, delta_pose;
    


    //New Odometry
    tf::Quaternion q(odom.pose.pose.orientation.x,
                     odom.pose.pose.orientation.y,
                     odom.pose.pose.orientation.z,
                     odom.pose.pose.orientation.w);

    
   tf::Matrix3x3 rot_matrix(q);

    for(int i=0;i<3;i++)
    {
        tf::Vector3 vec=rot_matrix.getRow(i);
        pose.data[i].x=vec.getX();
        pose.data[i].y=vec.getY();
        pose.data[i].z=vec.getZ();
        pose.data[i].w=0;
    }
    pose.data[3].x=0;
    pose.data[3].y=0;
    pose.data[3].z=0;
    pose.data[3].w=1;


    pose.data[0].w=odom.pose.pose.position.x;
    pose.data[1].w=odom.pose.pose.position.y;
    pose.data[2].w=odom.pose.pose.position.z;


    

    //Old Odometry

    tf::Quaternion q_(odom_.pose.pose.orientation.x,
                     odom_.pose.pose.orientation.y,
                     odom_.pose.pose.orientation.z,
                     odom_.pose.pose.orientation.w);

    
   tf::Matrix3x3 rot_matrix_(q_);

    for(int i=0;i<3;i++)
    {
        tf::Vector3 vec=rot_matrix_.getRow(i);
        pose_.data[i].x=vec.getX();
        pose_.data[i].y=vec.getY();
        pose_.data[i].z=vec.getZ();
        pose_.data[i].w=0;
    }
    pose_.data[3].x=0;
    pose_.data[3].y=0;
    pose_.data[3].z=0;
    pose_.data[3].w=1;
    pose_.data[0].w=odom_.pose.pose.position.x;
    pose_.data[1].w=odom_.pose.pose.position.y;
    pose_.data[2].w=odom_.pose.pose.position.z;
 
    pose = T_B_P * pose;
    pose_ = T_B_P * pose_;
    //delta_pose =  inverse(pose_) * pose;
    delta_pose =  pose *  inverse(pose_);


    


    Matrix4 p=kfusion->getPose();

   Matrix4 p_new = p * delta_pose;

   for(int i=0;i<4;i++)
   {
         std::cout<<p.data[i].x<<" "<<p.data[i].y<<" "<<p.data[i].z<<" "<<p.data[i].w<<std::endl;
        std::cout<<p_new.data[i].x<<" "<<p_new.data[i].y<<" "<<p_new.data[i].z<<" "<<p_new.data[i].w<<std::endl;
       
     }



    kfusion->setPose(p_new);
    //Integrate
    if(!kfusion->integration(params.camera,
                             params.integration_rate,
                             params.mu, frame))
        ROS_ERROR("integration faild");

    kfusion->raycasting(params.camera, params.mu, frame);
        
    if(publish_volume)
    {
        kfusion->renderVolume(volumeRender,
                            params.computationSize, 
                            frame, 
                            params.rendering_rate, 
                            params.camera, 
                            0.75 * params.mu);
        publishVolume();
    }
    
    
    if(publish_points && frame % publish_points_rate ==0)
         publishPoints();
     
    frame++;    
    odom_ = odom;
    
}

void depthCallback(const sensor_msgs::ImageConstPtr &depth)
{
    static bool first_time=true;
    params.inputSize.y=depth->height;
    params.inputSize.x=depth->width;

    if(kfusion==0)
    {
        params.inputSize.y=depth->height;
        params.inputSize.x=depth->width;
        initKFusion();
    }

    const float *in_data=(const float*)depth->data.data();
    if(strcmp(depth->encoding.c_str(), "32FC1")==0) //32FC1
    {
        if(inputDepthFl==0)
            inputDepthFl=new float[params.inputSize.x * params.inputSize.y];
            
        memcpy(inputDepthFl,depth->data.data(),params.inputSize.y*params.inputSize.x*sizeof(float) );
        kfusion->preprocessing2(inputDepthFl, params.inputSize);
    }
    else if(strcmp(depth->encoding.c_str(), "16UC1")==0) //16UC1
    {
        if(inputDepth==0)
            inputDepth = new uint16_t[params.inputSize.x * params.inputSize.y];   
        
        memcpy(inputDepth,depth->data.data(),params.inputSize.y*params.inputSize.x*2);
        kfusion->preprocessing(inputDepth, params.inputSize);
    }
    else
    {
        ROS_ERROR("Not supported depth format.");
        return;
    }
        
    if(!kfusion->tracking(params.camera,
                      params.icp_threshold,
                      params.tracking_rate, frame) )
    {
       ROS_ERROR("Tracking faild");
    }
    
    
    publishOdom();
    odom_rec=0;
}

void camInfoCallback(sensor_msgs::CameraInfoConstPtr msg)
{
    ROS_INFO("cam info");
    params.camera =  make_float4(msg->K[0],msg->K[4],msg->K[2],msg->K[5]);
}

void initKFusion()
{
    params.computationSize = make_uint2(
                params.inputSize.x / params.compute_size_ratio,
                params.inputSize.y / params.compute_size_ratio);

    
    ROS_INFO("camera is = %f, %f, %f, %f",
             params.camera.x,
             params.camera.y,
             params.camera.z,
             params.camera.w);
    
    Matrix4 poseMatrix;    
    
    for(int i=0;i<4;i++)
    {
        poseMatrix.data[i].x = 0;
        poseMatrix.data[i].y = 0;
        poseMatrix.data[i].z = 0;        
    }
    
     poseMatrix.data[0].x = 1;
     poseMatrix.data[1].y = 1;
     poseMatrix.data[2].z = 1;
    
    poseMatrix.data[0].w =  0;
    poseMatrix.data[1].w =  0;
    poseMatrix.data[2].w =  0;
    poseMatrix.data[3].w =  1;    

    poseMatrix.data[0].w +=  params.volume_direction.x;
    poseMatrix.data[1].w +=  params.volume_direction.y;
    poseMatrix.data[2].w +=  params.volume_direction.z;

    inputRGB     = new uchar3[params.inputSize.x * params.inputSize.y];
    depthRender  = new uchar4[params.computationSize.x * params.computationSize.y];
    volumeRender = new uchar4[params.computationSize.x * params.computationSize.y];
    
    kfusion = new Kfusion(params.computationSize, 
                          params.volume_resolution, 
                          params.volume_size, 
                          poseMatrix, 
                          params.pyramid);
}

void publishVolume()
{
    sensor_msgs::Image image;
    image.header.stamp=ros::Time::now();
    
    image.height=params.inputSize.y;
    image.width=params.inputSize.x;
    
    int step_size=sizeof(uchar)*4;
    image.is_bigendian=0;
    image.step=step_size*image.width;
    image.header.frame_id=std::string("kfusion_volume");   
    image.encoding=std::string("bgra8");
    
    image.header.frame_id=std::string("kfusion volume");
    uchar *ptr=(uchar*)volumeRender;
    image.data=std::vector<uchar>(ptr ,ptr+(params.computationSize.x * params.computationSize.y*step_size) );
    volume_pub.publish(image);
}

void publishOdom()
{
    Matrix4 pose = kfusion->getPose();

    tf::Vector3 vec[3];
    for(int i=0;i<3;i++)
    {
        vec[i]=tf::Vector3(pose.data[i].x,pose.data[i].y,pose.data[i].z);
    }

    tf::Matrix3x3 rot_matrix(vec[0].getX(),vec[0].getY(),vec[0].getZ(),
                             vec[1].getX(),vec[1].getY(),vec[1].getZ(),
                             vec[2].getX(),vec[2].getY(),vec[2].getZ() );

    tf::Quaternion q;

    rot_matrix.getRotation(q);

    nav_msgs::Odometry odom;
    odom.header.stamp = ros::Time::now();
    odom.header.frame_id = "kfusion_odom";

    //set the position
    odom.pose.pose.position.x = pose.data[0].w-params.volume_direction.x;
    odom.pose.pose.position.y = pose.data[1].w-params.volume_direction.y;
    odom.pose.pose.position.z = pose.data[2].w-params.volume_direction.z;

    //set quaternion
    odom.pose.pose.orientation.x=q.getX();
    odom.pose.pose.orientation.y=q.getY();
    odom.pose.pose.orientation.z=q.getZ();
    odom.pose.pose.orientation.w=q.getW();

    //set velocity to zero
    odom.child_frame_id = "base_link";
    odom.twist.twist.linear.x = 0;
    odom.twist.twist.linear.y = 0;
    odom.twist.twist.angular.z = 0;

    odom_pub.publish(odom);
}

void publishPoints()
{
    std::vector<float3> vertices;
    kfusion->getVertices(vertices);
    
    sensor_msgs::PointCloud pcloud;
    pcloud.header.stamp = ros::Time::now();
    pcloud.header.frame_id = "odom";
    pcloud.points.reserve(vertices.size());
    sensor_msgs::ChannelFloat32 ch;
    
//     pcloud.channels.reserve(vertices.size());
    
    for(int i=0;i<vertices.size();i++)
    {
        float3 vertex=vertices[i];
        geometry_msgs::Point32 p;


        p.x= vertex.z -params.volume_direction.z;
        p.y= -vertex.x +params.volume_direction.x;
        p.z= -vertex.y +params.volume_direction.y;
        pcloud.points.push_back(p);
        ch.values.push_back(1);    
    }
    
    pcloud.channels.push_back(ch);
    points_pub.publish(pcloud);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "kfusion_node",ros::init_options::AnonymousName);
    ros::NodeHandle n_p("~");
    std::string cam_info_topic,depth_topic,rgb_topic,odom_in_topic;
    if(!n_p.getParam("cam_info_topic", cam_info_topic))
    {
        cam_info_topic=std::string(CAM_INFO_TOPIC);
    }
    if(!n_p.getParam("depth_topic", depth_topic))
    {
        depth_topic=std::string(DEPTH_TOPIC);
    }
    if(!n_p.getParam("rgb_topic", rgb_topic))
    {
        rgb_topic=std::string(RGB_TOPIC);
    }
    if(!n_p.getParam("odom_input_topic", odom_in_topic))
    {
        odom_in_topic=std::string(PUB_ODOM_TOPIC);
    }    
    if(!n_p.getParam("odom_delay", odom_delay))
    {
        odom_delay=3;
    } 
    if(!n_p.getParam("publish_volume", publish_volume))
    {
        publish_volume=true;
    }
    if(!n_p.getParam("publish_points", publish_points))
    {
        publish_points=true;
    } 
    if(!n_p.getParam("publish_points_rate", publish_points_rate))
    {
        publish_points_rate=PUBLISH_POINT_RATE;
    } 
    readKfusionParams(n_p);
    
    std::vector<double> pose_list;
    n_p.getParam("T_B_P",pose_list);
 std::cout<<" TB B"<<std::endl;

    for(int i = 0; i<4; i++)
    {
        T_B_P.data[i].x = pose_list[4*i];
        T_B_P.data[i].y = pose_list[4*i+1];
        T_B_P.data[i].z = pose_list[4*i+2];
        T_B_P.data[i].w = pose_list[4*i+3];
        std::cout<<T_B_P.data[i].x<<" "<<T_B_P.data[i].y<<" "<<T_B_P.data[i].z<<" "<<T_B_P.data[i].w<<std::endl;
    }



    volume_pub = n_p.advertise<sensor_msgs::Image>(PUB_VOLUME_TOPIC, 1000);    
    odom_pub = n_p.advertise<nav_msgs::Odometry>(PUB_ODOM_TOPIC, 50);
    points_pub = n_p.advertise<sensor_msgs::PointCloud>(PUB_POINTS_TOPIC, 100);

    while(ros::ok())
    {
        sensor_msgs::CameraInfoConstPtr cam_info=ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic);
        if(cam_info)
        {
            camInfoCallback(cam_info);
            break;
        }
    }

    image_transport::ImageTransport it(n_p);
    image_transport::Subscriber sub = it.subscribe(depth_topic, 1, depthCallback);
    ros::Subscriber odom_sub = n_p.subscribe(odom_in_topic, 1, odomCallback);

//    cam_info_sub = n_p.subscribe(cam_info_topic, 1, camInfoCallback);

//    params.camera =  make_float4(msg.K[0],msg.K[4],msg.K[2],msg.K[5]);

//     message_filters::Subscriber<sensor_msgs::Image> rgb_sub(n_p, rgb_topic, 2);
    
    /*
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(n_p, rgb_topic, 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(n_p, depth_topic, 1);
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync(rgb_sub, depth_sub, 2);
    sync.registerCallback(boost::bind(&frameCallback, _1, _2));
    */

    ros::spin();
}
