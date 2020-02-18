#include <ros/ros.h>

#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>

#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <string.h>
#include <kernels.h>

#include <tf/LinearMath/Matrix3x3.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer_interface.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/CameraInfo.h>

#define CAM_INFO_TOPIC "/camera/depth/camera_info"
#define RGB_TOPIC "/camera/rgb/image_rect_color"
#define DEPTH_TOPIC "/camera/depth/image_rect"

#define PUB_VOLUME_TOPIC "/kfusion/volume_rendered"
#define PUB_ODOM_TOPIC "/kfusion/odom"
#define PUB_POINTS_TOPIC "/kfusion/pointCloud"

#define DEPTH_FRAME "camera_rgb_optical_frame"
#define VO_FRAME "visual_odom"
#define ODOM_FRAME "odom"
#define BASE_LINK "base_link"

#define PUBLISH_POINT_RATE 10

#define PUB_ODOM_PATH_TOPIC "/kfusion/odom_path"
nav_msgs::Path odomPath;
ros::Publisher odom_path_pub ;

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
kparams_t params;


;
/*
0,-1,  0 ,0,
0, 0, -1, 0,
1, 0,  0, 0,
0, 0,  0, 1 
*/

inline sMatrix4 fromVisionCord(const sMatrix4 &mat)
{
    static bool firstTime=true;
    static sMatrix4 T_B_P,invT_B_P;
    if(firstTime)
    {
        T_B_P.data[0].x=0;
        T_B_P.data[0].y=-1;
        T_B_P.data[0].z=0;
        T_B_P.data[0].w=0;
        
        T_B_P.data[1].x=0;
        T_B_P.data[1].y=0;
        T_B_P.data[1].z=-1;
        T_B_P.data[1].w=0;
        
        T_B_P.data[2].x=1;
        T_B_P.data[2].y=0;
        T_B_P.data[2].z=0;
        T_B_P.data[2].w=0;
        
        T_B_P.data[3].x=0;
        T_B_P.data[3].y=0;
        T_B_P.data[3].z=0;
        T_B_P.data[3].w=1;
        invT_B_P=inverse(T_B_P);
        firstTime=false;
    }
    
    return invT_B_P*mat*T_B_P;
}

//Buffers
uint16_t *inputDepth=0;
float *inputDepthFl=0;
uchar3 *inputRGB;
uchar4 *depthRender;
uchar4 *trackRender;
uchar4 *volumeRender;

//KFusion
Kfusion *kfusion=nullptr;


std::string odom_in_topic;

int frame = 0;
int odom_delay;
int odom_rec=0;
int odom_counter = 0;

//other params
bool publish_volume=true;
bool publish_points=true;
int publish_points_rate;

//ros pub/subs
ros::Subscriber cam_info_sub;
ros::Publisher volume_pub;
ros::Publisher odom_pub ;
ros::Publisher points_pub;
nav_msgs::Odometry odom_;

//hold previous pose
Matrix4 pose_old;
bool hasPoseOld=false;

//frames
std::string depth_frame,vo_frame,base_link_frame,odom_frame;

//Transformations
geometry_msgs::TransformStamped odom_to_vo,vo_to_odom;
geometry_msgs::TransformStamped cam_to_base,base_to_cam;
tf2_ros::Buffer tfBuffer;

//functions
void initKFusion();
void publishVolume();
void publishOdom();
void publishPoints();
geometry_msgs::Pose transform2pose(const geometry_msgs::Transform &trans);

void publishOdomPath(geometry_msgs::Pose &p)
{
//     nav_msgs::Path path;
    
    geometry_msgs::PoseStamped ps;
    ps.header.stamp = ros::Time::now();
    ps.header.frame_id = VO_FRAME;
    ps.pose=p;
    odomPath.poses.push_back(ps);
    
    nav_msgs::Path newPath=odomPath;
    newPath.header.stamp = ros::Time::now();
    newPath.header.frame_id = VO_FRAME;
    
    odom_path_pub.publish(newPath);
}

geometry_msgs::Pose transform2pose(const geometry_msgs::Transform &trans)
{
    geometry_msgs::Pose pose;
    pose.position.x=trans.translation.x;
    pose.position.y=trans.translation.y;
    pose.position.z=trans.translation.z;

    pose.orientation.x=trans.rotation.x;
    pose.orientation.y=trans.rotation.y;
    pose.orientation.z=trans.rotation.z;
    pose.orientation.w=trans.rotation.w;

    return pose;
}

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

    params=par;
}

void odomCallback(nav_msgs::OdometryConstPtr odom)
{
    ROS_DEBUG("odom %d",odom_rec);
    if(odom_rec!=odom_delay)
    {
        odom_rec++; 
        return;
    }
    odom_rec++; 


    geometry_msgs::Pose odomP;
    geometry_msgs::Pose poseTmp;

    /*Create tf transformation from odom message*/
    geometry_msgs::TransformStamped base_to_odom;
    base_to_odom.header.frame_id=odom_frame;
    base_to_odom.child_frame_id=base_link_frame;
    
    base_to_odom.transform.translation.x=odom->pose.pose.position.x;
    base_to_odom.transform.translation.y=odom->pose.pose.position.y;
    base_to_odom.transform.translation.z=odom->pose.pose.position.z;
    
    base_to_odom.transform.rotation.x=odom->pose.pose.orientation.x;
    base_to_odom.transform.rotation.y=odom->pose.pose.orientation.y;
    base_to_odom.transform.rotation.z=odom->pose.pose.orientation.z;
    base_to_odom.transform.rotation.w=odom->pose.pose.orientation.w;
    

    odomP=transform2pose(cam_to_base.transform);
    try
    {
        tf2::doTransform(odomP,odomP, base_to_odom);
        tf2::doTransform(odomP,odomP, odom_to_vo);
    }
    catch (tf2::TransformException &ex) 
    {
        
        ROS_WARN("Odom transformation failure %s\n", ex.what()); //Print exception which was caught    
        return;
    }

    Matrix4 pose, delta_pose;

    //New Odometry
    tf::Quaternion q(odomP.orientation.x,
                     odomP.orientation.y,
                     odomP.orientation.z,
                     odomP.orientation.w);

    
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

    
    pose.data[0].w=odomP.position.x;
    pose.data[1].w=odomP.position.y; 
    pose.data[2].w=odomP.position.z;
        
    pose.data[0].w += params.volume_direction.x;
    pose.data[1].w += params.volume_direction.y;
    pose.data[2].w += params.volume_direction.z;
    
    
    if(!hasPoseOld)
    {
        pose_old=pose;
        hasPoseOld=true;
        return;
    }
    if(kfusion==nullptr)
        return;    
    
    delta_pose = inverse(pose_old) * pose;
    pose_old=pose;
    hasPoseOld=true;

    Matrix4 p=kfusion->getPose();
    Matrix4 p_new = kfusion->getPose() * delta_pose;

    kfusion->setPose(p_new);

    //Integrate
    if(!kfusion->integration(params.camera,
                             params.integration_rate,
                             params.mu, frame))
        ROS_ERROR("integration faild");

    //Raycasting
    kfusion->raycasting(params.camera, params.mu, frame);

    if(publish_volume && false)
    {
        kfusion->renderVolume(volumeRender,
                            params.computationSize, 
                            frame, 
                            params.rendering_rate, 
                            params.camera, 
                            0.75 * params.mu);
        publishVolume();
    }
    
    
    if(publish_points && frame % publish_points_rate ==0 && false)
         publishPoints();

    frame++;
}

void depthCallback(const sensor_msgs::ImageConstPtr &depth)
{
    if(kfusion==0 )
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
        
    bool track_success=kfusion->tracking(params.camera,
                      params.icp_threshold,
                      params.tracking_rate, frame);
    odom_rec=0;
    if(! track_success)
       ROS_ERROR("Tracking faild");
    else
        publishOdom();
    
    //Integrate
    if(!kfusion->integration(params.camera,
                             params.integration_rate,
                             params.mu, frame))
        ROS_ERROR("integration faild");

    //Raycasting
    kfusion->raycasting(params.camera, params.mu, frame);

    if(publish_volume && false)
    {
        kfusion->renderVolume(volumeRender,
                            params.computationSize, 
                            frame, 
                            params.rendering_rate, 
                            params.camera, 
                            0.75 * params.mu);
        publishVolume();
    }
    
    
    if(publish_points && frame % publish_points_rate ==0 && false)
         publishPoints();

    frame++;
}

void camInfoCallback(sensor_msgs::CameraInfoConstPtr msg)
{
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
    Matrix4 kpose =  kfusion->getPose();
    
    sMatrix4 pose=fromVisionCord(kpose);
    
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
    geometry_msgs::Pose odom_pose;
    odom_pose.position.x=pose.data[0].w;
    odom_pose.position.y=pose.data[1].w;
    odom_pose.position.z=pose.data[2].w;
    odom_pose.orientation.x=q.getX();
    odom_pose.orientation.y=q.getY();
    odom_pose.orientation.z=q.getZ();
    odom_pose.orientation.w=q.getW();
    
    odom.header.stamp = ros::Time::now();

    //set velocity to zero    
    odom.twist.twist.linear.x = 0;
    odom.twist.twist.linear.y = 0;
    odom.twist.twist.angular.z = 0;
    
    odom.header.frame_id = vo_frame;
    odom.child_frame_id = depth_frame;

    odom.pose.pose=odom_pose;
    odom_pub.publish(odom);
    
    publishOdomPath(odom_pose);    

}

void publishPoints()
{
    std::vector<float3> vertices;
    kfusion->getVertices(vertices);
    
    sensor_msgs::PointCloud pcloud;
    pcloud.header.stamp = ros::Time::now();
    pcloud.header.frame_id = odom_frame;
    pcloud.points.reserve(vertices.size());
    sensor_msgs::ChannelFloat32 ch;    
    
    for(int i=0;i<vertices.size();i++)
    {
        float3 vertex=vertices[i];
        geometry_msgs::Point point;

        point.x= vertex.x -params.volume_direction.x;
        point.y= vertex.y -params.volume_direction.y;
        point.z= vertex.z -params.volume_direction.z;

        try
        {
            tf2::doTransform(point,point, vo_to_odom);
        }
        catch (tf2::TransformException &ex)
        {

            ROS_WARN("Odom transformation failure %s\n", ex.what()); //Print exception which was caught
            return;
        }

        geometry_msgs::Point32 p;
        p.x=point.x;
        p.y=point.y;
        p.z=point.z;

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

    std::string cam_info_topic,depth_topic,rgb_topic;
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
//     if(!n_p.getParam("odom_input_topic", odom_in_topic))
//     {
//         odom_in_topic=std::string(PUB_ODOM_TOPIC);
//     }    
//     if(!n_p.getParam("odom_delay", odom_delay))
//     {
//         odom_delay=3;
//     } 
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
    if(!n_p.getParam("vo_frame", vo_frame))
    {
        vo_frame=VO_FRAME;
    }    
    if(!n_p.getParam("depth_frame", depth_frame))
    {
        depth_frame=DEPTH_FRAME;
    }
    if(!n_p.getParam("base_link_frame", base_link_frame))
    {
        base_link_frame=BASE_LINK;
    }
    if(!n_p.getParam("odom_frame", odom_frame))
    {
        odom_frame=ODOM_FRAME;
    }


    ROS_INFO("Depth Frame:%s",depth_frame.c_str());
           
    readKfusionParams(n_p);
    
    if(publish_volume)
        volume_pub = n_p.advertise<sensor_msgs::Image>(PUB_VOLUME_TOPIC, 1000);

    if(publish_points)
        points_pub = n_p.advertise<sensor_msgs::PointCloud>(PUB_POINTS_TOPIC, 100);

    odom_pub = n_p.advertise<nav_msgs::Odometry>(PUB_ODOM_TOPIC, 50);

    ROS_INFO("Waiting camera info");
    while(ros::ok())
    {
        sensor_msgs::CameraInfoConstPtr cam_info=ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic);
        if(cam_info)
        {
            camInfoCallback(cam_info);
            break;
        }
    }
    ROS_INFO("Camera info received.");

#if 0
    ROS_INFO("Waiting tf transformation");
    do
    {
        tf2_ros::TransformListener tfListener(tfBuffer);        
        while(true)
        {
            try
            {
                odom_to_vo = tfBuffer.lookupTransform(depth_frame,odom_frame, ros::Time(0));
                cam_to_base = tfBuffer.lookupTransform(base_link_frame,depth_frame, ros::Time(0));
                break;
            }
            catch (tf2::TransformException &ex) 
            {
                ROS_WARN("Failure %s\n", ex.what()); //Print exception which was caught    
                ros::Duration(0.2).sleep();
            }
        }        
        
        tf2::Transform tr;
        tf2::fromMsg(odom_to_vo.transform,tr);
        tr=tr.inverse();
        vo_to_odom.transform=tf2::toMsg(tr);

        tf2::fromMsg(cam_to_base.transform,tr);
        tr=tr.inverse();
        base_to_cam.transform=tf2::toMsg(tr);
    }while(false);
#endif

    odom_path_pub = n_p.advertise<nav_msgs::Path>(PUB_ODOM_PATH_TOPIC, 50);

//     ros::Subscriber odom_sub = n_p.subscribe(odom_in_topic, odom_delay+1, odomCallback);
    ROS_INFO("Waiting depth message");

    image_transport::ImageTransport it(n_p);
    image_transport::Subscriber sub = it.subscribe(depth_topic, 1, depthCallback);
    
    ros::spin();
}
