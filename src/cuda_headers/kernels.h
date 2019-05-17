#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

#undef isnan
#undef isfinite

#include <TooN/TooN.h>
#include <TooN/se3.h>
#include <TooN/GR_SVD.h>


#include <kfusion.h>



// #include <values/Value.h>

#define INVALID -2   // this is used to mark invalid entries in normal or vertex maps



template<typename P>
inline Matrix4 toMatrix4(const TooN::SE3<P> & p) {
    const TooN::Matrix<4, 4, float> I = TooN::Identity;
    Matrix4 R;
    TooN::wrapMatrix<4, 4>(&R.data[0].x) = p * I;
    return R;
}

template<typename P>
inline sMatrix4 tosMatrix4(const TooN::SE3<P> & p) {
    const TooN::Matrix<4, 4, float> I = TooN::Identity;
    sMatrix4 R;
    TooN::wrapMatrix<4, 4>(&R.data[0].x) = p * I;
    return R;
}

template<typename P, typename A>
TooN::Matrix<6> makeJTJ(const TooN::Vector<21, P, A> & v) {
    TooN::Matrix<6> C = TooN::Zeros;
    C[0] = v.template slice<0, 6>();
    C[1].template slice<1, 5>() = v.template slice<6, 5>();
    C[2].template slice<2, 4>() = v.template slice<11, 4>();
    C[3].template slice<3, 3>() = v.template slice<15, 3>();
    C[4].template slice<4, 2>() = v.template slice<18, 2>();
    C[5][5] = v[20];

    for (int r = 1; r < 6; ++r)
        for (int c = 0; c < r; ++c)
            C[r][c] = C[c][r];

    return C;
}

template<typename T, typename A>
TooN::Vector<6> solve(const TooN::Vector<27, T, A> & vals) {
    const TooN::Vector<6> b = vals.template slice<0, 6>();
    const TooN::Matrix<6> C = makeJTJ(vals.template slice<6, 21>());

    TooN::GR_SVD<6, 6> svd(C);
    return svd.backsub(b, 1e6);
}

/// OBJ ///

class Kfusion {
    private:
        uint2 computationSize;
        float step;
        sMatrix4 pose;
        sMatrix4 oldPose;
        sMatrix4 deltaPose;
        sMatrix4 *viewPose;
        float3 volumeDimensions;
        uint3 volumeResolution;
        std::vector<int> iterations;
        bool _tracked = false;
        bool _integrated = false;
    public:

        Kfusion(uint2 inputSize, uint3 volumeResolution, float3 volumeDimensions,
                float3 initPose, std::vector<int> & pyramid) :
            computationSize(make_uint2(inputSize.x, inputSize.y)) {


            this->volumeDimensions = volumeDimensions;
            this->volumeResolution = volumeResolution;
            pose = toMatrix4(
                       TooN::SE3<float>(
                           TooN::makeVector(initPose.x, initPose.y, initPose.z, 0, 0, 0)));
            this->iterations.clear();
            for (std::vector<int>::iterator it = pyramid.begin();
                 it != pyramid.end(); it++) {
                this->iterations.push_back(*it);
            }

            step = min(volumeDimensions) / max(volumeResolution);
            viewPose = &pose;
            this->languageSpecificConstructor();
        }


        //Allow a kfusion object to be created with a pose which include orientation as well as position
        Kfusion(uint2 inputSize, uint3 volumeResolution, float3 volumeDimensions,
                Matrix4 initPose, std::vector<int> & pyramid) :
            computationSize(make_uint2(inputSize.x, inputSize.y)) {

            this->volumeDimensions = volumeDimensions;
            this->volumeResolution = volumeResolution;
            pose = initPose;

            this->iterations.clear();
            for (std::vector<int>::iterator it = pyramid.begin();
                 it != pyramid.end(); it++) {
                this->iterations.push_back(*it);
            }

            step = min(volumeDimensions) / max(volumeResolution);
            viewPose = &pose;
            this->languageSpecificConstructor();
        }


        void languageSpecificConstructor();
        ~Kfusion();

        void reset();

        void computeFrame(const ushort * inputDepth,
                          const uint2 inputSize, float4 k,
                          uint integration_rate, uint tracking_rate, float icp_threshold,
                          float mu, const uint frame) ;

        bool preprocessing(const ushort * inputDepth,
                           const uint2 inputSize);
        bool preprocessing2(const float *inputDepth, const uint2 inputSize) ;
        
        bool tracking(float4 k, float icp_threshold,
                      uint tracking_rate, uint frame);
        bool raycasting(float4 k, float mu, uint frame);
        bool integration(float4 k, uint integration_rate,
                         float mu, uint frame);

        void dumpVolume(const  char * filename);
        void renderVolume(uchar4 * out,
                          const uint2 outputSize, int, int,
                          float4 k, float largestep);
        void renderTrack(uchar4 * out,
                         const uint2 outputSize);
        void renderDepth(uchar4 * out,
                         uint2 outputSize);

        void getVertices(std::vector<float3> &vertices);

        sMatrix4 getPose() {
            return pose;
        }

        sMatrix4 getDeltaPose() const
        {
            return deltaPose;
        }

        void setPose(const sMatrix4 pose_) {
            pose=pose_;
        }
        void setViewPose(sMatrix4 *value = NULL) {
            if (value == NULL)
                viewPose = &pose;
            else
                viewPose = value;
        }
        sMatrix4 *getViewPose() {
            return (viewPose);
        }
        float3 getModelDimensions() {
            return (volumeDimensions);
        }
        uint3 getModelResolution() {
            return (volumeResolution);
        }
        uint2 getComputationResolution() {
            return (computationSize);
        }

};


