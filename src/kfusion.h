#ifndef KFUSION_H
#define KFUSION_H


#if defined(__GNUC__)
// circumvent packaging problems in gcc 4.7.0
#undef _GLIBCXX_ATOMIC_BUILTINS 
#undef _GLIBCXX_USE_INT128 

// need c headers for __int128 and uint16_t

#include <limits.h>
#endif

#include <stdint.h>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

#include "cutil_math.h"
#include "volume.h"

inline int divup(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
inline dim3 divup(uint2 a, dim3 b) {
    return dim3(divup(a.x, b.x), divup(a.y, b.y));
}
inline dim3 divup(dim3 a, dim3 b) {
    return dim3(divup(a.x, b.x), divup(a.y, b.y), divup(a.z, b.z));
}

struct KFusionConfig {

        bool combinedTrackAndReduce; // combine tracking and calculating linear system in one
        // this saves some time in tracking, but there is no per pixel output anymore

        float4 camera;              // camera configuration parameters
        uint2 inputSize;            // size of the input depth images
        float nearPlane, farPlane;  // values for raycasting in meters
        float mu;             // width of linear ramp, left and right of 0 in meters
        float maxweight; // maximal weight for volume integration, controls speed of updates

        int radius;                 // bilateral filter radius
        float delta;                // gaussian delta
        float e_delta;              // euclidean delta

        float dist_threshold;       // 3D distance threshold for ICP correspondences
        float normal_threshold; // dot product normal threshold for ICP correspondences
        std::vector<int> iterations;  // max number of iterations per level
        float track_threshold; // percent of tracked pixels to accept tracking result

        dim3 imageBlock;            // block size for image operations
        dim3 raycastBlock;          // block size for raycasting

        KFusionConfig() {

            combinedTrackAndReduce = false;

            camera = make_float4(160, 160, 160, 120);
            inputSize = make_uint2(320, 240);

            nearPlane = 0.4f;
            farPlane = 4.0f;
            mu = 0.1f;
            maxweight = 100.0f;

            radius = 2;
            delta = 4.0f;
            e_delta = 0.1f;

            dist_threshold = 0.1f;
            normal_threshold = 0.8f;
            iterations.push_back(5);
            iterations.push_back(5);
            iterations.push_back(5);
            track_threshold = 0.15f;

            imageBlock = dim3(32, 16);
            raycastBlock = dim3(32, 8);
        }

};

struct Matrix4 {
        float4 data[4];
        Matrix4() {
        }
        Matrix4(Matrix4 * src) {
            this->data[0] = make_float4(src->data[0].x, src->data[0].y,
                    src->data[0].z, src->data[0].w);
            this->data[1] = make_float4(src->data[1].x, src->data[1].y,
                    src->data[1].z, src->data[1].w);
            this->data[2] = make_float4(src->data[2].x, src->data[2].y,
                    src->data[2].z, src->data[2].w);
            this->data[3] = make_float4(src->data[3].x, src->data[3].y,
                    src->data[3].z, src->data[3].w);
        }

        inline __host__  __device__ float3 get_translation() const {
            return make_float3(data[0].w, data[1].w, data[2].w);
        }
};


typedef Matrix4 sMatrix4;


std::ostream & operator<<(std::ostream & out, const Matrix4 & m);
Matrix4 operator*(const Matrix4 & A, const Matrix4 & B);
Matrix4 inverse(const Matrix4 & A);

inline __host__  __device__ float4 operator*(const Matrix4 & M,
                                             const float4 & v) {
    return make_float4(dot(M.data[0], v), dot(M.data[1], v), dot(M.data[2], v),
            dot(M.data[3], v));
}

inline __host__  __device__ float3 operator*(const Matrix4 & M,
                                             const float3 & v) {
    return make_float3(dot(make_float3(M.data[0]), v) + M.data[0].w,
            dot(make_float3(M.data[1]), v) + M.data[1].w,
            dot(make_float3(M.data[2]), v) + M.data[2].w);
}

inline __host__  __device__ float3 rotate(const Matrix4 & M, const float3 & v) {
    return make_float3(dot(make_float3(M.data[0]), v),
            dot(make_float3(M.data[1]), v), dot(make_float3(M.data[2]), v));
}

inline Matrix4 getCameraMatrix(const float4 & k) {
    Matrix4 K;
    K.data[0] = make_float4(k.x, 0, k.z, 0);
    K.data[1] = make_float4(0, k.y, k.w, 0);
    K.data[2] = make_float4(0, 0, 1, 0);
    K.data[3] = make_float4(0, 0, 0, 1);
    return K;
}

inline Matrix4 getInverseCameraMatrix(const float4 & k) {
    Matrix4 invK;
    invK.data[0] = make_float4(1.0f / k.x, 0, -k.z / k.x, 0);
    invK.data[1] = make_float4(0, 1.0f / k.y, -k.w / k.y, 0);
    invK.data[2] = make_float4(0, 0, 1, 0);
    invK.data[3] = make_float4(0, 0, 0, 1);
    return invK;
}

inline __device__ uint2 thr2pos2() {
#ifdef __CUDACC__
    return make_uint2( __umul24(blockDim.x, blockIdx.x) + threadIdx.x,
                       __umul24(blockDim.y, blockIdx.y) + threadIdx.y);
#else
    return make_uint2(0);
#endif
}

struct Ref {
        Ref(void * d = NULL) :
            data(d) {
        }
        void * data;
};

struct Host {
        Host() :
            data(NULL) {
        }
        ~Host() {
            cudaFreeHost(data);
        }

        void alloc(uint size) {
            if (data != NULL) {
                cudaFreeHost(data);
            }
            cudaHostAlloc(&data, size, cudaHostAllocDefault);
        }
        void * data;
};

struct Device {
        void * data;
        Device() :
            data(NULL) {
        }
        ~Device() {
            cudaFree(data);
        }

        void alloc(uint size) {
            if (data != NULL) {
                cudaFree(data);
            }
            cudaError_t __cudaCalloc_err = cudaMalloc(&data, size);
            if (__cudaCalloc_err == cudaSuccess)
                cudaMemset(data, 0, size);

        }

};

struct HostDevice {
        HostDevice() :
            data(NULL) {
        }
        ~HostDevice() {
            cudaFreeHost(data);
        }

        void alloc(uint size) {
            if (data != NULL) {
                cudaFreeHost(data);
            }
            cudaHostAlloc(&data, size, cudaHostAllocMapped);
        }
        void * getDevice() const {
            void * devicePtr;
            cudaHostGetDevicePointer(&devicePtr, data, 0);
            return devicePtr;
        }
        void * data;
};

template<typename OTHER>
inline void image_copy(Ref & to, const OTHER & from, uint size) {
    to.data = from.data;
}

inline void image_copy(Host & to, const Host & from, uint size) {
    cudaMemcpy(to.data, from.data, size, cudaMemcpyHostToHost);
}

inline void image_copy(Host & to, const Device & from, uint size) {
    cudaMemcpy(to.data, from.data, size, cudaMemcpyDeviceToHost);
}

inline void image_copy(Host & to, const HostDevice & from, uint size) {
    cudaMemcpy(to.data, from.data, size, cudaMemcpyHostToHost);
}

inline void image_copy(Device & to, const Ref & from, uint size) {
    cudaMemcpy(to.data, from.data, size, cudaMemcpyDeviceToDevice);
}

inline void image_copy(Device & to, const Host & from, uint size) {
    cudaMemcpy(to.data, from.data, size, cudaMemcpyHostToDevice);
}

inline void image_copy(Device & to, const Device & from, uint size) {
    cudaMemcpy(to.data, from.data, size, cudaMemcpyDeviceToDevice);
}

inline void image_copy(Device & to, const HostDevice & from, uint size) {
    cudaMemcpy(to.data, from.getDevice(), size, cudaMemcpyDeviceToDevice);
}

inline void image_copy(HostDevice & to, const Host & from, uint size) {
    cudaMemcpy(to.data, from.data, size, cudaMemcpyHostToHost);
}

inline void image_copy(HostDevice & to, const Device & from, uint size) {
    cudaMemcpy(to.getDevice(), from.data, size, cudaMemcpyDeviceToDevice);
}

inline void image_copy(HostDevice & to, const HostDevice & from, uint size) {
    cudaMemcpy(to.data, from.data, size, cudaMemcpyHostToHost);
}

template<typename T, typename Allocator = Ref>
struct Image: public Allocator
{
    typedef T PIXEL_TYPE;
    uint2 size;

    Image() :Allocator()
    {
        size = make_uint2(0, 0);
    }
    Image(const uint2 & s)
    {
        size = make_uint2(0, 0);
        alloc(s);
    }

    void alloc(const uint2 & s)
    {
        if (s.x == size.x && s.y == size.y)
            return;

        Allocator::alloc(s.x * s.y * sizeof(T));
        size = s;
    }

    __device__
    T & el()
    {
        return operator[](thr2pos2());
    }

    __device__
    const T & el() const
    {
        return operator[](thr2pos2());
    }

    __device__
    T & operator[](const uint2 & pos)
    {
        return static_cast<T *>(Allocator::data)[pos.x + size.x * pos.y];
    }

    __device__
    const T & operator[](const uint2 & pos) const {
        return static_cast<const T *>(Allocator::data)[pos.x + size.x * pos.y];
    }

    Image<T> getDeviceImage()
    {
        return Image<T>(size, Allocator::getDevice());
    }

    operator Image<T>()
    {
        return Image<T>(size, Allocator::data);
    }

    template<typename A1>
    Image<T, Allocator> & operator=(const Image<T, A1> & other)
    {
        image_copy(*this, other, size.x * size.y * sizeof(T));
        return *this;
    }

    T * data()
    {
        return static_cast<T *>(Allocator::data);
    }

    const T * data() const
    {
        return static_cast<const T *>(Allocator::data);
    }
};

template<typename T>
struct Image<T, Ref> : public Ref {
        typedef T PIXEL_TYPE;
        uint2 size;

        Image() {
            size = make_uint2(0, 0);
        }
        Image(const uint2 & s, void * d) :
            Ref(d), size(s) {
        }

        __device__
        T & el() {
            return operator[](thr2pos2());
        }

        __device__
        const T & el() const {
            return operator[](thr2pos2());
        }

        __device__
        T & operator[](const uint2 & pos) {
            return static_cast<T *>(Ref::data)[pos.x + size.x * pos.y];
        }

        __device__
        const T & operator[](const uint2 & pos) const {
            return static_cast<const T *>(Ref::data)[pos.x + size.x * pos.y];
        }

        T * data() {
            return static_cast<T *>(Ref::data);
        }

        const T * data() const {
            return static_cast<const T *>(Ref::data);
        }
};

struct TrackData {
        int result;
        float error;
        float J[6];
};

struct KFusion
{

        void Init(const KFusionConfig & config); // allocates the volume and image data on the device
        void Clear();  // releases the allocated device memory

        void setPose(const Matrix4 & p); // sets the current pose of the camera

        // high level API to run a simple tracking - reconstruction loop
        void Reset(); // removes all reconstruction information

        template<typename A>
        void setDepth(const Image<float, A> & depth) // passes in a metric depth buffer as float array
        {
            abort();
        }

        void Raycast(); // Raycast the reference images to track against from the current pose
        int checkError();
};

int printCUDAError(); // print the last error
void synchroniseDevices(); // Synchronise CPU and GPU

// low level API without any state. These are the kernel functions

__global__ void initVolume(Volume volume, const float2 val);
__global__ void raycast(Image<float3> pos3D, Image<float3> normal,
                        const Volume volume, const Matrix4 view, const float nearPlane,
                        const float farPlane, const float step, const float largestep);
__global__ void integrate(Volume vol, const Image<float> depth,
                          const Matrix4 view, const float mu, const float maxweight);
__global__ void depth2vertex(Image<float3> vertex, const Image<float> depth,
                             const Matrix4 invK);
__global__ void vertex2normal(Image<float3> normal, const Image<float3> vertex);
__global__ void bilateral_filter(Image<float> out, const Image<float> in,
                                 const Image<float> gaussian, float e_d, int r);
__global__ void track(Image<TrackData> output, const Image<float3> inVertex,
                      const Image<float3> inNormal, const Image<float3> refVertex,
                      const Image<float3> refNormal, const Matrix4 Ttrack, const Matrix4 view,
                      const float dist_threshold, const float normal_threshold);
__global__ void reduce(float * out, const Image<TrackData> J, const uint2 size);
__global__ void trackAndReduce(float * out, const Image<float3> inVertex,
                               const Image<float3> inNormal, const Image<float3> refVertex,
                               const Image<float3> refNormal, const Matrix4 Ttrack, const Matrix4 view,
                               const float dist_threshold, const float normal_threshold);

__device__ __forceinline__ float4 raycast(const Volume volume, const uint2 pos,
                                          const Matrix4 view, const float nearPlane, const float farPlane,
                                          const float step, const float largestep)
{
    const float3 origin = view.get_translation();
    const float3 direction = rotate(view, make_float3(pos.x, pos.y, 1.f));

    // intersect ray with a box
    // http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
    // compute intersection of ray with all six bbox planes
    const float3 invR = make_float3(1.0f) / direction;
    const float3 tbot = -1 * invR * origin;
    const float3 ttop = invR * (volume.dim - origin);

    // re-order intersections to find smallest and largest on each axis
    const float3 tmin = fminf(ttop, tbot);
    const float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    const float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y),
                                     fmaxf(tmin.x, tmin.z));
    const float smallest_tmax = fminf(fminf(tmax.x, tmax.y),
                                      fminf(tmax.x, tmax.z));

    // check against near and far plane
    const float tnear = fmaxf(largest_tmin, nearPlane);
    const float tfar = fminf(smallest_tmax, farPlane);

    if (tnear < tfar)
    {
        // first walk with largesteps until we found a hit
        float t = tnear;
        float stepsize = largestep;
        float f_t = volume.interp(origin + direction * t);
        float f_tt = 0;

        if (f_t > 0) // ups, if we were already in it, then don't render anything here
        {
            for (; t < tfar; t += stepsize)
            {
                f_tt = volume.interp(origin + direction * t);
                if (f_tt < 0)                  // got it, jump out of inner loop
                    break;
                if (f_tt < 0.8f)               // coming closer, reduce stepsize
                    stepsize = step;
                f_t = f_tt;
            }
            if (f_tt < 0)
            {           // got it, calculate accurate intersection
                t = t + stepsize * f_tt / (f_t - f_tt);
                return make_float4(origin + direction * t, t);
            }
        }
    }
    return make_float4(0);
}

#endif // KFUSION_H
