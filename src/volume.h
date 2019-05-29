#ifndef VOLUME_H
#define VOLUME_H

#define IDX(a,b,c) a + b * size.x + c * size.x * size.y
struct Volume
{
    typedef float (Volume::*Fptr)(const uint3&) const;

    uint3 size;
    float3 dim;
    short2 *data;
    float3 *color;

    Volume() {
        size = make_uint3(0);
        dim = make_float3(1);
        data = NULL;
        color = NULL;
    }

    __device__
    float2 operator[](const uint3 & pos) const
    {
        const short2 d = data[pos.x + pos.y * size.x + pos.z * size.x * size.y];
        return make_float2(d.x * 0.00003051944088f, d.y); //  / 32766.0f
    }

    __device__
    float3 getColor(const uint3 & pos) const
    {
        return color[pos.x + pos.y * size.x + pos.z * size.x * size.y];
    }

    __device__
    float v(const uint3 & pos) const
    {
        return operator[](pos).x;
    }

    __device__
    float vs(const uint3 & pos) const
    {
        return data[pos.x + pos.y * size.x + pos.z * size.x * size.y].x;
    }

    __device__
    float red(const uint3 & pos) const
    {
        return color[pos.x + pos.y * size.x + pos.z * size.x * size.y].x;
    }

    __device__
    float green(const uint3 & pos) const
    {
        return color[pos.x + pos.y * size.x + pos.z * size.x * size.y].y;
    }

    __device__
    float blue(const uint3 & pos) const
    {
        return color[pos.x + pos.y * size.x + pos.z * size.x * size.y].z;
    }

    __device__
    void set(const uint3 & pos, const float2 & d)
    {
        uint idx=pos.x + pos.y * size.x + pos.z * size.x * size.y;
        data[idx] = make_short2(d.x * 32766.0f, d.y);
        color[idx] = make_float3(0.0,0.0,0.0);
    }

    __device__
    void set(const uint3 & pos, const float2 &d,const float3 &c)
    {
        data[pos.x + pos.y * size.x + pos.z * size.x * size.y] = make_short2(d.x * 32766.0f, d.y);
        color[pos.x + pos.y * size.x + pos.z * size.x * size.y] = c;
    }

    __device__
    float3 pos(const uint3 & p) const
    {
        return make_float3((p.x + 0.5f) * dim.x / size.x,
                           (p.y + 0.5f) * dim.y / size.y, (p.z + 0.5f) * dim.z / size.z);
    }

    __device__
    float interp(const float3 & pos) const
    {
        const Fptr fp = &Volume::vs;
        //TODO Do we need 0.00003051944088f??
        return generic_interp(pos,fp) * 0.00003051944088f;;
    }

    __device__
    float3 rgb_interp(const float3 & pos) const
    {

        float3 rgb;
        Fptr fptr = &Volume::red;
        rgb.x=generic_interp(pos,fptr);

        fptr = &Volume::green;
        rgb.y=generic_interp(pos,fptr);

        fptr = &Volume::blue;
        rgb.z=generic_interp(pos,fptr);
        return rgb;
    }

    __device__
    float generic_interp(const float3 & pos,const Fptr fp) const
    {
        const float3 scaled_pos = make_float3((pos.x * size.x / dim.x) - 0.5f,
                                              (pos.y * size.y / dim.y) - 0.5f,
                                              (pos.z * size.z / dim.z) - 0.5f);
        const int3 base = make_int3(floorf(scaled_pos));
        const float3 factor = fracf(scaled_pos);
        const int3 lower = max(base, make_int3(0));
        const int3 upper = min(base + make_int3(1),make_int3(size) - make_int3(1));

        float tmp0 =( (this->*fp) (make_uint3(lower.x, lower.y, lower.z)) * (1 - factor.x) +
                    (this->*fp) (make_uint3(upper.x, lower.y, lower.z)) * factor.x ) * (1 - factor.y);
        float tmp1 =( (this->*fp) (make_uint3(lower.x, upper.y, lower.z)) * (1 - factor.x) +
                    (this->*fp) (make_uint3(upper.x, upper.y, lower.z)) * factor.x) * factor.y ;
        float tmp2 =( (this->*fp) (make_uint3(lower.x, lower.y, upper.z)) * (1 - factor.x) +
                    (this->*fp) (make_uint3(upper.x, lower.y, upper.z)) * factor.x) * (1 - factor.y);
        float tmp3 =( (this->*fp) (make_uint3(lower.x, upper.y, upper.z)) * (1 - factor.x) +
                    (this->*fp) (make_uint3(upper.x, upper.y, upper.z)) * factor.x) * factor.y;

        return ( (tmp0+tmp1) * (1 - factor.z) + (tmp2+tmp3) * factor.z ) ;
    }

    __device__
    float3 grad(const float3 & pos) const
    {
        const float3 scaled_pos = make_float3((pos.x * size.x / dim.x) - 0.5f,
                                              (pos.y * size.y / dim.y) - 0.5f,
                                              (pos.z * size.z / dim.z) - 0.5f);
        const int3 base = make_int3(floorf(scaled_pos));
        const float3 factor = fracf(scaled_pos);
        const int3 lower_lower = max(base - make_int3(1), make_int3(0));
        const int3 lower_upper = max(base, make_int3(0));
        const int3 upper_lower = min(base + make_int3(1),
                                     make_int3(size) - make_int3(1));
        const int3 upper_upper = min(base + make_int3(2),
                                     make_int3(size) - make_int3(1));
        const int3 & lower = lower_upper;
        const int3 & upper = upper_lower;

        float3 gradient;

        gradient.x = ((
            ( vs(make_uint3(upper_lower.x, lower.y, lower.z))-vs(make_uint3(lower_lower.x, lower.y, lower.z))) * (1 - factor.x)
            + ( vs(make_uint3(upper_upper.x, lower.y, lower.z))-vs(make_uint3(lower_upper.x, lower.y, lower.z))) * factor.x) * (1 - factor.y)
            + ( (vs(make_uint3(upper_lower.x, upper.y, lower.z)) - vs(make_uint3(lower_lower.x, upper.y, lower.z)))* (1 - factor.x)
                + (vs(make_uint3(upper_upper.x, upper.y, lower.z))- vs(make_uint3(lower_upper.x, upper.y,lower.z))) * factor.x) * factor.y) * (1 - factor.z)
                     + (((vs(make_uint3(upper_lower.x, lower.y, upper.z))
                          - vs(make_uint3(lower_lower.x, lower.y, upper.z)))
                         * (1 - factor.x)
                         + (vs(make_uint3(upper_upper.x, lower.y, upper.z))
                            - vs(
                                make_uint3(lower_upper.x, lower.y,
                                           upper.z))) * factor.x)
                        * (1 - factor.y)
                        + ((vs(make_uint3(upper_lower.x, upper.y, upper.z))
                            - vs(
                                make_uint3(lower_lower.x, upper.y,
                                           upper.z))) * (1 - factor.x)
                           + (vs(
                                  make_uint3(upper_upper.x, upper.y,
                                             upper.z))
                              - vs(
                                  make_uint3(lower_upper.x,
                                             upper.y, upper.z)))
                           * factor.x) * factor.y) * factor.z;

        gradient.y =
                (((vs(make_uint3(lower.x, upper_lower.y, lower.z))
                   - vs(make_uint3(lower.x, lower_lower.y, lower.z)))
                  * (1 - factor.x)
                  + (vs(make_uint3(upper.x, upper_lower.y, lower.z))
                     - vs(
                         make_uint3(upper.x, lower_lower.y,
                                    lower.z))) * factor.x)
                 * (1 - factor.y)
                 + ((vs(make_uint3(lower.x, upper_upper.y, lower.z))
                     - vs(
                         make_uint3(lower.x, lower_upper.y,
                                    lower.z))) * (1 - factor.x)
                    + (vs(
                           make_uint3(upper.x, upper_upper.y,
                                      lower.z))
                       - vs(
                           make_uint3(upper.x,
                                      lower_upper.y, lower.z)))
                    * factor.x) * factor.y) * (1 - factor.z)
                + (((vs(make_uint3(lower.x, upper_lower.y, upper.z))
                     - vs(
                         make_uint3(lower.x, lower_lower.y,
                                    upper.z))) * (1 - factor.x)
                    + (vs(
                           make_uint3(upper.x, upper_lower.y,
                                      upper.z))
                       - vs(
                           make_uint3(upper.x,
                                      lower_lower.y, upper.z)))
                    * factor.x) * (1 - factor.y)
                   + ((vs(
                           make_uint3(lower.x, upper_upper.y,
                                      upper.z))
                       - vs(
                           make_uint3(lower.x,
                                      lower_upper.y, upper.z)))
                      * (1 - factor.x)
                      + (vs(
                             make_uint3(upper.x,
                                        upper_upper.y, upper.z))
                         - vs(
                             make_uint3(upper.x,
                                        lower_upper.y,
                                        upper.z)))
                      * factor.x) * factor.y)
                * factor.z;

        gradient.z = (((vs(make_uint3(lower.x, lower.y, upper_lower.z))
                        - vs(make_uint3(lower.x, lower.y, lower_lower.z)))
                       * (1 - factor.x)
                       + (vs(make_uint3(upper.x, lower.y, upper_lower.z))
                          - vs(make_uint3(upper.x, lower.y, lower_lower.z)))
                       * factor.x) * (1 - factor.y)
                      + ((vs(make_uint3(lower.x, upper.y, upper_lower.z))
                          - vs(make_uint3(lower.x, upper.y, lower_lower.z)))
                         * (1 - factor.x)
                         + (vs(make_uint3(upper.x, upper.y, upper_lower.z))
                            - vs(
                                make_uint3(upper.x, upper.y,
                                           lower_lower.z))) * factor.x)
                      * factor.y) * (1 - factor.z)
                     + (((vs(make_uint3(lower.x, lower.y, upper_upper.z))
                          - vs(make_uint3(lower.x, lower.y, lower_upper.z)))
                         * (1 - factor.x)
                         + (vs(make_uint3(upper.x, lower.y, upper_upper.z))
                            - vs(
                                make_uint3(upper.x, lower.y,
                                           lower_upper.z))) * factor.x)
                        * (1 - factor.y)
                        + ((vs(make_uint3(lower.x, upper.y, upper_upper.z))
                            - vs(
                                make_uint3(lower.x, upper.y,
                                           lower_upper.z)))
                           * (1 - factor.x)
                           + (vs(
                                  make_uint3(upper.x, upper.y,
                                             upper_upper.z))
                              - vs(
                                  make_uint3(upper.x, upper.y,
                                             lower_upper.z)))
                           * factor.x) * factor.y) * factor.z;

        return gradient
                * make_float3(dim.x / size.x, dim.y / size.y, dim.z / size.z)
                * (0.5f * 0.00003051944088f);
    }



    int init(uint3 s, float3 d)
    {
        size = s;
        dim = d;
        cudaFree(data);
        cudaFree(color);

        cudaError_t __cudaCalloc_err = cudaMalloc(&data,size.x * size.y * size.z * sizeof(short2));
        __cudaCalloc_err = cudaMalloc(&color,size.x * size.y * size.z * sizeof(float3));

        cudaMemset(data, 0, size.x * size.y * size.z * sizeof(short2));
        cudaMemset(color, 0, size.x * size.y * size.z * sizeof(float3));

        //TODO fix error checking
        return ((int) __cudaCalloc_err);

    }

    void release()
    {
        cudaFree(data);
        cudaFree(color);

        data = NULL;
        color=NULL;
    }
};

#endif // VOLUME_H
