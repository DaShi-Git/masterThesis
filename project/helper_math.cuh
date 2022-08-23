/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

#ifndef HELPER_MATH_H
#define HELPER_MATH_H

#ifndef __CUDACC__
#include "cuda_runtime.h"
#endif

typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifndef __CUDACC__
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline double fmin(double a, double b)
{
    return a < b ? a : b;
}

inline double fmax(double a, double b)
{
    return a > b ? a : b;
}

inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}

inline double rsqrt(double x)
{
	return 1.0 / sqrt(x);
}

#endif


////////////////////////////////////////////////////////////////////////////////
// CONSTANTS
////////////////////////////////////////////////////////////////////////////////
#ifndef M_E
#define M_E        2.71828182845904523536   // e
#endif
#ifndef M_LOG2E
#define M_LOG2E    1.44269504088896340736   // log2(e)
#endif
#ifndef M_LOG10E
#define M_LOG10E   0.434294481903251827651  // log10(e)
#endif
#ifndef M_LN2
#define M_LN2      0.693147180559945309417  // ln(2)
#endif
#ifndef M_LN10
#define M_LN10     2.30258509299404568402   // ln(10)
#endif
#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif
#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923   // pi/2
#endif
#ifndef M_PI_4
#define M_PI_4     0.785398163397448309616  // pi/4
#endif
#ifndef M_1_PI
#define M_1_PI     0.318309886183790671538  // 1/pi
#endif
#ifndef M_2_PI
#define M_2_PI     0.636619772367581343076  // 2/pi
#endif
#ifndef M_2_SQRTPI
#define M_2_SQRTPI 1.12837916709551257390   // 2/sqrt(pi)
#endif
#ifndef M_SQRT2
#define M_SQRT2    1.41421356237309504880   // sqrt(2)
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2  0.707106781186547524401  // 1/sqrt(2)
#endif


////////////////////////////////////////////////////////////////////////////////
// Type overloadings for real_t
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float square(float x)
{
    return x * x;
}
inline __host__ __device__ double square(double x)
{
    return x * x;
}
inline __host__ __device__ float cube(float x)
{
    return x * x * x;
}
inline __host__ __device__ double cube(double x)
{
    return x * x * x;
}

inline __host__ __device__ float sqrtr(float x)
{
    return sqrtf(x);
}
inline __host__ __device__ double sqrtr(double x)
{
    return sqrt(x);
}
inline __host__ __device__ float rsqrtr(float x)
{
    return rsqrtf(x);
}
inline __host__ __device__ double rsqrtr(double x)
{
    return rsqrt(x);
}

inline __host__ __device__ float rmin(float a, float b)
{
    return a < b ? a : b;
}

inline __host__ __device__ float rmax(float a, float b)
{
    return a > b ? a : b;
}

inline __host__ __device__ double rmin(double a, double b)
{
    return a < b ? a : b;
}

inline __host__ __device__ double rmax(double a, double b)
{
    return a > b ? a : b;
}

inline __host__ __device__ float rabs(float x) { return fabsf(x); }
inline __host__ __device__ double rabs(double x) { return fabs(x); }
inline __host__ __device__ float rexp(float x) { return expf(x); }
inline __host__ __device__ double rexp(double x) { return exp(x); }
inline __host__ __device__ float rlog(float x) { return logf(x); }
inline __host__ __device__ double rlog(double x) { return log(x); }
inline __host__ __device__ float rpow(float x, float y) { return powf(x, y); }
inline __host__ __device__ double rpow(double x, double y) { return pow(x, y); }

inline __host__ __device__ float rsin(float x) { return sinf(x); }
inline __host__ __device__ double rsin(double x) { return sin(x); }
inline __host__ __device__ float rasin(float x) { return asinf(x); }
inline __host__ __device__ double rasin(double x) { return asin(x); }

inline __host__ __device__ float rcos(float x) { return cosf(x); }
inline __host__ __device__ double rcos(double x) { return cos(x); }
inline __host__ __device__ float racos(float x) { return acosf(x); }
inline __host__ __device__ double racos(double x) { return acos(x); }

inline __host__ __device__ float rtan(float x) { return tanf(x); }
inline __host__ __device__ double rtan(double x) { return tan(x); }
inline __host__ __device__ float ratan(float x) { return atanf(x); }
inline __host__ __device__ double ratan(double x) { return atan(x); }

inline __host__ __device__ float cbrtr(float x) { return cbrtf(x); }
inline __host__ __device__ double cbrtr(double x) { return cbrt(x); }

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(float3 a)
{
    return make_float2(a.x, a.y);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}
inline __host__ __device__ float2 make_float2(uint2 a)
{
    return make_float2(float(a.x), float(a.y));
}

inline __host__ __device__ double2 make_double2(double s)
{
    return make_double2(s, s);
}
inline __host__ __device__ double2 make_double2(double3 a)
{
    return make_double2(a.x, a.y);
}
inline __host__ __device__ double2 make_double2(int2 a)
{
    return make_double2(double(a.x), double(a.y));
}
inline __host__ __device__ double2 make_double2(uint2 a)
{
    return make_double2(double(a.x), double(a.y));
}

inline __host__ __device__ int2 make_int2(int s)
{
    return make_int2(s, s);
}
inline __host__ __device__ int2 make_int2(int3 a)
{
    return make_int2(a.x, a.y);
}
inline __host__ __device__ int2 make_int2(uint2 a)
{
    return make_int2(int(a.x), int(a.y));
}
inline __host__ __device__ int2 make_int2(float2 a)
{
    return make_int2(int(a.x), int(a.y));
}

inline __host__ __device__ uint2 make_uint2(uint s)
{
    return make_uint2(s, s);
}
inline __host__ __device__ uint2 make_uint2(uint3 a)
{
    return make_uint2(a.x, a.y);
}
inline __host__ __device__ uint2 make_uint2(int2 a)
{
    return make_uint2(uint(a.x), uint(a.y));
}

inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float3 a)
{
    return a;
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}
inline __host__ __device__ float3 make_float3(double3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(uint3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline __host__ __device__ double3 make_double3(double s)
{
    return make_double3(s, s, s);
}
inline __host__ __device__ double3 make_double3(double2 a)
{
    return make_double3(a.x, a.y, 0.0f);
}
inline __host__ __device__ double3 make_double3(double2 a, double s)
{
    return make_double3(a.x, a.y, s);
}
inline __host__ __device__ double3 make_double3(double4 a)
{
    return make_double3(a.x, a.y, a.z);
}
inline __host__ __device__ double3 make_double3(int3 a)
{
    return make_double3(double(a.x), double(a.y), double(a.z));
}
inline __host__ __device__ double3 make_double3(uint3 a)
{
    return make_double3(double(a.x), double(a.y), double(a.z));
}
inline __host__ __device__ double3 make_double3(float3 a)
{
    return make_double3(double(a.x), double(a.y), double(a.z));
}

inline __host__ __device__ int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
inline __host__ __device__ int3 make_int3(int2 a)
{
    return make_int3(a.x, a.y, 0);
}
inline __host__ __device__ int3 make_int3(int2 a, int s)
{
    return make_int3(a.x, a.y, s);
}
inline __host__ __device__ int3 make_int3(uint3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ int3 make_int3(double3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

inline __host__ __device__ uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(uint2 a)
{
    return make_uint3(a.x, a.y, 0);
}
inline __host__ __device__ uint3 make_uint3(uint2 a, uint s)
{
    return make_uint3(a.x, a.y, s);
}
inline __host__ __device__ uint3 make_uint3(uint4 a)
{
    return make_uint3(a.x, a.y, a.z);
}
inline __host__ __device__ uint3 make_uint3(int3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(float2 a, float2 b)
{
    return make_float4(a.x, a.y, b.x, b.y);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline __host__ __device__ float4 make_float4(uint4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline __host__ __device__ float4 make_float4(float4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline __host__ __device__ float4 make_float4(double4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ double4 make_double4(double s)
{
    return make_double4(s, s, s, s);
}
inline __host__ __device__ double4 make_double4(double3 a)
{
    return make_double4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ double4 make_double4(double3 a, double w)
{
    return make_double4(a.x, a.y, a.z, w);
}
inline __host__ __device__ double4 make_double4(double2 a, double2 b)
{
    return make_double4(a.x, a.y, b.x, b.y);
}
inline __host__ __device__ double4 make_double4(int4 a)
{
    return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));
}
inline __host__ __device__ double4 make_double4(uint4 a)
{
    return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));
}

inline __host__ __device__ double4 make_double4(float4 a)
{
    return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));
}
inline __host__ __device__ double4 make_double4(double4 a)
{
    return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));
}
inline __host__ __device__ int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}
inline __host__ __device__ int4 make_int4(int3 a)
{
    return make_int4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ int4 make_int4(int3 a, int w)
{
    return make_int4(a.x, a.y, a.z, w);
}
inline __host__ __device__ int4 make_int4(uint4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline __host__ __device__ int4 make_int4(float4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline __host__ __device__ int4 make_int4(double4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}


inline __host__ __device__ uint4 make_uint4(uint s)
{
    return make_uint4(s, s, s, s);
}
inline __host__ __device__ uint4 make_uint4(uint3 a)
{
    return make_uint4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ uint4 make_uint4(uint3 a, uint w)
{
    return make_uint4(a.x, a.y, a.z, w);
}
inline __host__ __device__ uint4 make_uint4(int4 a)
{
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(const float2 &a)
{
    return make_float2(-a.x, -a.y);
}
inline __host__ __device__ double2 operator-(const double2& a)
{
    return make_double2(-a.x, -a.y);
}
inline __host__ __device__ int2 operator-(const int2 &a)
{
    return make_int2(-a.x, -a.y);
}
inline __host__ __device__ float3 operator-(const float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ double3 operator-(const double3& a)
{
    return make_double3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ int3 operator-(const int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float4 operator-(const float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ double4 operator-(const double4 &a)
{
    return make_double4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ int4 operator-(const int4 &a)
{
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(float2 &a, float b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ double2 operator+(double2 a, double2 b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(double2& a, double2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ double2 operator+(double2 a, double b)
{
    return make_double2(a.x + b, a.y + b);
}
inline __host__ __device__ double2 operator+(double b, double2 a)
{
    return make_double2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(double2& a, double b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ int2 operator+(int2 a, int b)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ int2 operator+(int b, int2 a)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(int2 &a, int b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(uint2 &a, uint2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ uint2 operator+(uint2 a, uint b)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ uint2 operator+(uint b, uint2 a)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(uint2 &a, uint b)
{
    a.x += b;
    a.y += b;
}


inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(double3 &a, double3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ double3 operator+(double3 a, double b)
{
    return make_double3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(double3 &a, double b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3 &a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ int3 operator+(int3 a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(int3 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ uint3 operator+(uint3 a, uint b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(uint3 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int b, int3 a)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ uint3 operator+(uint b, uint3 a)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ float4 operator+(float b, float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ double4 operator+(double4 a, double4 b)
{
    return make_double4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(double4 &a, double4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ double4 operator+(double4 a, double b)
{
    return make_double4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ double4 operator+(double b, double4 a)
{
    return make_double4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(double4 &a, double b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(int4 &a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ int4 operator+(int4 a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ int4 operator+(int b, int4 a)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(int4 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(uint4 &a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ uint4 operator+(uint4 a, uint b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ uint4 operator+(uint b, uint4 a)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(uint4 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(float b, float2 a)
{
    return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2 &a, float b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ double2 operator-(double2 a, double2 b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(double2& a, double2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ double2 operator-(double2 a, double b)
{
    return make_double2(a.x - b, a.y - b);
}
inline __host__ __device__ double2 operator-(double b, double2 a)
{
    return make_double2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(double2& a, double b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}
inline __host__ __device__ int2 operator-(int b, int2 a)
{
    return make_int2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(int2 &a, int b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ uint2 operator-(uint2 a, uint2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ uint2 operator-(uint2 a, uint b)
{
    return make_uint2(a.x - b, a.y - b);
}
inline __host__ __device__ uint2 operator-(uint b, uint2 a)
{
    return make_uint2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(double3& a, double3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ double3 operator-(double3 a, double b)
{
    return make_double3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ double3 operator-(double b, double3 a)
{
    return make_double3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(double3& a, double b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(int3 &a, int3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ int3 operator-(int3 a, int b)
{
    return make_int3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ int3 operator-(int b, int3 a)
{
    return make_int3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(int3 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ uint3 operator-(uint3 a, uint b)
{
    return make_uint3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ uint3 operator-(uint b, uint3 a)
{
    return make_uint3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ double4 operator-(double4 a, double4 b)
{
    return make_double4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(double4 &a, double4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ double4 operator-(double4 a, double b)
{
    return make_double4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ void operator-=(double4 &a, double b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ int4 operator-(int4 a, int4 b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(int4 &a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ int4 operator-(int4 a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ int4 operator-(int b, int4 a)
{
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(int4 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ uint4 operator-(uint4 a, uint4 b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ uint4 operator-(uint4 a, uint b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ uint4 operator-(uint b, uint4 a)
{
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ double2 operator*(double2 a, double2 b)
{
    return make_double2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(double2 &a, double2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ double2 operator*(double2 a, double b)
{
    return make_double2(a.x * b, a.y * b);
}
inline __host__ __device__ double2 operator*(double b, double2 a)
{
    return make_double2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(double2 &a, double b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(int2 &a, int2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ int2 operator*(int2 a, int b)
{
    return make_int2(a.x * b, a.y * b);
}
inline __host__ __device__ int2 operator*(int b, int2 a)
{
    return make_int2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(int2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ uint2 operator*(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ uint2 operator*(uint2 a, uint b)
{
    return make_uint2(a.x * b, a.y * b);
}
inline __host__ __device__ uint2 operator*(uint b, uint2 a)
{
    return make_uint2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(double3 &a, double3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ double3 operator*(double3 a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ double3 operator*(double b, double3 a)
{
    return make_double3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(double3 &a, double b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(int3 &a, int3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ int3 operator*(int3 a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ int3 operator*(int b, int3 a)
{
    return make_int3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(int3 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ uint3 operator*(uint3 a, uint b)
{
    return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ uint3 operator*(uint b, uint3 a)
{
    return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ double4 operator*(double4 a, double4 b)
{
    return make_double4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(double4 &a, double4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ double4 operator*(double4 a, double b)
{
    return make_double4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ double4 operator*(double b, double4 a)
{
    return make_double4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(double4 &a, double b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ int4 operator*(int4 a, int4 b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(int4 &a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(int4 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ uint4 operator*(uint4 a, uint b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ uint4 operator*(uint b, uint4 a)
{
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ float2 operator/(float b, float2 a)
{
    return make_float2(b / a.x, b / a.y);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ double3 operator/(double3 a, double3 b)
{
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(double3 &a, double3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ double3 operator/(double3 a, double b)
{
    return make_double3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(double3 &a, double b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ double3 operator/(double b, double3 a)
{
    return make_double3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
inline __host__ __device__ void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline __host__ __device__ float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline __host__ __device__ void operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline __host__ __device__ float4 operator/(float b, float4 a)
{
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

inline __host__ __device__ double4 operator/(double4 a, double4 b)
{
    return make_double4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ void operator/=(double4 &a, double4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline __host__ __device__ double4 operator/(double4 a, double b)
{
    return make_double4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __host__ __device__ void operator/=(double4 &a, double b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline __host__ __device__ double4 operator/(double b, double4 a)
{
    return make_double4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// comparison
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ int2 operator==(float2 a, float2 b)
{
	return make_int2(a.x == b.x, a.y == b.y);
}
inline __host__ __device__ int2 operator!=(float2 a, float2 b)
{
	return make_int2(a.x != b.x, a.y != b.y);
}
inline __host__ __device__ int2 operator<(float2 a, float2 b)
{
	return make_int2(a.x < b.x, a.y < b.y);
}
inline __host__ __device__ int2 operator<=(float2 a, float2 b)
{
	return make_int2(a.x <= b.x, a.y <= b.y);
}
inline __host__ __device__ int2 operator>(float2 a, float2 b)
{
	return make_int2(a.x > b.x, a.y > b.y);
}
inline __host__ __device__ int2 operator>=(float2 a, float2 b)
{
	return make_int2(a.x >= b.x, a.y >= b.y);
}

inline __host__ __device__ int3 operator==(float3 a, float3 b)
{
	return make_int3(a.x == b.x, a.y == b.y, a.z == b.z);
}
inline __host__ __device__ int3 operator!=(float3 a, float3 b)
{
	return make_int3(a.x != b.x, a.y != b.y, a.z != b.z);
}
inline __host__ __device__ int3 operator<(float3 a, float3 b)
{
	return make_int3(a.x < b.x, a.y < b.y, a.z < b.z);
}
inline __host__ __device__ int3 operator<=(float3 a, float3 b)
{
	return make_int3(a.x <= b.x, a.y <= b.y, a.z <= b.z);
}
inline __host__ __device__ int3 operator>(float3 a, float3 b)
{
	return make_int3(a.x > b.x, a.y > b.y, a.z > b.z);
}
inline __host__ __device__ int3 operator>=(float3 a, float3 b)
{
	return make_int3(a.x >= b.x, a.y >= b.y, a.z >= b.z);
}

inline __host__ __device__ int4 operator==(float4 a, float4 b)
{
	return make_int4(a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w);
}
inline __host__ __device__ int4 operator!=(float4 a, float4 b)
{
	return make_int4(a.x != b.x, a.y != b.y, a.z != b.z, a.w != b.w);
}
inline __host__ __device__ int4 operator<(float4 a, float4 b)
{
	return make_int4(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
}
inline __host__ __device__ int4 operator<=(float4 a, float4 b)
{
	return make_int4(a.x <= b.x, a.y <= b.y, a.z <= b.z, a.w <= b.w);
}
inline __host__ __device__ int4 operator>(float4 a, float4 b)
{
	return make_int4(a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w);
}
inline __host__ __device__ int4 operator>=(float4 a, float4 b)
{
	return make_int4(a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w);
}

inline __host__ __device__ int2 operator==(double2 a, double2 b)
{
    return make_int2(a.x == b.x, a.y == b.y);
}
inline __host__ __device__ int2 operator!=(double2 a, double2 b)
{
    return make_int2(a.x != b.x, a.y != b.y);
}
inline __host__ __device__ int2 operator<(double2 a, double2 b)
{
    return make_int2(a.x < b.x, a.y < b.y);
}
inline __host__ __device__ int2 operator<=(double2 a, double2 b)
{
    return make_int2(a.x <= b.x, a.y <= b.y);
}
inline __host__ __device__ int2 operator>(double2 a, double2 b)
{
    return make_int2(a.x > b.x, a.y > b.y);
}
inline __host__ __device__ int2 operator>=(double2 a, double2 b)
{
    return make_int2(a.x >= b.x, a.y >= b.y);
}

inline __host__ __device__ int3 operator==(double3 a, double3 b)
{
    return make_int3(a.x == b.x, a.y == b.y, a.z == b.z);
}
inline __host__ __device__ int3 operator!=(double3 a, double3 b)
{
    return make_int3(a.x != b.x, a.y != b.y, a.z != b.z);
}
inline __host__ __device__ int3 operator<(double3 a, double3 b)
{
    return make_int3(a.x < b.x, a.y < b.y, a.z < b.z);
}
inline __host__ __device__ int3 operator<=(double3 a, double3 b)
{
    return make_int3(a.x <= b.x, a.y <= b.y, a.z <= b.z);
}
inline __host__ __device__ int3 operator>(double3 a, double3 b)
{
    return make_int3(a.x > b.x, a.y > b.y, a.z > b.z);
}
inline __host__ __device__ int3 operator>=(double3 a, double3 b)
{
    return make_int3(a.x >= b.x, a.y >= b.y, a.z >= b.z);
}

inline __host__ __device__ int4 operator==(double4 a, double4 b)
{
    return make_int4(a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w);
}
inline __host__ __device__ int4 operator!=(double4 a, double4 b)
{
    return make_int4(a.x != b.x, a.y != b.y, a.z != b.z, a.w != b.w);
}
inline __host__ __device__ int4 operator<(double4 a, double4 b)
{
    return make_int4(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
}
inline __host__ __device__ int4 operator<=(double4 a, double4 b)
{
    return make_int4(a.x <= b.x, a.y <= b.y, a.z <= b.z, a.w <= b.w);
}
inline __host__ __device__ int4 operator>(double4 a, double4 b)
{
    return make_int4(a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w);
}
inline __host__ __device__ int4 operator>=(double4 a, double4 b)
{
    return make_int4(a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w);
}

inline __host__ __device__ int3 operator==(int3 a, int3 b)
{
    return make_int3(a.x == b.x, a.y == b.y, a.z == b.z);
}
inline __host__ __device__ int3 operator!=(int3 a, int3 b)
{
    return make_int3(a.x != b.x, a.y != b.y, a.z != b.z);
}
inline __host__ __device__ int3 operator<(int3 a, int3 b)
{
    return make_int3(a.x < b.x, a.y < b.y, a.z < b.z);
}
inline __host__ __device__ int3 operator<=(int3 a, int3 b)
{
    return make_int3(a.x <= b.x, a.y <= b.y, a.z <= b.z);
}
inline __host__ __device__ int3 operator>(int3 a, int3 b)
{
    return make_int3(a.x > b.x, a.y > b.y, a.z > b.z);
}
inline __host__ __device__ int3 operator>=(int3 a, int3 b)
{
    return make_int3(a.x >= b.x, a.y >= b.y, a.z >= b.z);
}

inline __host__ __device__ int4 operator==(int4 a, int4 b)
{
    return make_int4(a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w);
}
inline __host__ __device__ int4 operator!=(int4 a, int4 b)
{
    return make_int4(a.x != b.x, a.y != b.y, a.z != b.z, a.w != b.w);
}
inline __host__ __device__ int4 operator<(int4 a, int4 b)
{
    return make_int4(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
}
inline __host__ __device__ int4 operator<=(int4 a, int4 b)
{
    return make_int4(a.x <= b.x, a.y <= b.y, a.z <= b.z, a.w <= b.w);
}
inline __host__ __device__ int4 operator>(int4 a, int4 b)
{
    return make_int4(a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w);
}
inline __host__ __device__ int4 operator>=(int4 a, int4 b)
{
    return make_int4(a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w);
}

inline __host__ __device__ bool all(int2 v)
{
	return v.x && v.y;
}
inline __host__ __device__ bool all(int3 v)
{
	return v.x && v.y && v.z;
}
inline __host__ __device__ bool all(int4 v)
{
	return v.x && v.y && v.z && v.w;
}

inline __host__ __device__ bool any(int2 v)
{
	return v.x || v.y;
}
inline __host__ __device__ bool any(int3 v)
{
	return v.x || v.y || v.z;
}
inline __host__ __device__ bool any(int4 v)
{
	return v.x || v.y || v.z || v.w;
}

////////////////////////////////////////////////////////////////////////////////
// NaN
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ int2 isnan(float2 a)
{
    return make_int2(isnan(a.x) ? 1 : 0, isnan(a.y) ? 1 : 0);
}
inline __host__ __device__ int3 isnan(float3 a)
{
    return make_int3(isnan(a.x) ? 1 : 0, isnan(a.y) ? 1 : 0, isnan(a.z) ? 1 : 0);
}
inline __host__ __device__ int4 isnan(float4 a)
{
    return make_int4(isnan(a.x) ? 1 : 0, isnan(a.y) ? 1 : 0, isnan(a.z) ? 1 : 0, isnan(a.w) ? 1 : 0);
}
inline __host__ __device__ int2 isnan(double2 a)
{
    return make_int2(isnan(a.x) ? 1 : 0, isnan(a.y) ? 1 : 0);
}
inline __host__ __device__ int3 isnan(double3 a)
{
    return make_int3(isnan(a.x) ? 1 : 0, isnan(a.y) ? 1 : 0, isnan(a.z) ? 1 : 0);
}
inline __host__ __device__ int4 isnan(double4 a)
{
    return make_int4(isnan(a.x) ? 1 : 0, isnan(a.y) ? 1 : 0, isnan(a.z) ? 1 : 0, isnan(a.w) ? 1 : 0);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline  __host__ __device__ float2 fminf(float2 a, float2 b)
{
    return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}
inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline  __host__ __device__ float4 fminf(float4 a, float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

inline  __host__ __device__ double2 fmin(double2 a, double2 b)
{
    return make_double2(fmin(a.x, b.x), fmin(a.y, b.y));
}
inline __host__ __device__ double3 fmin(double3 a, double3 b)
{
    return make_double3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
}
inline  __host__ __device__ double4 fmin(double4 a, double4 b)
{
    return make_double4(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z), fmin(a.w, b.w));
}

inline __host__ __device__ int2 min(int2 a, int2 b)
{
    return make_int2(min(a.x,b.x), min(a.y,b.y));
}
inline __host__ __device__ int3 min(int3 a, int3 b)
{
    return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline __host__ __device__ int4 min(int4 a, int4 b)
{
    return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

inline __host__ __device__ uint2 min(uint2 a, uint2 b)
{
    return make_uint2(min(a.x,b.x), min(a.y,b.y));
}
inline __host__ __device__ uint3 min(uint3 a, uint3 b)
{
    return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline __host__ __device__ uint4 min(uint4 a, uint4 b)
{
    return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmaxf(float2 a, float2 b)
{
    return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline __host__ __device__ float4 fmaxf(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

inline __host__ __device__ double2 fmax(double2 a, double2 b)
{
    return make_double2(fmax(a.x, b.x), fmax(a.y, b.y));
}
inline __host__ __device__ double3 fmax(double3 a, double3 b)
{
    return make_double3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));
}
inline __host__ __device__ double4 fmax(double4 a, double4 b)
{
    return make_double4(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z), fmax(a.w, b.w));
}

inline __host__ __device__ int2 max(int2 a, int2 b)
{
    return make_int2(max(a.x,b.x), max(a.y,b.y));
}
inline __host__ __device__ int3 max(int3 a, int3 b)
{
    return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline __host__ __device__ int4 max(int4 a, int4 b)
{
    return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

inline __host__ __device__ uint2 max(uint2 a, uint2 b)
{
    return make_uint2(max(a.x,b.x), max(a.y,b.y));
}
inline __host__ __device__ uint3 max(uint3 a, uint3 b)
{
    return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline __host__ __device__ uint4 max(uint4 a, uint4 b)
{
    return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

inline __host__ __device__ double3 rmax(double3 a, double3 b)
{
    return make_double3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));
}
inline __host__ __device__ float3 rmax(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
inline __host__ __device__ double4 rmax(double4 a, double4 b)
{
    return make_double4(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z), fmax(a.w, b.w));
}
inline __host__ __device__ float4 rmax(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max/min coeff
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float max_coeff(float2 a)
{
    return fmaxf(a.x, a.y);
}
inline __host__ __device__ float max_coeff(float3 a)
{
    return fmaxf(a.x, fmaxf(a.y, a.z));
}
inline __host__ __device__ float max_coeff(float4 a)
{
    return fmaxf(a.x, fmaxf(a.y, fmaxf(a.z, a.w)));
}

inline __host__ __device__ double max_coeff(double2 a)
{
    return fmax(a.x, a.y);
}
inline __host__ __device__ double max_coeff(double3 a)
{
    return fmax(a.x, fmax(a.y, a.z));
}
inline __host__ __device__ double max_coeff(double4 a)
{
    return fmax(a.x, fmax(a.y, fmax(a.z, a.w)));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float2 lerp(float2 a, float2 b, float2 t)
{
    return a + t * (b - a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float3 t)
{
    return a + t * (b - a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float4 t)
{
    return a + t * (b - a);
}

inline __device__ __host__ double lerp(double a, double b, double t)
{
    return a + t * (b - a);
}
inline __device__ __host__ double2 lerp(double2 a, double2 b, double t)
{
    return a + t * (b - a);
}
inline __device__ __host__ double3 lerp(double3 a, double3 b, double t)
{
    return a + t * (b - a);
}
inline __device__ __host__ double4 lerp(double4 a, double4 b, double t)
{
    return a + t * (b - a);
}
inline __device__ __host__ double2 lerp(double2 a, double2 b, double2 t)
{
    return a + t * (b - a);
}
inline __device__ __host__ double3 lerp(double3 a, double3 b, double3 t)
{
    return a + t * (b - a);
}
inline __device__ __host__ double4 lerp(double4 a, double4 b, double4 t)
{
    return a + t * (b - a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}
inline __device__ __host__ double clamp(double f, double a, double b)
{
    return fmax(a, fmin(f, b));
}
inline __device__ __host__ int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}
inline __device__ __host__ uint clamp(uint f, uint a, uint b)
{
    return max(a, min(f, b));
}

inline __device__ __host__ float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ double2 clamp(double2 v, double a, double b)
{
    return make_double2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ double2 clamp(double2 v, double2 a, double2 b)
{
    return make_double2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ double3 clamp(double3 v, double a, double b)
{
    return make_double3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ double3 clamp(double3 v, double3 a, double3 b)
{
    return make_double3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ double4 clamp(double4 v, double a, double b)
{
    return make_double4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ double4 clamp(double4 v, double4 a, double4 b)
{
    return make_double4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ int2 clamp(int2 v, int a, int b)
{
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b)
{
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ int4 clamp(int4 v, int a, int b)
{
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b)
{
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b)
{
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b)
{
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b)
{
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b)
{
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline __host__ __device__ float dot3(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ double dot(double2 a, double2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ double dot(double4 a, double4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline __host__ __device__ double dot3(double4 a, double4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ int dot(int2 a, int2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ int dot(int3 a, int3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ int dot(int4 a, int4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ uint dot(uint2 a, uint2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ uint dot(uint3 a, uint3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ uint dot(uint4 a, uint4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// sum + prod
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float sum(float2 a)
{
    return a.x + a.y;
}
inline __host__ __device__ float sum(float3 a)
{
    return a.x + a.y + a.z;
}
inline __host__ __device__ float sum(float4 a)
{
    return a.x + a.y + a.z + a.w;
}
inline __host__ __device__ double sum(double2 a)
{
    return a.x + a.y;
}
inline __host__ __device__ double sum(double3 a)
{
    return a.x + a.y + a.z;
}
inline __host__ __device__ double sum(double4 a)
{
    return a.x + a.y + a.z + a.w;
}
inline __host__ __device__ int sum(int2 a)
{
    return a.x + a.y;
}
inline __host__ __device__ int sum(int3 a)
{
    return a.x + a.y + a.z;
}
inline __host__ __device__ int sum(int4 a)
{
    return a.x + a.y + a.z + a.w;
}

inline __host__ __device__ float prod(float2 a)
{
    return a.x * a.y;
}
inline __host__ __device__ float prod(float3 a)
{
    return a.x * a.y * a.z;
}
inline __host__ __device__ float prod(float4 a)
{
    return a.x * a.y * a.z * a.w;
}
inline __host__ __device__ double prod(double2 a)
{
    return a.x * a.y;
}
inline __host__ __device__ double prod(double3 a)
{
    return a.x * a.y * a.z;
}
inline __host__ __device__ double prod(double4 a)
{
    return a.x * a.y * a.z * a.w;
}
inline __host__ __device__ int prod(int2 a)
{
    return a.x * a.y;
}
inline __host__ __device__ int prod(int3 a)
{
    return a.x * a.y * a.z;
}
inline __host__ __device__ int prod(int4 a)
{
    return a.x * a.y * a.z * a.w;
}


////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float lengthSquared(float2 v)
{
	return dot(v, v);
}
inline __host__ __device__ float lengthSquared(float3 v)
{
	return dot(v, v);
}
inline __host__ __device__ float lengthSquared(float4 v)
{
	return dot(v, v);
}
inline __host__ __device__ float lengthSquared3(float4 v)
{
    return dot3(v, v);
}

inline __host__ __device__ double lengthSquared(double2 v)
{
    return dot(v, v);
}
inline __host__ __device__ double lengthSquared(double3 v)
{
    return dot(v, v);
}
inline __host__ __device__ double lengthSquared(double4 v)
{
    return dot(v, v);
}
inline __host__ __device__ double lengthSquared3(double4 v)
{
    return dot3(v, v);
}

inline __host__ __device__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float4 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length3(float4 v)
{
    return sqrtf(dot3(v, v));
}
inline __host__ __device__ double length(double2 v)
{
    return sqrt(dot(v, v));
}
inline __host__ __device__ double length(double3 v)
{
    return sqrt(dot(v, v));
}
inline __host__ __device__ double length(double4 v)
{
    return sqrt(dot(v, v));
}
inline __host__ __device__ double length3(double4 v)
{
    return sqrt(dot3(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float4 normalize3(float4 v)
{
    float invLen = rsqrtf(dot3(v, v));
    return v * invLen;
}

inline __host__ __device__ double2 normalize(double2 v)
{
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ double3 normalize(double3 v)
{
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ double4 normalize(double4 v)
{
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ double4 normalize3(double4 v)
{
    double invLen = rsqrt(dot3(v, v));
    return v * invLen;
}

inline __host__ __device__ float2 safeNormalize(float2 v)
{
    if (lengthSquared(v) < 1e-8) return v; //almost zero
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float3 safeNormalize(float3 v)
{
    if (lengthSquared(v) < 1e-8) return v; //almost zero
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float4 safeNormalize(float4 v)
{
    if (lengthSquared(v) < 1e-8) return v; //almost zero
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float4 safeNormalize3(float4 v)
{
    if (lengthSquared3(v) < 1e-8) return v; //almost zero
    float invLen = rsqrtf(dot3(v, v));
    return v * invLen;
}

inline __host__ __device__ double2 safeNormalize(double2 v)
{
    if (lengthSquared(v) < 1e-15) return v; //almost zero
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ double3 safeNormalize(double3 v)
{
    if (lengthSquared(v) < 1e-15) return v; //almost zero
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ double4 safeNormalize(double4 v)
{
    if (lengthSquared(v) < 1e-15) return v; //almost zero
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ double4 safeNormalize3(double4 v)
{
    if (lengthSquared3(v) < 1e-15) return v; //almost zero
    double invLen = rsqrt(dot3(v, v));
    return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 floorf(float2 v)
{
    return make_float2(floorf(v.x), floorf(v.y));
}
inline __host__ __device__ float3 floorf(float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ float4 floorf(float4 v)
{
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

inline __host__ __device__ float2 floor(float2 v)
{
    return make_float2(floorf(v.x), floorf(v.y));
}
inline __host__ __device__ float3 floor(float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ float4 floor(float4 v)
{
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

inline __host__ __device__ double2 floor(double2 v)
{
    return make_double2(floor(v.x), floor(v.y));
}
inline __host__ __device__ double3 floor(double3 v)
{
    return make_double3(floor(v.x), floor(v.y), floor(v.z));
}
inline __host__ __device__ double4 floor(double4 v)
{
    return make_double4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// ceil
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 ceilf(float2 v)
{
	return make_float2(ceilf(v.x), ceilf(v.y));
}
inline __host__ __device__ float3 ceilf(float3 v)
{
	return make_float3(ceilf(v.x), ceilf(v.y), ceilf(v.z));
}
inline __host__ __device__ float4 ceilf(float4 v)
{
	return make_float4(ceilf(v.x), ceilf(v.y), ceilf(v.z), ceilf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// round
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 roundf(float2 v)
{
	return make_float2(roundf(v.x), roundf(v.y));
}
inline __host__ __device__ float3 roundf(float3 v)
{
	return make_float3(roundf(v.x), roundf(v.y), roundf(v.z));
}
inline __host__ __device__ float4 roundf(float4 v)
{
	return make_float4(roundf(v.x), roundf(v.y), roundf(v.z), roundf(v.w));
}

inline __host__ __device__ float2 round(float2 v)
{
    return make_float2(roundf(v.x), roundf(v.y));
}
inline __host__ __device__ float3 round(float3 v)
{
    return make_float3(roundf(v.x), roundf(v.y), roundf(v.z));
}
inline __host__ __device__ float4 round(float4 v)
{
    return make_float4(roundf(v.x), roundf(v.y), roundf(v.z), roundf(v.w));
}

inline __host__ __device__ double2 round(double2 v)
{
    return make_double2(round(v.x), round(v.y));
}
inline __host__ __device__ double3 round(double3 v)
{
    return make_double3(round(v.x), round(v.y), round(v.z));
}
inline __host__ __device__ double4 round(double4 v)
{
    return make_double4(round(v.x), round(v.y), round(v.z), round(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float fracf(float v)
{
    return v - floorf(v);
}
inline __host__ __device__ float2 fracf(float2 v)
{
    return make_float2(fracf(v.x), fracf(v.y));
}
inline __host__ __device__ float3 fracf(float3 v)
{
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ float4 fracf(float4 v)
{
    return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmodf(float2 a, float2 b)
{
    return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
inline __host__ __device__ float3 fmodf(float3 a, float3 b)
{
    return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline __host__ __device__ float4 fmodf(float4 a, float4 b)
{
    return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fabs(float2 v)
{
    return make_float2(fabsf(v.x), fabsf(v.y));
}
inline __host__ __device__ float3 fabs(float3 v)
{
    return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}
inline __host__ __device__ float4 fabs(float4 v)
{
    return make_float4(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w));
}

inline __host__ __device__ int2 abs(int2 v)
{
    return make_int2(abs(v.x), abs(v.y));
}
inline __host__ __device__ int3 abs(int3 v)
{
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
inline __host__ __device__ int4 abs(int4 v)
{
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// pow
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 rpow(float2 a, float2 b)
{
    return make_float2(powf(a.x, b.x), powf(a.y, b.y));
}
inline __host__ __device__ float3 rpow(float3 a, float3 b)
{
    return make_float3(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z));
}
inline __host__ __device__ float4 rpow(float4 a, float4 b)
{
    return make_float4(powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z), powf(a.w, b.w));
}

inline __host__ __device__ double2 rpow(double2 a, double2 b)
{
    return make_double2(pow(a.x, b.x), pow(a.y, b.y));
}
inline __host__ __device__ double3 rpow(double3 a, double3 b)
{
    return make_double3(pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z));
}
inline __host__ __device__ double4 rpow(double4 a, double4 b)
{
    return make_double4(pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z), pow(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline __host__ __device__ float4 cross(float4 a, float4 b)
{
	return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0);
}
inline __host__ __device__ double3 cross(double3 a, double3 b)
{
	return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline __host__ __device__ double4 cross(double4 a, double4 b)
{
	return make_double4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float smoothstep(float a, float b, float x)
{
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(3.0f - (2.0f*y)));
}
inline __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x)
{
    float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float2(3.0f) - (make_float2(2.0f)*y)));
}
inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x)
{
    float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float3(3.0f) - (make_float3(2.0f)*y)));
}
inline __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x)
{
    float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float4(3.0f) - (make_float4(2.0f)*y)));
}

////////////////////////////////////////////////////////////////////////////////
// log, exp
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float2 rexp(float2 x)
{
    return make_float2(rexp(x.x), rexp(x.y));
}
inline __device__ __host__ float2 rlog(float2 x)
{
    return make_float2(rlog(x.x), rlog(x.y));
}
inline __device__ __host__ float3 rexp(float3 x)
{
    return make_float3(rexp(x.x), rexp(x.y), rexp(x.z));
}
inline __device__ __host__ float3 rlog(float3 x)
{
    return make_float3(rlog(x.x), rlog(x.y), rlog(x.z));
}
inline __device__ __host__ float4 rexp(float4 x)
{
    return make_float4(rexp(x.x), rexp(x.y), rexp(x.z), rexp(x.w));
}
inline __device__ __host__ float4 rlog(float4 x)
{
    return make_float4(rlog(x.x), rlog(x.y), rlog(x.z), rlog(x.w));
}

inline __device__ __host__ double2 rexp(double2 x)
{
    return make_double2(rexp(x.x), rexp(x.y));
}
inline __device__ __host__ double2 rlog(double2 x)
{
    return make_double2(rlog(x.x), rlog(x.y));
}
inline __device__ __host__ double3 rexp(double3 x)
{
    return make_double3(rexp(x.x), rexp(x.y), rexp(x.z));
}
inline __device__ __host__ double3 rlog(double3 x)
{
    return make_double3(rlog(x.x), rlog(x.y), rlog(x.z));
}
inline __device__ __host__ double4 rexp(double4 x)
{
    return make_double4(rexp(x.x), rexp(x.y), rexp(x.z), rexp(x.w));
}
inline __device__ __host__ double4 rlog(double4 x)
{
    return make_double4(rlog(x.x), rlog(x.y), rlog(x.z), rlog(x.w));
}

////////////////////////////////////////////////////////////////////////////////
// sign -> in {-1, +1}
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float2 sign(float2 v)
{
    return make_float2(copysign(1.f, v.x), copysign(1.f, v.y));
}
inline __device__ __host__ float3 sign(float3 v)
{
    return make_float3(copysign(1.f, v.x), copysign(1.f, v.y), copysign(1.f, v.z));
}
inline __device__ __host__ float4 sign(float4 v)
{
    return make_float4(copysign(1.f, v.x), copysign(1.f, v.y), copysign(1.f, v.z), copysign(1.f, v.w));
}

inline __device__ __host__ double2 sign(double2 v)
{
    return make_double2(copysign(1., v.x), copysign(1., v.y));
}
inline __device__ __host__ double3 sign(double3 v)
{
    return make_double3(copysign(1., v.x), copysign(1., v.y), copysign(1., v.z));
}
inline __device__ __host__ double4 sign(double4 v)
{
    return make_double4(copysign(1., v.x), copysign(1., v.y), copysign(1., v.z), copysign(1., v.w));
}

////////////////////////////////////////////////////////////////////////////////
// blending
////////////////////////////////////////////////////////////////////////////////

//front-to-back alpha blending
inline __device__ __host__ float4 blend(float4 rgbaFront, float4 rgbaBack)
{
    float3 rgbFront = make_float3(rgbaFront);
    float aFront = rgbaFront.w;
    float3 rgbBack = make_float3(rgbaBack);
    float aBack = rgbaBack.w;

    return make_float4(
        rgbFront + (1 - aFront) * rgbBack,
        aFront + (1 - aFront) * aBack
    );
}
inline __device__ __host__ double4 blend(double4 rgbaFront, double4 rgbaBack)
{
    double3 rgbFront = make_double3(rgbaFront);
    double aFront = rgbaFront.w;
    double3 rgbBack = make_double3(rgbaBack);
    double aBack = rgbaBack.w;

    return make_double4(
        rgbFront + (1 - aFront) * rgbBack,
        aFront + (1 - aFront) * aBack
    );
}

////////////////////////////////////////////////////////////////////////////////
// Special Functions
////////////////////////////////////////////////////////////////////////////////

//https://stackoverflow.com/a/40260471/1786598
inline __device__ __host__ float myErfInv(float x)
{
#if !defined(__CUDA_ARCH__)
    using namespace std;
#endif
    float tt1, tt2, lnx, sgn;
    sgn = (x < 0) ? -1.0f : 1.0f;

    x = (1 - x) * (1 + x);        // x = 1 - x*x;
    lnx = logf(x);

    constexpr float a = 0.15449436008930206298828125f;
    constexpr float b = 3.1415926535897932384626433832f * a;

    tt1 = 2 / b + 0.5f * lnx;
    tt2 = 1 / a * lnx;

    return(sgn * sqrtf(-tt1 + sqrtf(tt1 * tt1 - tt2)));
}

inline __device__ __host__ float4 erfinvf(float4 v)
{
#if defined(__CUDA_ARCH__)
    return make_float4(::erfinvf(v.x), ::erfinvf(v.y), ::erfinvf(v.z), ::erfinvf(v.w));
#else
    return make_float4(myErfInv(v.x), myErfInv(v.y), myErfInv(v.z), myErfInv(v.w));
#endif
}

inline __device__ __host__ float4 erff(float4 v)
{
#if !defined(__CUDA_ARCH__)
    using namespace std;
#endif
    return make_float4(erff(v.x), erff(v.y), erff(v.z), erff(v.w));
}

#endif
