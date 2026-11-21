#pragma once
#include <cmath>
#define GPU __host__ __device__

// runs f(i) for every i in [0, n) on the GPU, thousands at once
template <class F> __global__ void kernel(int n, F f) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) f(i); }

// vec3 / ivec3 that work on both CPU and GPU (glm breaks in GPU code)
template <class T> struct v3 {
    T x = 0, y = 0, z = 0;
    GPU v3() {}
    GPU v3(T s) : x(s), y(s), z(s) {}
    GPU v3(T a, T b, T c) : x(a), y(b), z(c) {}
    template <class U> GPU explicit v3(v3<U> o) : x(T(o.x)), y(T(o.y)), z(T(o.z)) {}
    GPU T& operator[](int i) { return i == 0 ? x : i == 1 ? y : z; }
    GPU T operator[](int i) const { return i == 0 ? x : i == 1 ? y : z; }
};
template <class T> struct same { using t = T; };
#define OP(o) \
    template <class T> GPU v3<T> operator o(v3<T> a, v3<T> b) { return v3<T>(a.x o b.x, a.y o b.y, a.z o b.z); } \
    template <class T> GPU v3<T> operator o(v3<T> a, typename same<T>::t s) { return v3<T>(a.x o s, a.y o s, a.z o s); } \
    template <class T> GPU v3<T> operator o(typename same<T>::t s, v3<T> a) { return v3<T>(s o a.x, s o a.y, s o a.z); } \
    template <class T> GPU v3<T>& operator o##=(v3<T>& a, v3<T> b) { return a = a o b; } \
    template <class T> GPU v3<T>& operator o##=(v3<T>& a, typename same<T>::t s) { return a = a o s; }
OP(+) OP(-) OP(*) OP(/)
#undef OP
template <class T> GPU v3<T> operator-(v3<T> a) { return v3<T>(-a.x, -a.y, -a.z); }
template <class T> GPU v3<T> min(v3<T> a, v3<T> b) { return v3<T>(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z); }
template <class T> GPU v3<T> max(v3<T> a, v3<T> b) { return v3<T>(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z); }
template <class T> GPU v3<T> clamp(v3<T> a, v3<T> lo, v3<T> hi) { return min(max(a, lo), hi); }
template <class T> GPU T dot(v3<T> a, v3<T> b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
template <class T> GPU v3<T> cross(v3<T> a, v3<T> b) { return v3<T>(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }
using vec3 = v3<float>;
using ivec3 = v3<int>;
GPU inline float length(vec3 a) { return sqrtf(dot(a, a)); }
GPU inline vec3 normalize(vec3 a) { return a / length(a); }
GPU inline float mix(float a, float b, float t) { return a + (b - a) * t; }
GPU inline vec3 mix(vec3 a, vec3 b, float t) { return a + (b - a) * t; }