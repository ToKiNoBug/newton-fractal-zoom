#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// typedef float real_t;
// typedef float _Complex complex_f32;
// typedef double real_f64;
// typedef  complex_f64;

typedef struct {
  int nearest_point_idx;
  double _Complex difference;
} single_result;

#define NF_OPENCL_MAKE_COMPUTE_FUNCTIONS_IMPL(real_t, complex_t, real_vec2_t,  \
                                              suffix)                          \
  complex_t item_at_order##suffix(__global const complex_t* parameters,        \
                                  const int order, int _order) {               \
    return parameters[order - _order - 1];                                     \
  }                                                                            \
                                                                               \
  complex_t iterate##suffix(__global const complex_t* parameters,              \
                            const int order, complex_t z) {                    \
    complex_t f = item_at_order##suffix(parameters, order, 0);                 \
    complex_t df = item_at_order##suffix(parameters, order, 1);                \
                                                                               \
    const int this_order = order;                                              \
    complex_t z_power = z;                                                     \
    complex_t temp;                                                            \
                                                                               \
    for (int n = 1; n < this_order; n++) {                                     \
      {                                                                        \
        temp = item_at_order##suffix(parameters, order, n) * z_power;          \
        f += temp;                                                             \
      }                                                                        \
      {                                                                        \
        if (n + 1 >= this_order) {                                             \
          temp = z_power;                                                      \
          temp *= (n + 1);                                                     \
        } else {                                                               \
          temp = item_at_order##suffix(parameters, order, n + 1) * z_power;    \
          temp *= (n + 1);                                                     \
        }                                                                      \
        df += temp;                                                            \
      }                                                                        \
      z_power *= z;                                                            \
    }                                                                          \
                                                                               \
    f += z_power;                                                              \
                                                                               \
    return z - (f / df);                                                       \
  }                                                                            \
                                                                               \
  void iterate_n##suffix(__global const complex_t* parameters, int order,      \
                         complex_t* z, int n) {                                \
    for (int i = 0; i < n; i++) {                                              \
      *z = iterate##suffix(parameters, order, *z);                             \
    }                                                                          \
  }                                                                            \
                                                                               \
  bool is_normal_real##suffix(real_t r) { return r == r; }                     \
                                                                               \
  bool is_normal_complex##suffix(complex_t z) {                                \
    real_vec2_t data = vload2(0, (const real_t*)&z);                           \
    return is_normal_real##suffix(data[0]) && is_normal_real##suffix(data[1]); \
  }                                                                            \
                                                                               \
  real_t compute_norm2##suffix(complex_t _z) {                                 \
    real_vec2_t z = vload2(0, (const real_t*)&_z);                             \
    return z[0] * z[0] + z[1] * z[1];                                          \
  }                                                                            \
                                                                               \
  bool compute_single##suffix(                                                 \
      __global const complex_t* parameters, __global const complex_t* points,  \
      const int order, complex_t* z, int iteration_times,                      \
      single_result* result_dest) {                                            \
    iterate_n##suffix(parameters, order, z, iteration_times);                  \
    if (!is_normal_complex##suffix(*z)) {                                      \
      return false;                                                            \
    }                                                                          \
                                                                               \
    int min_idx = -1;                                                          \
    complex_t min_diff;                                                        \
    real_t min_norm2 = INFINITY;                                               \
    for (int idx = 0; idx < order; idx++) {                                    \
      complex_t diff = *z - points[idx];                                       \
      real_t diff_norm2 = compute_norm2##suffix(diff);                         \
                                                                               \
      if (diff_norm2 < min_norm2) {                                            \
        min_idx = idx;                                                         \
        min_diff = diff;                                                       \
        min_norm2 = diff_norm2;                                                \
      }                                                                        \
    }                                                                          \
                                                                               \
    result_dest->nearest_point_idx = min_idx;                                  \
    result_dest->difference = min_diff;                                        \
                                                                               \
    return true;                                                               \
  }                                                                            \
                                                                               \
  kernel void run_computation##suffix(                                         \
      __global const complex_t* points, __global const complex_t* parameters,  \
      int order, int rows, int cols, complex_t r0c0, real_t r_unit,            \
      real_t c_unit, __global bool* dst_has_value,                             \
      __global uint8* dst_nearest_index,                                       \
      __global double _Complex* dst_complex_diff, int iteration_times) {       \
    const size_t global_offset = get_global_id(0);                             \
                                                                               \
    if (global_offset >= rows * cols) {                                        \
      return;                                                                  \
    }                                                                          \
                                                                               \
    const size_t r = global_offset / cols;                                     \
    const size_t c = global_offset % cols;                                     \
                                                                               \
    complex_t z = r0c0;                                                        \
    {                                                                          \
      real_vec2_t* _z = (real_vec2_t*)&z;                                      \
      _z[1] += r * r_unit;                                                     \
      _z[0] += c * c_unit;                                                     \
    }                                                                          \
    single_result result;                                                      \
    const bool ok = compute_single##suffix(parameters, points, order, &z,      \
                                           iteration_times, &result);          \
                                                                               \
    if (!ok) {                                                                 \
      result.nearest_point_idx = 255;                                          \
      result.difference = NAN;                                                 \
    }                                                                          \
    dst_has_value[global_offset] = ok;                                         \
    dst_nearest_index[global_offset] = result.nearest_point_idx;               \
    dst_complex_diff[global_offset] = result.difference;                       \
  }

#define NF_OPENCL_MAKE_COMPUTE_FUNCTIONS(type) \
  NF_OPENCL_MAKE_COMPUTE_FUNCTIONS_IMPL(type, type _Complex, type##2, _##type)

#ifndef NF_OPENCL_DISABLE_FP32
NF_OPENCL_MAKE_COMPUTE_FUNCTIONS(float);
#endif

#ifndef NF_OPENCL_DISABLE_FP64
NF_OPENCL_MAKE_COMPUTE_FUNCTIONS(double);
#endif