//
// Created by joseph on 6/25/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_NF_CUDA_MACROS_HPP
#define NEWTON_FRACTAL_ZOOM_NF_CUDA_MACROS_HPP

#ifdef __CUDACC__
#define NF_HOST_DEVICE_FUN __host__ __device__
#else
#define NF_HOST_DEVICE_FUN
#endif

#endif  // NEWTON_FRACTAL_ZOOM_NF_CUDA_MACROS_HPP
