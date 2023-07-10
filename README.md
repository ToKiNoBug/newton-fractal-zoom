# Newton Fractal Zoom

A collection of tools to make a zooming video of newton fractal.

## Supported floating point types

This project supports computing newton fractal with multi-precision. Supported types are listed as below:

1. C/C++ standard types: `float` and `double`
2. gcc extension: `__float128`
    - Only available on Linux
3. Boost::multiprecision: instantiations of `cpp_bin_float`. Now floats with 16, 32 and 64 bytes are supported (meets
   IEEE-754 standards).
4. GNU mpfr types with dynamic precision: **almost infinite precision!**
    - Only available on Linux

## Build

Supported platforms:

1. Linux
    - gcc (12+)

      It's suggested to build with gcc12. If you don't have a gcc13 installed into your system, clang15+ will also work.
2. Windows
    - clang-cl (16+, lower versions are not tested)
    - msvc (May works, not tested)