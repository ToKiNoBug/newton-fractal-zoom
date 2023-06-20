
#include "run_compute.h"
#include <newton_fractal.h>
#include <fstream>
#include <fmt/format.h>
#include <omp.h>
#include <atomic>

#include <gmp.h>

std::atomic<size_t> num_malloc{0};
std::atomic<size_t> num_realloc{0};

void* (*realloc_func_ptr)(void*, size_t, size_t) = nullptr;

void* my_malloc(size_t sz) noexcept {
  num_malloc++;
  return malloc(sz);
}

void* my_realloc(void* ptr, size_t a, size_t b) {
  num_realloc++;
  return realloc_func_ptr(ptr, a, b);
}

void replace_memory_functions_gmp() noexcept {
  mp_get_memory_functions(nullptr, &realloc_func_ptr, nullptr);
  mp_set_memory_functions(my_malloc, my_realloc, nullptr);
}

tl::expected<void, std::string> run_compute(const compute_task& ct) noexcept {
  nf::meta_data metadata;
  {
    std::ifstream ifs{ct.filename};
    auto md_e = nf::load_metadata(ifs);
    if (!md_e.has_value()) {
      return tl::make_unexpected(
          fmt::format("Failed to laod meta data. Detail: {}", md_e.error()));
    }
    metadata = std::move(md_e.value());
  }

  fu::unique_map cplx{static_cast<size_t>(metadata.rows),
                      static_cast<size_t>(metadata.cols),
                      sizeof(std::complex<double>)};

  fu::unique_map idx{static_cast<size_t>(metadata.rows),
                     static_cast<size_t>(metadata.cols), sizeof(uint8_t)};

  fu::unique_map has_result{static_cast<size_t>(metadata.rows),
                            static_cast<size_t>(metadata.cols), sizeof(bool)};

  nf::newton_equation_base::compute_option option{has_result, idx, cplx};

  omp_set_num_threads(ct.threads);

  if (!metadata.obj_creator->set_precision(*metadata.window)) {
    return tl::make_unexpected("Failed to update precision for window.");
  }
  
  if (!metadata.obj_creator->set_precision(*metadata.equation)) {
    return tl::make_unexpected("Failed to update precision for equation.");
  }

  replace_memory_functions_gmp();
  double wtime = omp_get_wtime();
  metadata.equation->compute(*metadata.window, metadata.iteration, option);
  wtime = omp_get_wtime() - wtime;

  fmt::print("Computation finished with {} seconds.\n", wtime);

  fmt::print("malloc is runned {} times, and realloc runned {} times.\n",
             num_malloc.load(), num_realloc.load());
  const double times =
      double(metadata.rows) * metadata.cols * metadata.iteration;
  fmt::print("Avarange times: malloc {}/iter, realloc {}/iter.\n",
             num_malloc.load() / times, num_realloc.load() / times);

  return {};
}