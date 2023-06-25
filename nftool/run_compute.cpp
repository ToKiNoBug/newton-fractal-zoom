
#include "run_compute.h"
#include <newton_fractal.h>
#include <fstream>
#include <fmt/format.h>
#include <omp.h>
#include <atomic>
#include <newton_archive.h>
#ifdef NEWTON_FRACTAL_MPC_SUPPORT
#include <gmp.h>
#endif

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
#ifdef NEWTON_FRACTAL_MPC_SUPPORT
  mp_get_memory_functions(nullptr, &realloc_func_ptr, nullptr);
  mp_set_memory_functions(my_malloc, my_realloc, nullptr);
#endif
}

tl::expected<void, std::string> run_compute(const compute_task& ct) noexcept {
  nf::newton_archive ar;
  {
    std::ifstream ifs{ct.filename};
    auto md_e = nf::load_metadata(ifs, false);
    if (!md_e.has_value()) {
      return tl::make_unexpected(
          fmt::format("Failed to laod meta data. Detail: {}", md_e.error()));
    }
    ar.info() = std::move(md_e.value());
  }
  ar.setup_matrix();

  nf::newton_equation_base::compute_option option{ar.map_has_result(),
                                                  ar.map_nearest_point_idx(),
                                                  ar.map_complex_difference()};

  omp_set_num_threads(ct.threads);

  if (!ar.info().obj_creator()->is_fixed_precision()) {
    if (!ar.info().obj_creator()->set_precision(*ar.info().window())) {
      return tl::make_unexpected("Failed to update precision for window.");
    }

    if (!ar.info().obj_creator()->set_precision(*ar.info().equation())) {
      return tl::make_unexpected("Failed to update precision for equation.");
    }
  }

  if (ct.track_memory) {
    replace_memory_functions_gmp();
  }

  double wtime = omp_get_wtime();
  ar.info().equation()->compute(*ar.info().window(), ar.info().iteration,
                                option);
  wtime = omp_get_wtime() - wtime;

  fmt::print("Computation finished with {} seconds.\n", wtime);
  if (ct.track_memory) {
    fmt::print("malloc is runned {} times, and realloc runned {} times.\n",
               num_malloc.load(), num_realloc.load());
    const double times =
        double(ar.info().rows) * ar.info().cols * ar.info().iteration;
    fmt::print("Avarange times: malloc {}/iter, realloc {}/iter.\n",
               num_malloc.load() / times, num_realloc.load() / times);
  }

  tl::expected<void, std::string> ret{};

  if (ct.filename.empty() && ct.return_archive == nullptr) {
    fmt::print(
        "No value assigned to -o, the computation result will not be saved.\n");
    return {};
  }

  {
    auto exp = ar.save(ct.archive_filename);
    if (!exp.has_value()) {
      fmt::print("Failed to save computation result. Detail: {}", exp.error());
    }
  }

  if (ct.return_archive != nullptr) {
    *ct.return_archive = std::move(ar);
  }

  return {};
}