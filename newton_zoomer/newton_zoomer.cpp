//
// Created by joseph on 7/2/23.
//

#include "newton_zoomer.h"

newton_zoomer::newton_zoomer(QWidget *parent)
    : fractal_utils::zoom_window{parent} {
  this->set_label_widget(new newton_label{this});

  if (this->m_window_stack.empty()) {
    this->m_window_stack.emplace();
  }
}

[[nodiscard]] std::unique_ptr<fu::wind_base> newton_zoomer::create_wind()
    const noexcept {
  std::unique_ptr<fu::wind_base> ret{
      this->template_metadata().window()->create_another()};
  this->template_metadata().window()->copy_to(ret.get());
  return ret;
}

void newton_zoomer::compute(const fu::wind_base &wind,
                            std::any &archive) const noexcept {
  {
    auto *ar_ptr = std::any_cast<nf::newton_archive>(&archive);
    if (ar_ptr == nullptr) {
      archive = nf::newton_archive{};
    }
  }

  auto &ar = *std::any_cast<nf::newton_archive>(&archive);

  ar.info() = this->template_metadata();

  {
    const bool copy_success = wind.copy_to(ar.info().window());
    assert(copy_success);
  }
  ar.setup_matrix();
  {
    nf::newton_equation_base::compute_option opt{
        .bool_has_result{ar.map_has_result()},
        .u8_nearest_point_idx{ar.map_nearest_point_idx()},
        .f64complex_difference{ar.map_complex_difference()}};
    ar.info().equation()->compute(*ar.info().window(), ar.info().iteration,
                                  opt);
  }
}

void newton_zoomer::render(std::any &archive, const fu::wind_base &wind,
                           fu::map_view image_u8c3) const noexcept {
  auto &ar = *std::any_cast<nf::newton_archive>(&archive);
  thread_local nf::cpu_renderer renderer;

  renderer.render(this->render_config, ar.map_has_result(),
                  ar.map_nearest_point_idx(), ar.map_complex_difference(),
                  image_u8c3, 0, 0);
}

std::string newton_zoomer::encode_hex(const fu::wind_base &wind_src,
                                      std::string &err) const noexcept {
  err.clear();
  auto exp =
      this->template_metadata().obj_creator()->encode_centerhex(wind_src);
  if (!exp.has_value()) {
    err = std::move(exp.error());
    return {};
  }
  return exp.value();
}

void newton_zoomer::decode_hex(std::string_view hex,
                               std::unique_ptr<fu::wind_base> &wind_unique_ptr,
                               std::string &err) const noexcept {
  err.clear();

  if (wind_unique_ptr == nullptr) {
    wind_unique_ptr.reset(this->template_metadata().window()->create_another());
    const bool copy_success =
        this->template_metadata().window()->copy_to(wind_unique_ptr.get());
    assert(copy_success);
  }

  auto exp = this->template_metadata().obj_creator()->decode_centerhex(
      hex, *wind_unique_ptr);
  if (!exp.has_value()) {
    err = std::move(exp.error());
  }
}

void newton_zoomer::set_template_metadata(
    newton_fractal::meta_data &&src) & noexcept {
  this->m_template_metadata = std::move(src);

  while (!this->m_window_stack.empty()) {
    this->m_window_stack.pop();
  }

  this->m_window_stack.emplace(this->m_template_metadata.rows,
                               this->m_template_metadata.cols, 0);
  this->map_base = {(size_t)this->template_metadata().rows,
                    (size_t)this->template_metadata().cols, 0};

  {
    this->current_result().wind.reset(
        this->template_metadata().window()->create_another());

    const bool copy_success = this->template_metadata().window()->copy_to(
        this->current_result().wind.get());
    assert(copy_success);
  }

  this->refresh_range_display();
  this->refresh_image_display();
}