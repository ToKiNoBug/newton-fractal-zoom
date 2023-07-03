//
// Created by joseph on 7/2/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_NEWTON_ZOOMER_H
#define NEWTON_FRACTAL_ZOOM_NEWTON_ZOOMER_H
#include <zoom_window.h>
#include "newton_label.h"
#include <newton_archive.h>
#include <newton_render.h>

class newton_zoomer final : public fractal_utils::zoom_window {
 private:
  newton_fractal::meta_data m_template_metadata;

 public:
  explicit newton_zoomer(QWidget *parent = nullptr);
  ~newton_zoomer() final = default;

  nf::render_config render_config;

  [[nodiscard]] inline auto &template_metadata() noexcept {
    return this->m_template_metadata;
  }
  [[nodiscard]] const auto &template_metadata() const noexcept {
    return this->m_template_metadata;
  }
  void set_template_metadata(newton_fractal::meta_data &&src) & noexcept;

 protected:
  [[nodiscard]] std::unique_ptr<fu::wind_base> create_wind()
      const noexcept final;

  void compute(const fu::wind_base &wind,
               std::any &archive) const noexcept final;

  void render(std::any &archive, const fu::wind_base &wind,
              fu::map_view image_u8c3) const noexcept final;

  std::string encode_hex(const fu::wind_base &wind_src,
                         std::string &err) const noexcept final;

  void decode_hex(std::string_view hex,
                  std::unique_ptr<fu::wind_base> &wind_unique_ptr,
                  std::string &err) const noexcept final;

  QString export_frame(QString filename, const fu::wind_base &wind,
                       fu::constant_view image_u8c3,
                       std::any &custom) const noexcept final;

 public slots:

  void received_wheel_move(std::array<int, 2> pos, bool is_scaling_up) final;
};

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_ZOOMER_H
