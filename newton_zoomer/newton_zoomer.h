//
// Created by joseph on 7/2/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_NEWTON_ZOOMER_H
#define NEWTON_FRACTAL_ZOOM_NEWTON_ZOOMER_H
#include <zoom_window.h>
#include "newton_label.h"
#include <newton_archive.h>
#include <newton_render.h>
#include <list>
#include <QTranslator>

#include <QKeyEvent>

extern nf::cpu_renderer cpu_renderer;
extern tl::expected<std::unique_ptr<nf::gpu_render>, std::string>
    gpu_renderer_exp;
extern tl::expected<std::unique_ptr<nf::render_config_gpu_interface>,
                    std::string>
    gpu_render_config_exp;

enum class cursor_state_t : uint8_t { none, add_point, erase_point };
class newton_zoomer final : public fractal_utils::zoom_window {
  Q_OBJECT
 private:
  newton_fractal::meta_data m_template_metadata;
  mutable std::list<double> m_computation_log;

  QString m_title{"Newton fractal zoomer"};

  cursor_state_t m_cursor_state{cursor_state_t::none};

  QTranslator m_translator_newton_zoomer;

 public:
  explicit newton_zoomer(QWidget *parent = nullptr);
  ~newton_zoomer() final = default;

  nf::render_config render_config;
  bool auto_precision{false};
  bool gpu_render{false};

  QString set_language(fu::language_t lang) & noexcept final;

  [[nodiscard]] inline auto render_config_capacity() const noexcept {
    return this->render_config.methods.size();
  }

  [[nodiscard]] inline auto cursor_state() const noexcept {
    return this->m_cursor_state;
  }

  inline void set_cursor_state(cursor_state_t cs) & noexcept {
    this->m_cursor_state = cs;
  }

  [[nodiscard]] inline auto &template_metadata() noexcept {
    return this->m_template_metadata;
  }
  [[nodiscard]] const auto &template_metadata() const noexcept {
    return this->m_template_metadata;
  }
  void set_template_metadata(newton_fractal::meta_data &&src) & noexcept;

  void refresh_range_display() & noexcept final;

  void refresh_image_display() & noexcept final;

  [[nodiscard]] std::optional<double> fps(size_t statistic_num) const noexcept;

  void clear_computation_log() & noexcept { this->m_computation_log.clear(); }

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

  void on_btn_revert_clicked() final;
  void on_btn_repaint_clicked() final;

  void when_btn_edit_points_clicked() noexcept;

  void when_btn_add_point_clicked() noexcept;
  void when_btn_erase_point_clicked() noexcept;

 public:
  void update_equation(std::span<const std::complex<double>> points) & noexcept;

 protected:
  void keyPressEvent(QKeyEvent *event) final;
};

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_ZOOMER_H
