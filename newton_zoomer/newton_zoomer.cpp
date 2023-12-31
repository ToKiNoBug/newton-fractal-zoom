//
// Created by joseph on 7/2/23.
//

#include <fstream>
#include <omp.h>
#include <ranges>
#include <thread>
#include <QMessageBox>
#include <QCoreApplication>
#include "zoomer_custom_widget.h"
#include "newton_zoomer.h"
#include "point_form.h"

newton_zoomer::newton_zoomer(QWidget *parent)
    : fractal_utils::zoom_window{parent} {
  if (this->m_window_stack.empty()) {
    this->m_window_stack.emplace();
  }
  this->set_label_widget(new newton_label{this});
  this->label_widget()->setAutoFillBackground(true);
  this->label_widget()->setScaledContents(true);

  this->set_frame_file_extensions("*.nfar;*.nfar.zst;;*.json");

  this->setWindowTitle(this->m_title);

  auto custom_widgets = new zoomer_custom_widget{this};
  this->set_custom_widget(custom_widgets);
  connect(custom_widgets->pushbutton_edit_points(), &QPushButton::clicked, this,
          &newton_zoomer::when_btn_edit_points_clicked);
  connect(custom_widgets->toolbutton_add_point(), &QToolButton::clicked, this,
          &newton_zoomer::when_btn_add_point_clicked);
  connect(custom_widgets->toolbutton_erase_point(), &QToolButton::clicked, this,
          &newton_zoomer::when_btn_erase_point_clicked);
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

  if (this->auto_precision &&
      !this->template_metadata().obj_creator()->is_fixed_precision()) {
    const int new_prec =
        this->template_metadata().obj_creator()->suggested_precision_of(
            wind, this->rows(), this->cols());
    fmt::print("Current precision: {}\n", new_prec);

    ar.info().set_precision(new_prec);
    //    {
    //      auto temp =
    //      this->template_metadata().clone_with_precision(new_prec); if
    //      (!temp.has_value()) {
    //        QMessageBox::critical(
    //            nullptr, "Failed to compute",
    //            QString{
    //                fmt::format(
    //                    "Cannot update precision of metadata,
    //                    clone_with_precision " "failed with following
    //                    information: \n{}", temp.error()) .data()});
    //        exit(1);
    //      }
    //      ar.info() = std::move(temp.value());
    //    }
  }

  ar.setup_matrix();
  {
    nf::newton_equation_base::compute_option opt{
        .bool_has_result{ar.map_has_result()},
        .u8_nearest_point_idx{ar.map_nearest_point_idx()},
        .f64complex_difference{ar.map_complex_difference()}};
    QString err_title{"Failed to compute, nfzoom must crash"}, err_message;
    bool has_exception{false};
    try {
      ar.info().equation()->compute(*ar.info().window(), ar.info().iteration,
                                    opt);
    } catch (const std::exception &e) {
      has_exception = true;
      err_message = tr("Exception caught during computation, may be a "
                       "gpu-related error. Detail:\n%1")
                        .arg(e.what());
    } catch (...) {
      has_exception = true;
      err_message = tr(
          "Unknown exception caught, we can not retrieve any error message.");
    }
    if (has_exception) {
      QMessageBox::critical(
          nullptr, err_title, err_message,
          QMessageBox::StandardButtons{QMessageBox::StandardButton::Close});
      exit(114514);
    }
  }

  this->m_computation_log.emplace_back(omp_get_wtime());
}

#define handle_error(exp_obj)                             \
  if (!(exp_obj)) {                                       \
    fmt::print("Render failed: {}\n", (exp_obj).error()); \
    exit(2);                                              \
  }

void newton_zoomer::render(std::any &archive, const fu::wind_base &wind,
                           fu::map_view image_u8c3) const noexcept {
  auto &ar = *std::any_cast<nf::newton_archive>(&archive);
  //  thread_local nf::cpu_renderer renderer;
  //  thread_local auto gpu_renderer_exp =
  //      nf::gpu_render::create(image_u8c3.rows(), image_u8c3.cols());
  //  thread_local auto gpu_render_config_exp =
  //      nf::render_config_gpu_interface::create();

  if (!this->gpu_render) {
    ::cpu_renderer.set_threads((int)std::thread::hardware_concurrency());
    ::cpu_renderer.render(this->render_config, ar.map_has_result(),
                          ar.map_nearest_point_idx(),
                          ar.map_complex_difference(), image_u8c3, 0, 0);
    return;
  }

  auto &gpu_config = ::gpu_render_config_exp.value();
  auto &gpu_renderer = ::gpu_renderer_exp.value();
  auto err = gpu_config->set_config(this->render_config);
  handle_error(err);
  err = gpu_renderer->render(*gpu_config, ar.map_has_result(),
                             ar.map_nearest_point_idx(),
                             ar.map_complex_difference(), image_u8c3, 0, 0);
  handle_error(err);
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

  //  while (!this->m_window_stack.empty()) {
  //    this->m_window_stack.pop();
  //  }
  //
  //  this->m_window_stack.emplace(this->m_template_metadata.rows,
  //                               this->m_template_metadata.cols, 0);
  this->reset(this->template_metadata().rows, this->template_metadata().cols);
  //  this->map_base = {(size_t)this->template_metadata().rows,
  //                    (size_t)this->template_metadata().cols, 0};

  //  {
  //    this->current_result().wind.reset(
  //        this->template_metadata().window()->create_another());
  //
  //    const bool copy_success = this->template_metadata().window()->copy_to(
  //        this->current_result().wind.get());
  //    assert(copy_success);
  //  }
  //
  //  this->refresh_range_display();
  //  this->refresh_image_display();
}

QString newton_zoomer::export_frame(QString _filename,
                                    const fu::wind_base &wind,
                                    fu::constant_view image_u8c3,
                                    std::any &custom) const noexcept {
  const std::string filename = _filename.toLocal8Bit().data();
  auto *ar_ptr = std::any_cast<newton_fractal::newton_archive>(&custom);
  if (ar_ptr == nullptr) {
    return tr(
        "The passed std::any reference is not a newton_archive instance.");
  }

  if (_filename.endsWith(".json")) {
    auto res = save_metadata(ar_ptr->info(), nf::float_save_format::hex_string);
    if (!res.has_value()) {
      return QString::fromUtf8(res.error());
    }
    std::ofstream ofs{filename};
    ofs << res.value().dump(2);
    ofs.close();
    return {};
  }

  auto err = ar_ptr->save(filename);
  if (!err.has_value()) {
    return QString::fromUtf8(err.error());
  }

  return {};
}

void newton_zoomer::received_wheel_move(std::array<int, 2> pos,
                                        bool is_scaling_up) {
  if (this->auto_precision &&
      !this->template_metadata().obj_creator()->is_fixed_precision()) {
    auto &cur = this->current_result();
    const int new_prec =
        this->template_metadata().obj_creator()->suggested_precision_of(
            *cur.wind, this->rows(), this->cols()) +
        2;
    auto new_objc = this->template_metadata().obj_creator()->copy();
    new_objc->set_precision(new_prec);
    auto err = new_objc->set_precision(*cur.wind);
    if (!err.has_value()) {
      QMessageBox::critical(
          this, tr("Failed to update precision for window"),
          tr("object_creator::set_precision(fu::wind&) failed with "
             "following information:\n%1")
              .arg(err.error().data()));
      exit(1);
    }
  }
  fu::zoom_window::received_wheel_move(pos, is_scaling_up);

  fmt::print("size of stack: {}\n", this->m_window_stack.size());
}

void newton_zoomer::refresh_range_display() & noexcept {
  {
    //    QImage null_img{QSize{img_width, img_height},
    //                    QImage::Format::Format_ARGB32};
    //    null_img.fill(0x00FFFFFF);
    //    this->label_widget()->setPixmap(QPixmap::fromImage(null_img));
  }
  zoom_window::refresh_range_display();
  auto lb = dynamic_cast<newton_label *>(this->label_widget());
  if (lb->pixmap().isNull()) {
    const int img_height = int(this->rows() * this->scale());
    const int img_width = int(this->cols() * this->scale());
    lb->resize(img_width, img_height);
  }
  lb->reset(*this->template_metadata().equation());
  lb->repaint_points(*this->current_result().wind);
}

void newton_zoomer::refresh_image_display() & noexcept {
  zoom_window::refresh_image_display();

  auto fps = this->fps(5);
  if (fps.has_value()) {
    this->setWindowTitle(
        QStringLiteral("%1 fps: %2").arg(this->m_title).arg(fps.value()));
  } else {
    this->setWindowTitle(this->m_title);
  }
}

void newton_zoomer::on_btn_revert_clicked() {
  zoom_window::on_btn_revert_clicked();
  fmt::print("size of stack: {}\n", this->m_window_stack.size());
}

void newton_zoomer::on_btn_repaint_clicked() {
  zoom_window::on_btn_repaint_clicked();
  fmt::print("size of stack: {}\n", this->m_window_stack.size());
}

void newton_zoomer::update_equation(
    std::span<const std::complex<double>> points) & noexcept {
  this->template_metadata().equation()->reset(points);

  this->compute_current();
  this->render_current();
  this->refresh_image_display();
}

std::optional<double> newton_zoomer::fps(size_t statistic_num) const noexcept {
  //  if (max_counted_time_span < 0) {
  //    return std::nullopt;
  //  }
  const auto current = omp_get_wtime();
  // const auto counting_min = current - max_counted_time_span;
  if (this->m_computation_log.empty()) {
    return std::nullopt;
  }

  const double latest_time = this->m_computation_log.back();
  size_t num_frames{0};
  double oldest_time{-1};
  for (auto time : this->m_computation_log | std::views::reverse) {
    if (num_frames > statistic_num) {
      break;
    }

    oldest_time = time;
    num_frames++;
  }

  if (num_frames <= 1) {
    return std::nullopt;
  }

  if (latest_time == oldest_time) {
    return std::nullopt;
  }

  // fmt::print("num_frames = {}\n", num_frames);

  return double(num_frames - 1) / (latest_time - oldest_time);
}

void newton_zoomer::when_btn_edit_points_clicked() noexcept {
  if (this->m_window_stack.empty()) {
    return;
  }

  //  auto &archive =
  //      std::any_cast<const nf::newton_archive
  //      &>(this->current_result().archive);

  auto eq = this->m_template_metadata.equation();
  std::vector<std::complex<double>> points;
  points.reserve(eq->order());
  for (int idx = 0; idx < eq->order(); idx++) {
    points.emplace_back(eq->point_at(idx));
  }

  auto wind = new point_form{points, this};
  wind->setWindowFlag(Qt::WindowType::Dialog);
  wind->setAttribute(Qt::WidgetAttribute::WA_DeleteOnClose);
  wind->setAttribute(Qt::WidgetAttribute::WA_AlwaysStackOnTop);

  connect(wind, &point_form::points_changed,
          [this](std::span<const std::complex<double>> points) {
            this->update_equation(points);
            this->refresh_range_display();
          });

  wind->show();
}

void newton_zoomer::keyPressEvent(QKeyEvent *event) {
  if (event->key() == Qt::Key::Key_Escape) {
    this->set_cursor_state(cursor_state_t::none);
    return;
  }

  zoom_window::keyPressEvent(event);
}

void newton_zoomer::when_btn_add_point_clicked() noexcept {
  this->set_cursor_state(cursor_state_t::add_point);
}

void newton_zoomer::when_btn_erase_point_clicked() noexcept {
  this->set_cursor_state(cursor_state_t::erase_point);
}

QString newton_zoomer::set_language(fu::language_t lang) & noexcept {
  auto err = zoom_window::set_language(lang);
  if (!err.isEmpty()) {
    return err;
  }

  QString translator_file;
  switch (lang) {
    case fu::language_t::en_US:
      QCoreApplication::removeTranslator(&this->m_translator_newton_zoomer);
      return {};
    case fu::language_t::zh_CN:
      translator_file = ":/i18n/nfzoom_zh_CN.qm";
      break;
  }

  {
    const bool ok = this->m_translator_newton_zoomer.load(translator_file);
    if (!ok) {
      return QStringLiteral("Failed to load %1").arg(translator_file);
    }
  }
  {
    const bool ok =
        QCoreApplication::installTranslator(&this->m_translator_newton_zoomer);
    if (!ok) {
      return QStringLiteral("Failed to install %1").arg(translator_file);
    }
  }
  dynamic_cast<zoomer_custom_widget *>(this->custom_widget())->retranslate_ui();
  return {};
}