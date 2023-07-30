#include "newton_label.h"
#include <QPainter>
#include <memory>
#include <QtWidgets>
#include <fmt/format.h>
#include "newton_zoomer.h"

draggable_label::draggable_label(QWidget *parent) : QLabel(parent) {}

void draw_point(QPaintDevice &device, const QPoint &offset,
                std::optional<int> number, const QFontMetrics &font_metrics,
                const draw_option &option) noexcept {
  QPainter painter{&device};
  {
    QBrush brush{QColor{option.background_color}};
    painter.setBrush(brush);
    QPen pen{QColor{option.text_color}};
    painter.setPen(pen);
  }
  {
    const QSize device_size{device.width(), device.height()};
    painter.fillRect(QRect{QPoint{0, 0}, device_size}, QColor{0, 0, 0, 0});
    painter.drawEllipse(
        QRect{offset, QSize{option.icon_size, option.icon_size}});
  }
  if (number.has_value()) {
    const auto text = QString::number(number.value());
    const auto text_rect = font_metrics.tightBoundingRect(text);
    const float h_spacing =
        std::max(0.0f, float(device.height() - text_rect.height()) / 2) +
        float(offset.y());
    const float w_spacing =
        std::max(0.0f, float(device.width() - text_rect.width()) / 2) +
        float(offset.x());

    painter.drawText(QPointF{w_spacing, h_spacing}, text);
  }
  painter.end();
}

void draw_cross(QPaintDevice &device, const QPoint &offset,
                const draw_option &option) noexcept {
  QPainter painter{&device};
  QPen pen_red{QColor{Qt::GlobalColor::red}};
  QPen pen_white{QColor{Qt::GlobalColor::white}};
  pen_white.setWidth(3);
  pen_red.setWidth(2);

  const QSize device_size{device.width(), device.height()};
  painter.fillRect(QRect{QPoint{0, 0}, device_size}, QColor{0, 0, 0, 0});

  auto fun_draw = [offset, option, &painter]() {
    painter.drawLine(offset,
                     offset + QPoint{option.icon_size, option.icon_size});
    painter.drawLine(offset + QPoint{0, +option.icon_size},
                     offset + QPoint{+option.icon_size, 0});
  };
  painter.setPen(pen_white);
  fun_draw();
  painter.setPen(pen_red);
  fun_draw();

  painter.end();
}

void draggable_label::draw(int index, const draw_option &option) & noexcept {
  this->resize(option.icon_size, option.icon_size);

  draw_point(*this, {0, 0}, index, this->fontMetrics(), option);
  if (this->m_draw_cross) {
    draw_cross(*this, {0, 0}, option);
    // this->m_draw_cross = false;
  }
}

void draggable_label::draw() & noexcept {
  this->draw(this->index().value_or(-1), this->option());
}

newton_label *draggable_label::impl_parent_label() const noexcept {
  return dynamic_cast<newton_label *>(this->parent());
}

std::optional<int> draggable_label::index() const noexcept {
  auto idx = this->parent_label()->extract_index(this);
  if (idx.has_value()) {
    return idx.value();
  }
  return std::nullopt;
}

[[nodiscard]] draw_option draggable_label::option() const noexcept {
  return this->parent_label()->option;
}

void draggable_label::paintEvent(QPaintEvent *e) {
  QLabel::paintEvent(e);
  this->draw();
}

void draggable_label::mousePressEvent(QMouseEvent *e) {
  if (this->m_draw_cross && this->parent_label()->zoomer()->cursor_state() ==
                                cursor_state_t::erase_point) {
    emit this->erased(this);
    return;
  }

  if (e->buttons() & Qt::MouseButton::LeftButton) {
    this->point_before_drag = e->pos();
  }
}

const QString mime_data_tag_NF_private{"self_this_pointer"};

void draggable_label::enterEvent(QEnterEvent *event) {
  this->parent_label()->clear_drawable_point();

  if (this->parent_label()->zoomer()->cursor_state() ==
      cursor_state_t::erase_point) {
    this->m_draw_cross = true;
    this->repaint();
  }
}

void draggable_label::leaveEvent(QEvent *event) {
  this->m_draw_cross = false;
  this->repaint();
}

void draggable_label::mouseMoveEvent(QMouseEvent *event) {
  if (!event->buttons() & Qt::MouseButton::LeftButton) {
    return;
  }
  if (!this->point_before_drag.has_value()) {
    return;
  }
  //  const int start_drag_distance = 1;
  //  if ((event->pos() - this->point_before_drag.value()).manhattanLength() <
  //      start_drag_distance) {
  //    return;
  //  }
  // fmt::print("startDragDistance = {}\n", QApplication::startDragDistance());

  auto *drag = new QDrag{this};
  auto *mime_data = new QMimeData;
  {
    auto self = this;
    mime_data->setData(mime_data_tag_NF_private,
                       QByteArray{(const char *)&self, sizeof(void *)});
  }
  drag->setMimeData(mime_data);
  drag->exec(Qt::MoveAction);

  //  fmt::print("draggable_label::mouseMoveEvent: event->pos = [{}, {}]\n",
  //             event->pos().x(), event->pos().y());
  //  auto current_pos = this->pos() + event->pos();
  //  fmt::print("draggable_label::mouseMoveEvent: current pos = [{}, {}]\n",
  //             current_pos.x(), current_pos.y());
  //
  //  this->setGeometry(current_pos.x(), current_pos.y(),
  //  this->option().icon_size,
  //                    this->option().icon_size);
}

newton_label::newton_label(newton_zoomer *parent)
    : scalable_label(parent), m_zoomer{parent} {
  this->setAcceptDrops(true);
}

// void draggable_label::dragMoveEvent(QDragMoveEvent* event) {
//   auto event_pos = event->position().toPoint();
//   fmt::print("draggable_label::dragMoveEvent: event_pos = [{}, {}]\n",
//              event_pos.x(), event_pos.y());
//   auto current_pos = this->pos() + event_pos;
//   fmt::print("draggable_label::dragMoveEvent: current pos = [{}, {}]\n",
//              current_pos.x(), current_pos.y());
// }

newton_zoomer *newton_label::zoomer() const noexcept { return this->m_zoomer; }

void newton_label::repaint_point(
    const fractal_utils::wind_base &wind, draggable_label *label,
    std::complex<double> coordinate) const noexcept {
  std::array<double, 2> coord{coordinate.real(), coordinate.imag()};
  auto center = wind.displayed_center();

  const double x_offset =
      (coord[0] - center[0]) / wind.displayed_x_span();  // [-0.5,0.5]
  const double y_offset =
      (coord[1] - center[1]) / wind.displayed_y_span();  //[-0.5,0.5]

  const int _width = this->width();
  const int _height = this->height();

  int w_pos = int(x_offset * _width + double(_width) / 2);
  int h_pos = int(-y_offset * _height + double(_height) / 2);

  if (w_pos < 0 || w_pos >= this->width() || h_pos < 0 ||
      h_pos >= this->height()) {
    label->hide();
    return;
  }

  const auto icon_size = this->option.icon_size;
  w_pos -= icon_size / 2;
  h_pos -= icon_size / 2;

  label->setGeometry(w_pos, h_pos, icon_size, icon_size);
  label->show();
}

void newton_label::repaint_point(const fractal_utils::wind_base &wind,
                                 size_t idx) & noexcept {
  if (wind.displayed_x_span() == 0 || wind.displayed_y_span() == 0) {
    return;
  }
  auto &pair = this->m_points[idx];
  this->repaint_point(wind, pair.label.get(), pair.coordinate);
}

newton_label::label_point_pair newton_label::make_draggable_label(
    std::complex<double> coord) & noexcept {
  label_point_pair ret;
  ret.coordinate = coord;
  ret.label = std::make_unique<draggable_label>(this);
  connect(ret.label.get(), &draggable_label::erased, this,
          &newton_label::when_point_erased);
  return ret;
}

void newton_label::mousePressEvent(QMouseEvent *e) {
  const auto current_cursor_state = this->m_zoomer->cursor_state();

  switch (current_cursor_state) {
    case cursor_state_t::add_point: {
      {
        const auto capacity = this->zoomer()->render_config_capacity();
        if (this->m_points.size() + 1 > capacity) {
          QMessageBox::warning(
              this, "Can not add more points",
              QStringLiteral(
                  "The assigned render config can hold only %1 points")
                  .arg(capacity));
          return;
        }
      }

      const auto pos = e->pos();
      auto coord =
          this->m_zoomer->template_metadata().window()->displayed_coordinate(
              {this->height(), this->width()}, {pos.y(), pos.x()});

      this->m_points.emplace_back(
          this->make_draggable_label({coord[0], coord[1]}));
      auto current_points = this->current_points();
      this->m_zoomer->update_equation(current_points);
      this->repaint_points(*this->m_zoomer->template_metadata().window());
      break;
    }
    case cursor_state_t::erase_point:
      return;
    case cursor_state_t::none:
      scalable_label::mousePressEvent(e);
      return;
  }
}

void newton_label::mouseMoveEvent(QMouseEvent *e) {
  switch (this->m_zoomer->cursor_state()) {
    case cursor_state_t::add_point:
    case cursor_state_t::erase_point:
      this->m_drawable_point = e->pos();
      this->repaint();
      break;
    default:
      if (this->m_drawable_point.has_value()) {
        this->m_drawable_point.reset();
        this->repaint();
      }
      break;
  }
  scalable_label::mouseMoveEvent(e);
}

void newton_label::reset(const nf::newton_equation_base &equation) & noexcept {
  this->m_points.clear();
  this->m_points.reserve(equation.order());

  for (int o = 0; o < equation.order(); o++) {
    this->m_points.emplace_back(
        this->make_draggable_label(equation.point_at(o)));
  }
}

void newton_label::repaint_points(
    const fractal_utils::wind_base &wind) & noexcept {
  if (wind.displayed_x_span() == 0 || wind.displayed_y_span() == 0) {
    return;
  }

  for (size_t idx = 0; idx < this->m_points.size(); idx++) {
    this->repaint_point(wind, idx);
  }
}

draggable_label *newton_label::extract_draggable_label(
    const QMimeData *src) const noexcept {
  auto idx = this->extract_index(src);
  if (idx.has_value()) {
    return this->m_points[idx.value()].label.get();
  }
  return nullptr;
}

[[nodiscard]] std::optional<size_t> newton_label::extract_index(
    const QMimeData *src) const noexcept {
  if (src == nullptr) {
    return std::nullopt;
  }
  if (!src->hasFormat(mime_data_tag_NF_private)) {
    return std::nullopt;
  }

  auto data = src->data(mime_data_tag_NF_private);
  draggable_label *lb{nullptr};
  if (data.size() != sizeof(size_t)) {
    return std::nullopt;
  }

  memcpy(&lb, data.data(), sizeof(size_t));
  return this->extract_index(lb);
}

std::optional<size_t> newton_label::extract_index(
    const draggable_label *lb) const noexcept {
  for (size_t idx = 0; idx < this->m_points.size(); idx++) {
    auto &pair = this->m_points[idx];
    if (lb == pair.label.get()) {
      return idx;
    }
  }
  return std::nullopt;
}

void newton_label::dragEnterEvent(QDragEnterEvent *event) {
  if (this->extract_draggable_label(event->mimeData()) != nullptr) {
    event->acceptProposedAction();
  } else {
    scalable_label::dragEnterEvent(event);
  }
}

void newton_label::dropEvent(QDropEvent *event) {
  auto index_opt = this->extract_index(event->mimeData());
  if (!index_opt.has_value()) {
    return;
  }
  auto &pair = this->m_points[index_opt.value()];
  auto pos = event->position();
  auto *current_wind = this->zoomer()->current_result().wind.get();
  const auto new_coord = current_wind->displayed_coordinate(
      {this->height(), this->width()}, {(int)pos.y(), (int)pos.x()});
  pair.coordinate = {new_coord[0], new_coord[1]};

  this->repaint_points(*current_wind);

  pair.label->point_before_drag.reset();

  event->acceptProposedAction();
  this->zoomer()->clear_computation_log();
}

std::vector<std::complex<double>> newton_label::current_points()
    const noexcept {
  std::vector<std::complex<double>> ret;
  ret.reserve(this->m_points.size());
  for (auto &p : this->m_points) {
    ret.emplace_back(p.coordinate);
  }
  return ret;
}

void newton_label::dragMoveEvent(QDragMoveEvent *event) {
  const auto idx_opt = this->extract_index(event->mimeData());
  if (!idx_opt.has_value()) {
    return;
  }
  auto event_pos = event->position().toPoint();
  const auto idx = idx_opt.value();

  const auto &wind = *this->zoomer()->current_result().wind;

  auto current_coord = wind.displayed_coordinate(
      {this->height(), this->width()}, {event_pos.y(), event_pos.x()});
  std::complex<double> current_cplx{current_coord[0], current_coord[1]};

  this->repaint_point(wind, this->m_points[idx].label.get(), current_cplx);

  auto points = this->current_points();
  points[idx] = current_cplx;

  this->zoomer()->update_equation(points);

  //    fmt::print("newton_label::dragMoveEvent: event_pos = [{}, {}]\n",
  //               event_pos.x(), event_pos.y());
  //    auto current_pos = this->pos() + event_pos;
  //    fmt::print("newton_label::dragMoveEvent: current pos = [{}, {}]\n",
  //               current_pos.x(), current_pos.y());
}

void newton_label::paintEvent(QPaintEvent *e) {
  scalable_label::paintEvent(e);

  if (this->m_drawable_point.has_value()) {
    auto offset = this->m_drawable_point.value();
    switch (this->m_zoomer->cursor_state()) {
      case cursor_state_t::add_point:
        draw_point(*this, offset, std::nullopt, this->fontMetrics(),
                   this->option);
        break;
      case cursor_state_t::erase_point:
        draw_cross(*this, offset, this->option);
        break;
      default:
        break;
    }
  }
}

void newton_label::leaveEvent(QEvent *e) {
  this->m_drawable_point.reset();
  this->repaint();
}

void newton_label::when_point_erased(draggable_label *lb) {
  const auto idx_opt = this->extract_index(lb);
  if (!idx_opt.has_value()) {
    return;
  }

  if (this->m_points.size() <= 2) {
    QMessageBox::warning(
        this, "Can not erase this point",
        QStringLiteral(
            "There are only %1 points, but expected at least 2 points.")
            .arg(this->m_points.size()));
    return;
  }

  const auto idx = idx_opt.value();

  this->m_points.erase(this->m_points.begin() + int64_t(idx));

  const auto current_points = this->current_points();
  this->zoomer()->update_equation(current_points);
  this->repaint_points(*this->zoomer()->template_metadata().window());
}