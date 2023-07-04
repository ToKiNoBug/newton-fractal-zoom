#include "newton_label.h"
#include <QPainter>
#include <memory>
#include "newton_zoomer.h"
#include <fmt/format.h>

draggable_label::draggable_label(QWidget* parent) : QLabel(parent) {}

void draggable_label::draw(int index, const draw_option& option) & noexcept {
  this->resize(option.icon_size, option.icon_size);

  QPainter painter{this};
  {
    QBrush brush{QColor{option.background_color}};
    painter.setBrush(brush);
    QPen pen{QColor{option.text_color}};
    painter.setPen(pen);
  }
  painter.fillRect(QRect{QPoint{0, 0}, this->size()}, QColor{0, 0, 0, 0});
  painter.drawEllipse(QRect{QPoint{0, 0}, this->size()});
  const auto text = QString::number(index);
  const auto text_rect = this->fontMetrics().tightBoundingRect(text);
  const float h_spacing =
      std::max(0.0f, float(this->height() - text_rect.height()) / 2);
  const float w_spacing =
      std::max(0.0f, float(this->width() - text_rect.width()) / 2);

  painter.drawText(QPointF{w_spacing, h_spacing}, text);
  painter.end();
}

void draggable_label::draw() & noexcept {
  this->draw(this->index().value_or(-1), this->option());
}

newton_label* draggable_label::impl_parent_label() const noexcept {
  return dynamic_cast<newton_label*>(this->parent());
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

void draggable_label::paintEvent(QPaintEvent* e) {
  this->draw();
  QLabel::paintEvent(e);
}

void draggable_label::mousePressEvent(QMouseEvent* e) {
  if (e->buttons() & Qt::MouseButton::LeftButton) {
    this->point_before_drag = e->pos();
  }
}

const QString mime_data_tag_NF_private{"self_this_pointer"};

void draggable_label::mouseMoveEvent(QMouseEvent* event) {
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

  QDrag* drag = new QDrag{this};
  QMimeData* mimedata = new QMimeData;
  {
    auto self = this;
    mimedata->setData(mime_data_tag_NF_private,
                      QByteArray{(const char*)&self, sizeof(self)});
  }
  drag->setMimeData(mimedata);
  Qt::DropAction da = drag->exec(Qt::MoveAction);

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

newton_label::newton_label(newton_zoomer* parent)
    : scalable_label(parent), m_zoomer{parent} {
  this->setAcceptDrops(true);
};

// void draggable_label::dragMoveEvent(QDragMoveEvent* event) {
//   auto event_pos = event->position().toPoint();
//   fmt::print("draggable_label::dragMoveEvent: event_pos = [{}, {}]\n",
//              event_pos.x(), event_pos.y());
//   auto current_pos = this->pos() + event_pos;
//   fmt::print("draggable_label::dragMoveEvent: current pos = [{}, {}]\n",
//              current_pos.x(), current_pos.y());
// }

newton_zoomer* newton_label::zoomer() const noexcept { return this->m_zoomer; }

void newton_label::repaint_point(
    const fractal_utils::wind_base& wind, draggable_label* label,
    std::complex<double> coordinate) const noexcept {
  std::array<double, 2> coord{coordinate.real(), coordinate.imag()};
  auto center = wind.displayed_center();

  const double x_offset =
      (coord[0] - center[0]) / wind.displayed_x_span();  // [-0.5,0.5]
  const double y_offset =
      (coord[1] - center[1]) / wind.displayed_y_span();  //[-0.5,0.5]

  int w_pos = int(x_offset * this->width() + this->width() / 2);
  int h_pos = int(-y_offset * this->height() + this->height() / 2);

  const int _width = this->width();
  const int _height = this->height();

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

void newton_label::repaint_point(const fractal_utils::wind_base& wind,
                                 size_t idx) & noexcept {
  if (wind.displayed_x_span() == 0 || wind.displayed_y_span() == 0) {
    return;
  }
  auto& pair = this->m_points[idx];
  this->repaint_point(wind, pair.label.get(), pair.coordinate);
}

void newton_label::mousePressEvent(QMouseEvent* e) {
  //  if (e->buttons() & Qt::MouseButton::LeftButton) {
  //    for (size_t idx = 0; idx < this->m_points.size(); idx++) {
  //      auto& point = this->m_points[idx];
  //      if (point.label->geometry().contains(e->pos())) {
  //        QMimeData* md = new QMimeData;
  //        {
  //          dragged_option opt{this, idx};
  //          md->setData("info", QByteArray{(const char*)&opt, sizeof(opt)});
  //        }
  //        QDrag* drag = new QDrag{this};
  //
  //        drag->setMimeData(md);
  //
  //        Qt::DropAction da = drag->exec();
  //      }
  //    }
  //  }

  scalable_label::mousePressEvent(e);
}

void newton_label::reset(const nf::newton_equation_base& equation) & noexcept {
  this->m_points.clear();
  this->m_points.reserve(equation.order());

  for (int o = 0; o < equation.order(); o++) {
    label_point_pair temp;
    temp.coordinate = equation.point_at(o);
    temp.label = std::make_unique<draggable_label>(this);
    this->m_points.emplace_back(std::move(temp));
  }
}

void newton_label::repaint_points(
    const fractal_utils::wind_base& wind) & noexcept {
  if (wind.displayed_x_span() == 0 || wind.displayed_y_span() == 0) {
    return;
  }

  for (size_t idx = 0; idx < this->m_points.size(); idx++) {
    this->repaint_point(wind, idx);
  }
}

draggable_label* newton_label::extract_draggable_label(
    const QMimeData* src) const noexcept {
  auto idx = this->extract_index(src);
  if (idx.has_value()) {
    return this->m_points[idx.value()].label.get();
  }
  return nullptr;
}

[[nodiscard]] std::optional<size_t> newton_label::extract_index(
    const QMimeData* src) const noexcept {
  if (src == nullptr) {
    return std::nullopt;
  }
  if (!src->hasFormat(mime_data_tag_NF_private)) {
    return std::nullopt;
  }

  auto data = src->data(mime_data_tag_NF_private);
  draggable_label* lb{nullptr};
  if (data.size() != sizeof(lb)) {
    return std::nullopt;
  }

  memcpy(&lb, data.data(), sizeof(lb));
  return this->extract_index(lb);
}

std::optional<size_t> newton_label::extract_index(
    const draggable_label* lb) const noexcept {
  for (size_t idx = 0; idx < this->m_points.size(); idx++) {
    auto& pair = this->m_points[idx];
    if (lb == pair.label.get()) {
      return idx;
    }
  }
  return std::nullopt;
}

void newton_label::dragEnterEvent(QDragEnterEvent* event) {
  if (this->extract_draggable_label(event->mimeData()) != nullptr) {
    event->acceptProposedAction();
  } else {
    scalable_label::dragEnterEvent(event);
  }
}

void newton_label::dropEvent(QDropEvent* event) {
  auto index_opt = this->extract_index(event->mimeData());
  if (!index_opt.has_value()) {
    return;
  }
  auto& pair = this->m_points[index_opt.value()];
  auto pos = event->position();
  auto* current_wind = this->zoomer()->current_result().wind.get();
  const auto new_coord = current_wind->displayed_coordinate(
      {this->height(), this->width()}, {(int)pos.y(), (int)pos.x()});
  pair.coordinate = {new_coord[0], new_coord[1]};

  this->repaint_points(*current_wind);

  pair.label->point_before_drag.reset();

  event->acceptProposedAction();
}

void newton_label::dragMoveEvent(QDragMoveEvent* event) {
  const auto idx_opt = this->extract_index(event->mimeData());
  if (!idx_opt.has_value()) {
    return;
  }
  auto event_pos = event->position().toPoint();
  const auto idx = idx_opt.value();

  const auto& wind = *this->zoomer()->current_result().wind;

  auto current_cplx = wind.displayed_coordinate({this->height(), this->width()},
                                                {event_pos.y(), event_pos.x()});

  this->repaint_point(wind, this->m_points[idx].label.get(),
                      std::complex<double>{current_cplx[0], current_cplx[1]});

  //    fmt::print("newton_label::dragMoveEvent: event_pos = [{}, {}]\n",
  //               event_pos.x(), event_pos.y());
  //    auto current_pos = this->pos() + event_pos;
  //    fmt::print("newton_label::dragMoveEvent: current pos = [{}, {}]\n",
  //               current_pos.x(), current_pos.y());
}