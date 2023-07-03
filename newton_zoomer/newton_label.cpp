#include "newton_label.h"
#include <QPainter>

draggable_label::draggable_label(QWidget* parent) : QLabel(parent) {}

void draggable_label::draw() & noexcept {
  this->resize(this->option.icon_size, this->option.icon_size);

  QPainter painter{this};
  {
    QBrush brush{QColor{this->option.background_color}};
    painter.setBrush(brush);
    QPen pen{QColor{this->option.text_color}};
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

void draggable_label::mouseMoveEvent(QMouseEvent* event) {
  if (!event->buttons() & Qt::MouseButton::LeftButton) {
    return;
  }
  if ((event->pos() - this->point_before_drag.value()).manhattanLength() <
      QApplication::startDragDistance()) {
    return;
  }
}

fractal_utils::zoom_window* newton_label::parent_wind() const noexcept {
  return dynamic_cast<fractal_utils::zoom_window*>(this->parent());
}

void newton_label::repaint_points() & noexcept {
  for (size_t idx = 0; idx < this->m_points.size(); idx++) {
    auto& point = this->m_points[idx];
  }
}

void newton_label::mousePressEvent(QMouseEvent* e) {
  if (e->buttons() & Qt::MouseButton::LeftButton) {
    for (size_t idx = 0; idx < this->m_points.size(); idx++) {
      auto& point = this->m_points[idx];
      if (point.label->geometry().contains(e->pos())) {
        QMimeData* md = new QMimeData;
        {
          dragged_option opt{this, idx};
          md->setData("info", QByteArray{(const char*)&opt, sizeof(opt)});
        }
        QDrag* drag = new QDrag{this};

        drag->setMimeData(md);

        Qt::DropAction da = drag->exec();
      }
    }
  }
  
  scalable_label::mousePressEvent(e);
}