//
// Created by joseph on 7/2/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_NEWTON_LABEL_H
#define NEWTON_FRACTAL_ZOOM_NEWTON_LABEL_H

#include <QtWidgets>
#include <scalable_label.h>
#include <QLabel>
#include <memory>
#include <complex>
#include <zoom_window.h>
#include <QEvent>
#include <optional>

class draggable_label;
class newton_label;

struct draw_option {
  uint32_t background_color{0xFFFFFFFF};
  uint32_t text_color{0xFF000000};
  int icon_size{10};
};

class draggable_label final : public QLabel {
 public:
  explicit draggable_label(QWidget* parent = nullptr);
  ~draggable_label() final = default;

  void draw() & noexcept;
  int index{0};
  draw_option option;

  std::optional<QPoint> point_before_drag{std::nullopt};

 signals:
  void dragged(draggable_label* self);
  void released(draggable_label* self);

 protected:
  void paintEvent(QPaintEvent* e) override;

  void mousePressEvent(QMouseEvent* e) override;
  void mouseMoveEvent(QMouseEvent* e) override;
};

struct dragged_option {
  newton_label* label_ptr;
  size_t point_index;
};

class newton_label final : public scalable_label {
 public:
  struct label_point_pair {
    std::unique_ptr<draggable_label> label{nullptr};
    std::complex<double> coordinate;
  };

 private:
  std::vector<label_point_pair> m_points;

 public:
  explicit newton_label(fractal_utils::zoom_window* parent = nullptr)
      : scalable_label(parent){};
  ~newton_label() final { this->m_points.clear(); };

  [[nodiscard]] auto& points() noexcept { return this->m_points; }
  [[nodiscard]] const auto& points() const noexcept { return this->m_points; }

  void repaint_points() & noexcept;

  [[nodiscard]] fractal_utils::zoom_window* parent_wind() const noexcept;

 protected:
  void mousePressEvent(QMouseEvent* e) override;
};

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_LABEL_H
