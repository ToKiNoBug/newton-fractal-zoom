//
// Created by joseph on 7/2/23.
//

#ifndef NEWTON_FRACTAL_ZOOM_NEWTON_LABEL_H
#define NEWTON_FRACTAL_ZOOM_NEWTON_LABEL_H

#include <QWidget>
#include <scalable_label.h>
#include <QLabel>
#include <memory>
#include <complex>
#include <zoom_window.h>
#include <QEvent>
#include <optional>
#include <newton_fractal.h>
#include <QMouseEvent>
#include <QEnterEvent>

class draggable_label;
class newton_label;
class newton_zoomer;

struct draw_option {
  uint32_t background_color{0xFFFFFFFF};
  uint32_t text_color{0xFF000000};
  int icon_size{10};
};

class draggable_label final : public QLabel {
 private:
  bool m_draw_cross{false};

  [[nodiscard]] newton_label* impl_parent_label() const noexcept;

 public:
  explicit draggable_label(QWidget* parent = nullptr);
  ~draggable_label() final = default;

  void draw() & noexcept;
  void draw(int index, const draw_option& option) & noexcept;

  std::optional<QPoint> point_before_drag{std::nullopt};

  [[nodiscard]] std::optional<int> index() const noexcept;
  [[nodiscard]] draw_option option() const noexcept;

  [[nodiscard]] newton_label* parent_label() noexcept {
    return this->impl_parent_label();
  }
  [[nodiscard]] const newton_label* parent_label() const noexcept {
    return this->impl_parent_label();
  }

 signals:
  //  void dragged(draggable_label* self, QPoint pos);
  //  void released(draggable_label* self);

 protected:
  void paintEvent(QPaintEvent* e) override;

  void mousePressEvent(QMouseEvent* e) override;
  void mouseMoveEvent(QMouseEvent* e) override;

  void enterEvent(QEnterEvent* event) override;
  void leaveEvent(QEvent* event) override;

  // void dragMoveEvent(QDragMoveEvent* e) override;
};

void draw_point(QPaintDevice& device, const QPoint& offset,
                std::optional<int> number, const QFontMetrics& font_metrics,
                const draw_option& option) noexcept;

void draw_cross(QPaintDevice& device, const QPoint& offset,
                const draw_option& option) noexcept;

struct dragged_option {
  newton_label* label_ptr;
  size_t point_index;
};

class newton_label final : public scalable_label {
  // Q_OBJECT
 public:
  struct label_point_pair {
    std::unique_ptr<draggable_label> label{nullptr};
    std::complex<double> coordinate;
  };

 private:
  std::vector<label_point_pair> m_points;
  newton_zoomer* const m_zoomer;

  std::optional<QPoint> m_drawable_point{std::nullopt};

 public:
  draw_option option;
  explicit newton_label(newton_zoomer* parent);

  ~newton_label() final { this->m_points.clear(); }

  [[nodiscard]] auto& points() noexcept { return this->m_points; }
  [[nodiscard]] const auto& points() const noexcept { return this->m_points; }

  void repaint_point(const fractal_utils::wind_base& wind,
                     draggable_label* label,
                     std::complex<double> coordinate) const noexcept;

  void repaint_point(const fractal_utils::wind_base& wind,
                     size_t idx) & noexcept;

  [[nodiscard]] newton_zoomer* zoomer() const noexcept;

  void reset(const nf::newton_equation_base& equation) & noexcept;

  void repaint_points(const fractal_utils::wind_base& wind) & noexcept;

  void clear_drawable_point() & noexcept {
    this->m_drawable_point.reset();
    this->repaint();
  }

  [[nodiscard]] draggable_label* extract_draggable_label(
      const QMimeData* src) const noexcept;

  [[nodiscard]] std::optional<size_t> extract_index(
      const QMimeData* src) const noexcept;

  [[nodiscard]] std::optional<size_t> extract_index(
      const draggable_label* ptr) const noexcept;

  [[nodiscard]] std::vector<std::complex<double>> current_points()
      const noexcept;

 protected:
  void mousePressEvent(QMouseEvent* e) override;

  void mouseMoveEvent(QMouseEvent* e) final;

  void dragEnterEvent(QDragEnterEvent* event) override;

  void dropEvent(QDropEvent* event) override;
  void dragMoveEvent(QDragMoveEvent* e) override;

  void paintEvent(QPaintEvent* e) override;

  void leaveEvent(QEvent* e) override;
};

#endif  // NEWTON_FRACTAL_ZOOM_NEWTON_LABEL_H
