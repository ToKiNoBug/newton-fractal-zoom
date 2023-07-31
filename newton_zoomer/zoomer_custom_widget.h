//
// Created by David on 2023/7/29.
//

#ifndef NEWTON_FRACTAL_ZOOM_ZOOMER_CUSTOM_WIDGET_H
#define NEWTON_FRACTAL_ZOOM_ZOOMER_CUSTOM_WIDGET_H

#include <QWidget>
#include <QPushButton>
#include <QToolButton>

QT_BEGIN_NAMESPACE
namespace Ui {
class zoomer_custom_widget;
}
QT_END_NAMESPACE

class zoomer_custom_widget final : public QWidget {
  Q_OBJECT

 public:
  explicit zoomer_custom_widget(QWidget *parent = nullptr);
  ~zoomer_custom_widget() final;

  [[nodiscard]] QPushButton *pushbutton_edit_points() const noexcept;
  [[nodiscard]] QToolButton *toolbutton_add_point() const noexcept;
  [[nodiscard]] QToolButton *toolbutton_erase_point() const noexcept;

  void retranslate_ui() & noexcept;

 private:
  Ui::zoomer_custom_widget *ui;
};

#endif  // NEWTON_FRACTAL_ZOOM_ZOOMER_CUSTOM_WIDGET_H
