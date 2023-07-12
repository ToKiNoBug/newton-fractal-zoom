//
// Created by David on 2023/7/12.
//

#ifndef NEWTON_FRACTAL_ZOOM_LAUNCHER_WIND_H
#define NEWTON_FRACTAL_ZOOM_LAUNCHER_WIND_H

#include <QMainWindow>
#include <tl/expected.hpp>

QT_BEGIN_NAMESPACE
namespace Ui {
class launcher_wind;
}
QT_END_NAMESPACE

class launcher_wind : public QMainWindow {
  Q_OBJECT

 public:
  explicit launcher_wind(QWidget *parent = nullptr);
  ~launcher_wind() override;

  [[nodiscard]] tl::expected<void, QString> grab_compute_configs(
      const QString &dir) & noexcept;
  [[nodiscard]] tl::expected<void, QString> grab_render_configs(
      const QString &dir) & noexcept;

 private:
  Ui::launcher_wind *ui;

 private slots:
  void on_pb_start_clicked() noexcept;
};

#endif  // NEWTON_FRACTAL_ZOOM_LAUNCHER_WIND_H
