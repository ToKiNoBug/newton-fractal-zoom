//
// Created by David on 2023/7/29.
//

#ifndef NEWTON_FRACTAL_ZOOM_POINT_FORM_H
#define NEWTON_FRACTAL_ZOOM_POINT_FORM_H

#include <QWidget>
#include <QAbstractTableModel>
#include <complex>
#include <span>

class newton_zoomer;

QT_BEGIN_NAMESPACE
namespace Ui {
class point_form;
}
QT_END_NAMESPACE

class point_model : public QAbstractTableModel {
 private:
  std::vector<std::complex<double>> m_points;

 public:
  explicit point_model(QObject *parent = nullptr);
  ~point_model() override = default;

  void reset_points(std::span<const std::complex<double>> points) & noexcept;

  [[nodiscard]] int rowCount(const QModelIndex &qmi) const final {
    if (qmi.isValid()) {
      return 0;
    }
    return (int)this->m_points.size() + 1;
  }

  [[nodiscard]] int columnCount(const QModelIndex &qmi) const final {
    if (qmi.isValid()) {
      return 0;
    }
    return 2;
  }

  [[nodiscard]] QVariant headerData(int section, Qt::Orientation orientation,
                                    int role) const final;

  [[nodiscard]] Qt::ItemFlags flags(const QModelIndex &qmi) const final;

  [[nodiscard]] QVariant data(const QModelIndex &qmi, int role) const final;

  [[nodiscard]] bool setData(const QModelIndex &qmi, const QVariant &data,
                             int role) final;

  [[nodiscard]] auto &points() const noexcept { return this->m_points; }

  void refresh_all_data() & noexcept;
};

class point_form : public QWidget {
  Q_OBJECT
 private:
  Ui::point_form *ui;
  point_model *m_point_model;

 public:
  explicit point_form(std::span<const std::complex<double>> points,
                      newton_zoomer *parent = nullptr);
  ~point_form() override;

 signals:

  void points_changed(std::span<const std::complex<double>> points);

 public slots:
  void on_dbb_buttonbox_accepted();
  void on_dbb_buttonbox_rejected();
};

#endif  // NEWTON_FRACTAL_ZOOM_POINT_FORM_H
