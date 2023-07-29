//
// Created by David on 2023/7/29.
//

// You may need to build the project (run Qt uic code generator) to get
// "ui_point_form.h" resolved

#include "point_form.h"
#include "ui_point_form.h"
#include "newton_zoomer.h"

point_model::point_model(QObject *parent) : QAbstractTableModel{parent} {}

void point_model::reset_points(
    std::span<const std::complex<double>> points) & noexcept {
  this->m_points.resize(points.size());
  memcpy(this->m_points.data(), points.data(), points.size_bytes());

  this->refresh_all_data();
}

void point_model::refresh_all_data() & noexcept {
  this->dataChanged(
      this->createIndex(0, 0),
      this->createIndex(this->rowCount({}), this->columnCount({})));
}

Qt::ItemFlags point_model::flags(const QModelIndex &qmi) const {
  if (!qmi.isValid()) {
    return QAbstractTableModel::flags(qmi);
  }

  return Qt::ItemFlags{Qt::ItemFlag::ItemIsEnabled,
                       Qt::ItemFlag::ItemIsEditable,
                       Qt::ItemFlag::ItemIsSelectable};
}

QVariant point_model::data(const QModelIndex &qmi, int role) const {
  auto default_result = QVariant{};
  if (!qmi.isValid()) {
    return default_result;
  }
  if (qmi.row() >= this->rowCount({}) ||
      qmi.column() >= this->columnCount({})) {
    return default_result;
  }

  switch (role) {
    case Qt::ItemDataRole::DisplayRole: {
      const auto &cplx = this->m_points[qmi.row()];
      if (qmi.column() == 0) {
        return QString::number(cplx.real());
      }
      if (qmi.column() == 1) {
        return QString::number(cplx.imag());
      }
      return default_result;
    }

    case Qt::ItemDataRole::WhatsThisRole: {
      if (qmi.column() == 0) {
        return QStringLiteral("The real part of a point");
      }
      if (qmi.column() == 1) {
        return QStringLiteral("The imaginary part of a point");
      }
      return default_result;
    }

    default:
      return default_result;
  }
}

bool point_model::setData(const QModelIndex &qmi, const QVariant &data,
                          int role) {
  if (!qmi.isValid()) {
    return false;
  }
  if (role != Qt::ItemDataRole::EditRole) {
    return false;
  }

  const int idx = qmi.row();

  const QString str = data.toString();
  bool ok = false;
  const double val = str.toDouble(&ok);
  if (!ok) {
    return false;
  }
  if (!std::isfinite(val)) {
    return false;
  }

  const bool is_real = (qmi.column() == 0);

  auto &cplx_num = this->m_points[idx];
  if (is_real) {
    cplx_num.real(val);
  } else {
    cplx_num.imag(val);
  }

  emit this->dataChanged(qmi, qmi, {Qt::ItemDataRole::DisplayRole});

  return true;
}

point_form::point_form(std::span<const std::complex<double>> points,
                       newton_zoomer *parent)
    : QWidget(parent), ui(new Ui::point_form) {
  ui->setupUi(this);
  this->m_point_model = new point_model{this};
  this->m_point_model->reset_points(points);
  this->ui->tv_buttons->setModel(this->m_point_model);
}

point_form::~point_form() { delete ui; }

void point_form::on_dbb_buttonbox_accepted() {
  emit this->points_changed(this->m_point_model->points());
  delete this;
}

void point_form::on_dbb_buttonbox_rejected() { delete this; }