//
// Created by David on 2023/7/12.
//

// You may need to build the project (run Qt uic code generator) to get
// "ui_launcher_wind.h" resolved

#include "launcher_wind.h"
#include "ui_launcher_wind.h"
#include <QDebug>
#include <QDir>
#include <QProcess>
#include <QMessageBox>

launcher_wind::launcher_wind(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::launcher_wind) {
  ui->setupUi(this);
}

launcher_wind::~launcher_wind() { delete ui; }

tl::expected<void, QString> launcher_wind::grab_compute_configs(
    const QString &dir_path) & noexcept {
  QDir dir{dir_path};
  if (!dir.exists()) {
    return tl::make_unexpected(
        QStringLiteral("%1 doesn't exists.").arg(dir_path));
  }

  this->ui->cb_compute->clear();

  auto entries = dir.entryList();
  for (const auto &filename : entries) {
    // qDebug() << "Processing " << filename;
    if (!filename.endsWith(".json")) {
      continue;
    }
    const QString abs_path = QStringLiteral("%1/%2").arg(dir_path, filename);
    if (!QFile{abs_path}.exists()) {
      continue;
    }

    this->ui->cb_compute->addItem(filename, abs_path);
  }
  return {};
}
tl::expected<void, QString> launcher_wind::grab_render_configs(
    const QString &dir_path) & noexcept {
  QDir dir{dir_path};
  if (!dir.exists()) {
    return tl::make_unexpected(
        QStringLiteral("%1 doesn't exists.").arg(dir_path));
  }

  this->ui->cb_render->clear();

  auto entries = dir.entryList();
  for (const auto &filename : entries) {
    // qDebug() << "Processing " << filename;
    if (!filename.endsWith(".json")) {
      continue;
    }
    const QString abs_path = QStringLiteral("%1/%2").arg(dir_path, filename);
    if (!QFile{abs_path}.exists()) {
      continue;
    }

    this->ui->cb_render->addItem(filename, abs_path);
  }
  return {};
}

void launcher_wind::on_pb_start_clicked() noexcept {
  const auto compute_src = this->ui->cb_compute->currentData().toString();
  const auto render_json = this->ui->cb_render->currentData().toString();

  QString args = QStringLiteral("%1 --rj %2 --scale %3")
                     .arg(compute_src, render_json)
                     .arg(this->ui->sb_scale->value());

  auto arg_list = args.split(' ');

  // auto process = new QProcess{this};

  auto ok = QProcess::startDetached("nfzoom", arg_list, "");
  if (!ok) {
    QMessageBox::warning(this, "Failed to start detached process",
                         QStringLiteral("arguments: %1").arg(args));
    return;
  }
}