#include <QApplication>
#include <QMainWindow>
#include <QMessageBox>
#include <QDir>
#include <QFileInfo>
#include <QLocale>
#include <QTranslator>
#include "launcher_wind.h"

int main(int argc, char** argv) {
  QApplication qapp{argc, argv};
  QTranslator translator;
  {
    QLocale locale;
    if (locale.language() == QLocale::Language::Chinese) {
      const auto ok = translator.load(":/i18n/nfzoom_launcher_zh_CN.qm");
      if (ok) {
        QCoreApplication::installTranslator(&translator);
      }
    }
  }

  launcher_wind wind;
  wind.show();

  QString location;
  { location = QFileInfo{argv[0]}.dir().path(); }

  {
    auto err = wind.grab_compute_configs(
        QStringLiteral("%1/../compute_presets").arg(location));
    if (!err) {
      QMessageBox::critical(&wind, "Failed to load compute presets",
                            err.error());
      return 1;
    }
  }
  {
    auto err = wind.grab_render_configs(
        QStringLiteral("%1/../render_presets").arg(location));
    if (!err) {
      QMessageBox::critical(&wind, "Failed to load render presets",
                            err.error());
      return 1;
    }
  }

  return qapp.exec();
}