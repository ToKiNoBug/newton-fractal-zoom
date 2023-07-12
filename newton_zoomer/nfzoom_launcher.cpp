#include <QApplication>
#include <QMainWindow>
#include "launcher_wind.h"

int main(int argc, char** argv) {
  QApplication qapp{argc, argv};

  launcher_wind wind;
  wind.show();

  return qapp.exec();
}