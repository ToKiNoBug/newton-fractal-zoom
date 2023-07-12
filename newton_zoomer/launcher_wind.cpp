//
// Created by David on 2023/7/12.
//

// You may need to build the project (run Qt uic code generator) to get
// "ui_launcher_wind.h" resolved

#include "launcher_wind.h"
#include "ui_launcher_wind.h"

launcher_wind::launcher_wind(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::launcher_wind) {
  ui->setupUi(this);
}

launcher_wind::~launcher_wind() { delete ui; }
