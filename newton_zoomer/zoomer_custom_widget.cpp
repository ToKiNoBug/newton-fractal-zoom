//
// Created by David on 2023/7/29.
//

// You may need to build the project (run Qt uic code generator) to get
// "ui_zoomer_custom_widget.h" resolved

#include "zoomer_custom_widget.h"
#include "ui_zoomer_custom_widget.h"

zoomer_custom_widget::zoomer_custom_widget(QWidget *parent)
    : QWidget(parent), ui(new Ui::zoomer_custom_widget) {
  ui->setupUi(this);
}

zoomer_custom_widget::~zoomer_custom_widget() { delete ui; }

QPushButton *zoomer_custom_widget::pushbutton_edit_points() const noexcept {
  return this->ui->pb_edit_points;
}

QToolButton *zoomer_custom_widget::toolbutton_add_point() const noexcept {
  return this->ui->tb_add_point;
}

QToolButton *zoomer_custom_widget::toolbutton_erase_point() const noexcept {
  return this->ui->tb_erase_point;
}