
#include "newton_equation.hpp"
#include <sstream>

namespace nf = newton_fractal;

using mpc_eq = nf::newton_equation<boostmp::mpc_complex>;

nf::newton_equation<boostmp::mpc_complex>::newton_equation(
    std::span<boostmp::mpc_complex> points) {
  this->_parameters.reserve(points.size());
  for (const auto& p : points) {
    this->add_point(p);
  }
}

nf::newton_equation<boostmp::mpc_complex>::newton_equation(
    std::span<boostmp::mpc_complex> points, int precision)
    : newton_equation{points} {
  this->set_precision(precision);
}

std::optional<int> mpc_eq::precision() const noexcept {
  if (this->_parameters.empty()) {
    return std::nullopt;
  }

  int ret = this->_parameters.front().precision();
  for (const auto& val : this->_parameters) {
    if (ret != val.precision()) {
      return std::nullopt;
    }
  }

  return ret;
}

void mpc_eq::set_precision(int p) & noexcept {
  for (auto& val : this->_parameters) {
    val.precision(p);
  }
  // this->buffer.set_precision(p);
}

void mpc_eq::add_point(const boostmp::mpc_complex& p) & noexcept {
  const int order_before = this->order();

  if (order_before > 0) {
    // temp = this->parameters().back() * (-p);
    this->parameters().emplace_back(0);
  } else {
    this->parameters().emplace_back(-p);
    return;
  }

  complex_type temp;
  for (int i = order_before - 1; i >= 0; i--) {
    temp = this->parameters()[i];
    temp *= -p;
    this->parameters()[i + 1] += temp;
  }
  this->parameters()[0] -= p;
}

void format_complex(const boostmp::mpc_complex& z,
                    std::stringstream& ss) noexcept {
  ss << z.real();
  if (z.imag() >= 0) {
    ss << '+';
  }
  ss << z.imag() << 'i';
}

// std::ostream & operator<<(const )

std::string mpc_eq::to_string() const noexcept {
  if (this->order() == 0) {
    return "0 = 0";
  }

  std::stringstream ss;

  ss << fmt::format("z^{}", this->order());

  for (int i = 0; i < this->parameters().size(); i++) {
    const int current_order = this->order() - i - 1;
    if (current_order > 0) {
      ss << " + (" << this->parameters()[i] << ") * z^" << current_order;
      /*
      fmt::format_to(std::back_inserter(ret), " + ({}+{}i) * z^{}",
                     double(this->parameters()[i].real()),
                     double(this->parameters()[i].imag()), current_order);
      */
    } else {
      ss << " + (" << this->parameters()[i] << ") = 0";
      /*
      fmt::format_to(std::back_inserter(ret), " + ({}+{}i) = 0",
                     double(this->parameters()[i].real()),
                     double(this->parameters()[i].imag()));
      */
    }
  }

  return ss.str();
}

mpc_eq::buffer_t& mpc_eq::buffer() noexcept {
  thread_local buffer_t buf;
  return buf;
}

void mpc_eq::compute_difference(const complex_type& z,
                                complex_type& dst) const noexcept {
  // dst = 0;
  dst = this->item_at_order(0);
  auto& z_power = buffer().complex_arr[0];
  z_power = z;

  auto& temp = buffer().complex_arr[1];
  for (int o = 1; o < this->order(); o++) {
    temp = z_power * this->item_at_order(o);
    dst += temp;
    z_power *= z;
  }

  dst += z_power;
}

void mpc_eq::iterate_inplace(complex_type& z) const noexcept {
  assert(this->order() > 1);
  auto& buf = this->buffer();

  auto& f = buf.complex_arr[0];
  auto& df = buf.complex_arr[1];

  auto& z_power = buf.complex_arr[2];
  auto& temp = buf.complex_arr[3];

  f = this->item_at_order(0);
  df = this->item_at_order(1);

  z_power = z;

  for (int n = 1; n < this->order(); n++) {
    {
      temp = this->item_at_order(n) * z_power;
      f += temp;
    }
    {
      if (n + 1 >= this->order()) {
        temp = z_power;
        temp *= (n + 1);
      } else {
        temp = this->item_at_order(n + 1) * z_power;
        temp *= (n + 1);
      }
      df += temp;
    }
    z_power *= z;
  }

  f += z_power;
  f /= df;
  f *= -1;
  z += f;
}

mpc_eq::complex_type mpc_eq::iterate(const complex_type& z) const noexcept {
  complex_type temp{z};
  this->iterate_inplace(temp);
  return temp;
}

void mpc_eq::iterate_n(complex_type& z, int n) const noexcept {
  for (int i = 0; i < n; i++) {
    this->iterate_inplace(z);
  }
}
