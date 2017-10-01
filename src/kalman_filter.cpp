#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  //new estimate
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(4, 4);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd z_pred = cartesianToPolar(x_);
  VectorXd y = z - z_pred;

  if (y(1) > M_PI)
	  while (y(1) > M_PI)
		  y(1) -= 2 * M_PI;
  else if (y(1) < -M_PI)
	  while (y(1) < -M_PI)
		  y(1) =+ 2 * M_PI;

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  //new estimate
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(4, 4);
  P_ = (I - K * H_) * P_;
}

VectorXd KalmanFilter::cartesianToPolar(const VectorXd &x) {
  VectorXd h(3);

  float px = x(0);
  float py = x(1);
  float vx = x(2);
  float vy = x(3);

  float rho = sqrt(px * px + py * py);
  float phi = atan2(py, px);
  float rho_dot = (px * vx + py * vy) / rho;

  if(fabs(rho) < 0.0001){
    std::cout << "cartesianToPolar () - Error - Division by Zero" << std::endl;
    return h;
  }

  h << rho,
	   phi,
	   rho_dot;

  return h;
}

