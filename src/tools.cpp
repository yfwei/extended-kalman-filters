#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  if (estimations.size() <= 0)
    return rmse;

  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size())
    return rmse;

  //accumulate squared residuals
  for(int i=0; i < estimations.size(); ++i){
    VectorXd e = estimations[i] - ground_truth[i];
    e = (e.array() * e.array());
    rmse += e;
  }

  //calculate the mean
  rmse /= estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3, 4);

  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float c1 = px * px + py * py;
  float c2 = sqrt(c1);
  float c3 = c1 * c2;

  //check division by zero
  if(fabs(c1) < 0.0001){
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }

  float c4 = px / c2;
  float c5 = py / c2;

  Hj << c4, c5, 0, 0,
		-py / c1, px / c1, 0, 0,
		py * (vx * py - vy * px) / c3, px * (vy * px - vx * py) / c3, c4, c5;

  return Hj;
}
