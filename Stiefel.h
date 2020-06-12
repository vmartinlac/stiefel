
#pragma once

#include <Eigen/Eigen>

class StiefelProblem
{
public:

    virtual int getNumVectors() = 0;

    virtual int getDimension() = 0;

    virtual void evaluate(
        const Eigen::MatrixXd& X,
        double& value,
        Eigen::MatrixXd& gradient) = 0;
};

class StiefelAcceleratedGradientSolver
{
public:

    StiefelAcceleratedGradientSolver();
    
    void solve(StiefelProblem* problem, Eigen::MatrixXd& solution, bool use_initial);

protected:

    double dualSpaceInnerProduct(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const Eigen::MatrixXd& Z);

    void retract(
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y,
        Eigen::MatrixXd& Z);

protected:

    double myGammaZero;
    double myLambda;
    double myCR;
    double myCL;
    double myEpsilon;
};

