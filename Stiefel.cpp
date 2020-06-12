#include "Stiefel.h"

StiefelAcceleratedGradientSolver::StiefelAcceleratedGradientSolver()
{
    myGammaZero = 1.0;
    myLambda = 1.5;
    myCR = 1.0;
    myCL = 1.0;
    myEpsilon = 1.0e-5;
}

void StiefelAcceleratedGradientSolver::solve(StiefelProblem* problem, Eigen::MatrixXd& solution, bool use_initial)
{
    const int N = problem->getDimension();
    const int M = problem->getNumVectors();

    Eigen::MatrixXd X;
    Eigen::MatrixXd Y;
    Eigen::MatrixXd Z;
    double value_at_X = 0.0;
    double value_at_Y = 0.0;
    double value_at_Z = 0.0;
    Eigen::MatrixXd gradient_at_X;
    Eigen::MatrixXd gradient_at_Y;
    Eigen::MatrixXd gradient_at_Z;
    double squared_norm_gradient_at_Y = 0.0;

    const double epsilon = 0.0;
    double gamma = myGammaZero;
    int kappa = 0;

    if(use_initial)
    {
        if(solution.rows() != N || solution.cols() != M)
        {
            throw std::runtime_error("internal error");
        }

        X = solution;
    }
    else
    {
        X = Eigen::MatrixXd::Identity(N,M);
    }

    Y = X;

    problem->evaluate(X, value_at_X, gradient_at_X);

    while(dualSpaceInnerProduct(X, gradient_at_X, gradient_at_X) > epsilon*epsilon)
    {
        problem->evaluate(Y, value_at_Y, gradient_at_Y);
        squared_norm_gradient_at_Y = dualSpaceInnerProduct(Y, gradient_at_Y, gradient_at_Y);

        retract(Y, -gamma*gradient_at_Y, Z);
        problem->evaluate(Z, value_at_Z, gradient_at_Z);

        while(value_at_Z < value_at_Y - myCL*gamma*squared_norm_gradient_at_Y)
        {
            gamma *= myLambda;

            retract(Y, -gamma*gradient_at_Y, Z);
            problem->evaluate(Z, value_at_Z, gradient_at_Z);
        }

        while(value_at_Z > value_at_Y  - 0.5*gamma*squared_norm_gradient_at_Y)
        {
            gamma /= myLambda;

            retract(Y, -gamma*gradient_at_Y, Z);
            problem->evaluate(Z, value_at_Z, gradient_at_Z);
        }

        if( value_at_Z > value_at_X - myCR*gamma*squared_norm_gradient_at_Y )
        {
            X = Z;
            Y = Z;
            kappa = 0;
        }
        else
        {
            //Eigen::MatrixXd V = 2*Z*(Eigen::MatrixXd::Identity() + Z.transpose() * X).inv();
            //V -= -1.5

            kappa++;
        }
    }
}

double StiefelAcceleratedGradientSolver::dualSpaceInnerProduct(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const Eigen::MatrixXd& Z)
{
    const int N = X.rows();
    return ( Y.transpose() * ( Eigen::MatrixXd::Identity(N,N) + X*X.transpose() ) * Z ).trace();
}

