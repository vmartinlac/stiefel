#include <iostream>
#include "Stiefel.h"

class MyProblem : public StiefelProblem
{
public:

    MyProblem()
    {
        myMat.resize(3,3);
        myMat <<
            3.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 5.0;
    }

    int getNumVectors()
    {
        return 1;
    }

    int getDimension()
    {
        return 3;
    }

    void evaluate(
        const Eigen::MatrixXd& X,
        double& value,
        Eigen::MatrixXd& gradient)
    {
        value = (X.transpose() * myMat * X)(0,0);
        gradient = 2.0 * myMat * X;
    }

protected:

    Eigen::MatrixXd myMat;
};

int main(int num_args, char** args)
{
    MyProblem pb;

    Eigen::MatrixXd X(3, 1);
    X <<
        1.0,
        0.0,
        0.0;

    StiefelAcceleratedGradientSolver solver;
    solver.solve(&pb, X, true);

    return 0;
}

