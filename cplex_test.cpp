#include<iostream>

#include"test_lib.h"


//ILOSTLBEGIN
using namespace std;

int main(int argc, char ** argv) {

	test_cplex();
    std::cout << "Testing cplex" << std::endl;
}

// make cmd :g++ -m64 -fPIC -fno-strict-aliasing -fexceptions -DNDEBUG -I/opt/ibm/ILOG/CPLEX_Studio221/cplex/include -I/opt/ibm/ILOG/CPLEX_Studio221/concert/include cplex_test.cpp 
//
// g++ -m64 -fPIC -fno-strict-aliasing -fexceptions -DNDEBUG -I/opt/ibm/ILOG/CPLEX_Studio221/cplex/include -I/opt/ibm/ILOG/CPLEX_Studio221/concert/include -L/opt/ibm/ILOG/CPLEX_Studio221/cplex/lib/x86-64_linux/static_pic -L/opt/ibm/ILOG/CPLEX_Studio221/concert/lib/x86-64_linux/static_pic cplex_test.cpp -lconcert -lilocplex -lcplex -lm -lpthread -ldl
