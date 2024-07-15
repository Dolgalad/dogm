#ifndef MIS_H
#define MIS_H

#include<fstream>
#include<vector>
#include<filesystem>

#include<boost/graph/adjacency_list.hpp>
#include<boost/graph/erdos_renyi_generator.hpp>
#include<boost/graph/graph_traits.hpp>
#include<boost/random/mersenne_twister.hpp>

#include"ilcplex/cplex.h"
#include"ilcplex/ilocplex.h"

#include"metis_graph.h"

//using namespace std;


using namespace std;
using namespace std::filesystem;
using namespace boost;

typedef adjacency_list < setS, vecS, undirectedS> Graph;
typedef erdos_renyi_iterator<boost::random::mt19937, Graph> ERGen;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef property_map<Graph, vertex_index_t>::type IndexMap;
typedef graph_traits<Graph>::vertex_iterator vertex_iter;

struct MISLPSolution {
	vector<double> x, cost, reduced_cost;
	vector<double> slack, rhs, dual;
	void save(path filename);
};

struct MISLPSolution cplex_solve_relaxed_mis(Graph& g);
struct MISLPSolution cplex_solve_relaxed_mis(METISGraph& g);

void cplex_solve_mis(Graph& g);


#endif
