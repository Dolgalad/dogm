#include<iostream>
#include<filesystem>
#include<string>

#include<boost/graph/adjacency_list.hpp>
#include<boost/graph/erdos_renyi_generator.hpp>
#include<boost/graph/graph_traits.hpp>
#include <boost/random/uniform_int_distribution.hpp>
//#include<boost/random/linear_congruential.hpp>
#include <boost/random/mersenne_twister.hpp>

//#include"metis_graph.h"
#include"mis.h"

using namespace std;
using namespace std::filesystem;
using namespace boost;

typedef adjacency_list < setS, vecS, undirectedS> Graph;
//typedef erdos_renyi_iterator<boost::minstd_rand, Graph> ERGen;
typedef erdos_renyi_iterator<boost::random::mt19937, Graph> ERGen;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef property_map<Graph, vertex_index_t>::type IndexMap;
typedef graph_traits<Graph>::vertex_iterator vertex_iter;

METISGraph from_boost_graph(Graph& g) {
	unsigned long n = num_vertices(g);
	unsigned long m = num_edges(g);
	vector<vector<unsigned long> > vnl, ewl, vwl;

        IndexMap index = get(vertex_index, g);

	graph_traits<Graph>::vertex_iterator vi, vi_end;
        graph_traits<Graph>::adjacency_iterator ai, ai_end;

        for (boost::tie(vi, vi_end) = vertices(g); vi != vi_end; ++vi) {
		vector<unsigned long> neighbors;
                for (boost::tie(ai, ai_end) = adjacent_vertices(*vi, g);
                     ai != ai_end; ++ai)
			neighbors.push_back(index[*ai]);
		vnl.push_back(neighbors);
	}
	return METISGraph(n,m,0,0,vnl,ewl,vwl);
}


int main(int argc, char ** argv) {
	if(argc<7) {
		cout << "Wrong number of arguments" << endl;
		return 1;
	}
	std::filesystem::path output_dir = std::filesystem::path(argv[1]);
	std::filesystem::path train_dir = output_dir / "train";
	std::filesystem::path test_dir = output_dir / "test";

	uint64_t train_n = atoi(argv[2]), test_n = atoi(argv[3]);
	uint64_t nv_min = atoi(argv[4]), nv_max = atoi(argv[5]);
	double edge_prob = atof(argv[6]);
	cout << "Creating MIS Erdos-Renyi dataset" << endl;
	cout << "\toutput directory : " << output_dir << endl;
	cout << "\ttrain instances  : " << train_n << endl;
	cout << "\ttest instances : " << test_n << endl;
	cout << "\tvertex count range : [" << nv_min << "," << nv_max << "]" << endl;
	cout << "\tedge probability : " << edge_prob << endl;

	bool create_dirs = true;

	if(exists(output_dir) || exists(train_dir) || exists(test_dir)) {
		cout << "Output director " << output_dir << " already exists. Overwrite contents [y,n]? ";

		string ans;
		cin >> ans;
		if(ans.compare("n")==0 || ans.compare("N")==0) create_dirs=false;
	}
	if(create_dirs) {
		if(!exists(output_dir)) create_directory(output_dir);
		if(!exists(train_dir)) create_directory(train_dir);
		if(!exists(test_dir)) create_directory(test_dir);
	}

	// train instances
	//boost::minstd_rand gen;
	boost::random::mt19937 gen;
	boost::random::uniform_int_distribution n_dist(nv_min, nv_max);
	for(uint64_t i=0;i<train_n;i++) {
		int n = n_dist(gen);
		Graph g(ERGen(gen, n, edge_prob), ERGen(), n);
		METISGraph mg = from_boost_graph(g);
		cout << "Solving with CPLEX" << endl << flush;
		struct MISLPSolution sol = cplex_solve_relaxed_mis(g);
		//cplex_solve_mis(g);

		path graph_path = train_dir / to_string(i).append(".graph");
		path features_path = train_dir / to_string(i).append("_features.csv");
		cout << "Graph path    : " << graph_path << endl;
		cout << "Features path : " << features_path << endl;

		// save graph
		mg.save(graph_path);
		// save lp solution data
		sol.save(features_path);
	}

	// test instances
	for(uint64_t i=0;i<test_n;i++) {
		int n = n_dist(gen);
		Graph g(ERGen(gen, n, edge_prob), ERGen(), n);
		METISGraph mg = from_boost_graph(g);
		cout << "Solving with CPLEX" << endl << flush;
		struct MISLPSolution sol = cplex_solve_relaxed_mis(mg);
		//cplex_solve_mis(g);

		path graph_path = test_dir / to_string(i).append(".graph");
		path features_path = test_dir / to_string(i).append("_features.csv");
		cout << "Graph path    : " << graph_path << endl;
		cout << "Features path : " << features_path << endl;

		// save graph
		mg.save(graph_path);
		// save lp solution data
		sol.save(features_path);
	}


	return 0;


}
