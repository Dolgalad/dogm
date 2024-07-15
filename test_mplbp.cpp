#include<iostream>
//#include<cmath>
//#include<vector>
#include<chrono>
#include<filesystem>

#include<torch/torch.h>

//#include"metis_graph.h"
#include"gnngraph.h"
#include"belief_propagation.h"


using namespace std;
using namespace torch;
using namespace torch::indexing;
using namespace std::chrono;
using namespace std::filesystem;

FactorGraph from_metis(METISGraph& g) {
	struct FactorGraph fg;
	fg.n = g.nv();
	fg.nF = g.nv()+g.ne();
	for(int i=0;i<fg.n;i++) {
		fg.factor_neighbors.push_back({i});
		fg.variable_neighbors.push_back({i});
	}
	vector<std::pair<int,int> > edges = g.edges();
	for(size_t i=0;i<g.ne();i++) {
		std::pair<int,int> e = edges[i];
		fg.factor_neighbors.push_back({e.first,e.second});
		fg.variable_neighbors[e.first].push_back(i+fg.n);
		fg.variable_neighbors[e.second].push_back(i+fg.n);
	}

	cout << "in from_metis" << endl;
	cout << "\t" << fg.factor_neighbors.size() << endl;
	cout << "\t" << fg.variable_neighbors.size() << endl;
	return fg;
}

int main(int argc, char ** argv) {
	cout << "Testing Sum-Product/Max-Sum Loopy Belief Propagation implementation" << endl;

	// create a factor graph
	METISGraph g = METISGraph::load("mis_er_n5-8_p0.15/train/1.graph");
	cout << "before creating factor graph" << endl;
	struct FactorGraph fg = from_metis(g);
	cout << "after creating factor graph" << endl;


	struct FGEdges edges;
	cout << "before make_edges" << endl;
	edges.make_edges(fg);
	cout << "after make_edges" << endl;


	// initialize data
	//auto opts = torch::TensorOptions().requires_grad(true);
	//Tensor theta = torch::ones({fg.num_sufficient_statistics()}, torch::requires_grad());
	//Tensor theta = torch::arange(fg.num_sufficient_statistics()).to(torch::kFloat);
	Tensor theta = torch::randn({fg.num_sufficient_statistics()}, torch::requires_grad()).to(torch::kFloat);

	Tensor q = torch::zeros({fg.num_f2v_messages()});
	//Tensor mu = torch::zeros({fg.num_sufficient_statistics()});

	auto start = high_resolution_clock::now();
	//Tensor ans = loopy_belief_propagation(theta, q, edges, fg, "max_sum", 100, 1e-6);
	Tensor ans = loopy_belief_propagation(theta, q, edges, fg, "sum_product", 100, 1e-6);

	auto stop = high_resolution_clock::now();
	cout << ans << endl;
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "t = " << duration.count() << endl;

	ans.sum().backward();
	cout << theta.grad() << endl;


	return 0;
}
