#include<iostream>
#include<filesystem>

#include<torch/torch.h>

#include"belief_propagation.h"
#include"message_passing.h"

using namespace std;
using namespace std::filesystem;
using namespace std::chrono;
using namespace torch;
using namespace torch::indexing;


FactorGraph from_metis(METISGraph& g) {
	struct FactorGraph fg;
	fg.n = g.nv();
	fg.nF = g.nv()+g.ne();
	for(int i=0;i<fg.n;i++) {
		fg.factor_neighbors.push_back({i});
		fg.variable_neighbors.push_back({i});
	}
	vector<std::pair<int,int> > edges = g.edges();
	for(long unsigned int i=0;i<g.ne();i++) {
		std::pair<int,int> e = edges[i];
		fg.factor_neighbors.push_back({e.first,e.second});
		fg.variable_neighbors[e.first].push_back(i+fg.n);
		fg.variable_neighbors[e.second].push_back(i+fg.n);
	}
	return fg;
};



struct MISGNNData {
	torch::Tensor edge_index;
	torch::Tensor x,y;
	torch::Tensor var_mask;
	torch::Tensor var_marginal_mask;
	shared_ptr<struct FGEdges> bp_edges;
	shared_ptr<struct FactorGraph> fg;
	int n,m;

	MISGNNData() {
		//cout << "MISGNNData empty constructor" << endl;
		bp_edges = make_shared<struct FGEdges>();
		fg = make_shared<struct FactorGraph>();
	};
	MISGNNData(const struct MISGNNData& other) : edge_index(other.edge_index), x(other.x), y(other.y), var_mask(other.var_mask), var_marginal_mask(other.var_marginal_mask), bp_edges(other.bp_edges), fg(other.fg), n(other.n), m(other.m)	{
		//cout << "MISGNNData copy constructor" << endl;
	};


	void to(torch::Device device, bool pin_memory=false) {
		edge_index = edge_index.to(device, pin_memory);
		x = x.to(device, pin_memory);
		y = y.to(device, pin_memory);
		var_mask = var_mask.to(device, pin_memory);
		var_marginal_mask = var_marginal_mask.to(device, pin_memory);
		bp_edges->to(device, pin_memory);
	}

	void load_graph(path filename) {
		METISGraph g = METISGraph::load(filename);
		//struct FactorGraph tmp_fg = from_metis(g);
		fg = make_shared<struct FactorGraph>(from_metis(g));


		bp_edges->make_edges(*fg);
		//fg = make_shared<struct FactorGraph>(tmp_fg);

		n=g.nv(), m=g.ne();
		var_mask = torch::zeros({n+m}).to(torch::kBool);
		var_mask.index({Slice(0,n)}) = 1;
		edge_index = torch::zeros({4*m,2}).to(torch::kLong);

		int nSS = fg->num_sufficient_statistics();

		var_marginal_mask = torch::zeros({nSS},{torch::kBool});
		var_marginal_mask.index_put_({Slice(0,2*n)}, tensor({0,1}).to(torch::kBool).repeat(n));

		vector<std::pair<int,int> > edges;
		int c = 0;
		for(int i=0;i<n;i++) {
			vector<unsigned long> neighbors = g.neighbors(i);
			for(size_t j=0;j<neighbors.size();j++) {
				std::pair<int,int> edge(i,neighbors[j]);
				//if(find(edges.begin(), edges.end(), EdgeFinder(edge))!=std::string::npos) {
				if(find_if(edges.begin(), edges.end(), EdgeFinder(edge))==edges.end()) {
					int u = edge.first;
					int v = edge.second;
					edge_index.index({4*c,0}) = u;
					edge_index.index({4*c,1}) = c+n;
					edge_index.index({4*c+1,0}) = c+n;
					edge_index.index({4*c+1,1}) = u;
					edge_index.index({4*c+2,0}) = v;
					edge_index.index({4*c+2,1}) = c+n;
					edge_index.index({4*c+3,0}) = c+n;
					edge_index.index({4*c+3,1}) = v;
					edges.push_back(edge);
					c++;
				}
			}
		}

	}
	void load_features(path filename) {

		x = torch::zeros({n+m,3}).to(torch::kFloat);

		ifstream file;
		file.open(filename);
		if(file.is_open()) {
			string line;
			int c = 0;
			while(getline(file, line)) {
				vector<string> w = split_string(line,",");
				if(w.size()==3) {
					x.index({c,0}) = stof(w[0]);
					x.index({c,1}) = stof(w[1]);
					x.index({c,2}) = stof(w[2]);
					c++;
				}
			}
		}
		file.close();

	};
	void load_solution(path filename) {

		y = torch::zeros({n}).to(torch::kBool);
		ifstream file;
		file.open(filename);
		if(file.is_open()) {
			string line;
			int c = 0;
			while(getline(file, line)) {
				y.index({c}) = stoi(line);
				c++;
			}
		}
		file.close();

	};

	static shared_ptr<struct MISGNNData> load(path graph_filename, path feature_filename, path solution_filename) {
		shared_ptr<struct MISGNNData> g = make_shared<struct MISGNNData>();

		g->load_graph(graph_filename);

		g->load_features(feature_filename);

		g->load_solution(solution_filename);

		return g;
	};


};

struct FactorThetaPredictor : torch::nn::Module {
	FactorThetaPredictor()  {
		l1 = make_shared<struct MPS>(3,64,64);
		fc1 = register_module("fc1", torch::nn::Linear(64, 2));
		fc2 = register_module("fc2", torch::nn::Linear(64, 4));

		register_module("l1", l1);

	};
	Tensor forward(Tensor& x, Tensor& edge_index, MISGNNData& data) {
		x = l1->forward(x, edge_index);

		Tensor theta;
		int v_offset=0,f_offset=0;

		Tensor theta_v = fc1->forward(x.index({data.var_mask}));
		Tensor theta_f = fc2->forward(x.index({data.var_mask.logical_not()}));

		for(int s=0;s<data.bp_edges->num_vars.size(0);s++) {
			int vn=data.bp_edges->num_vars[s].item<int>();
			int fn=data.bp_edges->num_fact[s].item<int>()-vn;
			if(theta.size(0)==0) {
				theta = torch::concatenate({
						torch::flatten(theta_v.index({Slice(v_offset,v_offset+vn),Slice()})), 
						torch::flatten(theta_f.index({Slice(f_offset,f_offset+fn),Slice()}))
						});
			}
			else {
				theta = torch::concatenate({
						theta, 
						torch::concatenate({
								torch::flatten(theta_v.index({Slice(v_offset,v_offset+vn),Slice()})), 
								torch::flatten(theta_f.index({Slice(f_offset,f_offset+fn),Slice()}))
								})
						});
			}
			v_offset += vn;
			f_offset += fn;


		}
		Tensor q = torch::zeros({data.fg->num_f2v_messages()},{x.device()});
		return loopy_belief_propagation(theta, q, *data.bp_edges, *data.fg, "sum_product", 100, 1e-4);

		//return ans;
	};
	std::shared_ptr<struct MPS>   l1{nullptr};
	torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};



int main(int argc, char ** argv) {
	torch::Device device = torch::kCPU;
	if (torch::cuda::is_available()) {
		device = torch::kCUDA;
	}

	path checkpoint_path = argv[1];
	path output_path = argv[2];


	//initialize model
	auto mnet = make_shared<struct FactorThetaPredictor>();
	mnet->to(device);
	torch::load(mnet, checkpoint_path);

	mnet->to(torch::kCPU);
	torch::save(mnet, output_path);

	return 0;

}
