#include<iostream>
#include<vector>
#include<filesystem>
#include<chrono>

#include<torch/torch.h>
#include"argparse.hpp"

#include"belief_propagation.h"
#include"message_passing.h"
#include"mis.h"


using namespace std;
using namespace std::filesystem;
using namespace torch;
using namespace std::chrono;

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
	std::shared_ptr<struct FGEdges> bp_edges;
	std::shared_ptr<struct FactorGraph> fg;
	int n,m;

	MISGNNData() {
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


		auto make_edges_start = high_resolution_clock::now();
		bp_edges->make_edges(*fg);
		auto make_edges_stop = high_resolution_clock::now();
		auto make_edges_duration = duration_cast<microseconds>(make_edges_stop-make_edges_start);
		cout << "make edges time = " << make_edges_duration.count() << endl;

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

	static std::shared_ptr<struct MISGNNData> load(path graph_filename, path feature_filename, path solution_filename) {
		std::shared_ptr<struct MISGNNData> g = make_shared<struct MISGNNData>();

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
	Tensor forward(Tensor& x, Tensor& edge_index, MISGNNData& data, string bp_mode="sum_product") {
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
		return loopy_belief_propagation(theta, q, *data.bp_edges, *data.fg, bp_mode, 100, 1e-4);

		//return ans;
	};
	std::shared_ptr<struct MPS>   l1{nullptr};
	torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

vector<std::pair<std::shared_ptr<MISGNNData>,Tensor> > load_dataset(path dataset_path, torch::Device device) {
	vector<std::pair<std::shared_ptr<MISGNNData>,Tensor> > dataset;
	for(const auto& entry : directory_iterator(dataset_path)) {
		if(endswith(entry.path().string(), ".graph")) {
			path graph_path = entry.path();
			path feature_path = replace_all(graph_path.string(),".graph", "_features.csv");
			path solution_path = replace_all(graph_path.string(),".graph", ".sol");
			std::shared_ptr<MISGNNData> mis = MISGNNData::load(graph_path.string(), feature_path.string(), solution_path.string());

			mis->to(device);
			dataset.push_back(std::pair<std::shared_ptr<MISGNNData>,Tensor>(mis,mis->y));

		}
	}
	return dataset;

};

Tensor mis_solve_greedy_scores(METISGraph& g, Tensor& scores) {
	Tensor ans = torch::zeros({g.nv()}).to(torch::kBool);
	Tensor queue = torch::zeros({g.nv()}).to(torch::kBool);
	while(!torch::all(queue).item<bool>()) {
		Tensor candidates = queue.logical_not().argwhere();
		int idx = torch::argmax(scores.index({queue.logical_not()})).item<int>();
		int u = candidates[idx].item<int>();
		ans[u] = 1;
		queue[u] = 1;
		// set neighbors
		for(int v : g.neighbors(u)) {
			if((queue[v]==0).item<bool>()) {
				ans[v] = 0;
				queue[v] = 1;
			}
		}
	}
	return ans;
};

template<typename T> float mean(vector<T>& vals) {
	float ans= 0;
	for(size_t i=0;i<vals.size();i++) ans += vals[i];
	ans /= vals.size();
	return ans;

};


int main(int argc, char ** argv) {
	argparse::ArgumentParser parser("mis_train");
	parser.add_argument("dataset_path")
		.help("dataset path");
	parser.add_argument("checkpoint_path")
		.help("checkpoint path");
	parser.add_argument("output")
		.help("output file");

	try {
		parser.parse_args(argc, argv);
	} catch (const std::exception& err) {
		std::cerr << err.what() << endl;
		std::cerr << parser;
		return 1;
	}

	// check device
	torch::Device device = torch::kCPU;
	if (torch::cuda::is_available()) {
		device = torch::kCUDA;
	}

	// KaMIS executable path
	path kamis_exec = "../../KaMIS/deploy/weighted_branch_reduce";


	// dataset path
	path dataset_path = parser.get<string>("dataset_path");
	path checkpoint_path = parser.get<string>("checkpoint_path");
	path output_path = parser.get<string>("output");

	if(!exists(dataset_path)) {
		cout << "Could not find dataset " << dataset_path << endl;
		return 2;
	}
	if(!exists(checkpoint_path)) {
		cout << "Could not find checkpoint " << checkpoint_path << endl;
		return 3;
	}
	if(exists(output_path)) {
		string ans;
		cout << "Output file " << output_path << " exists. Overwrite,quit,append [o,q,a] ? ";
		cin >> ans;
		if(ans.compare("o")==0) remove(output_path);
		if(ans.compare("q")==0) return 4;
	}

	// save graph scores and solutions
	path output_dir = replace_all(output_path.string(), ".csv", "");
	cout << "output_dir = " << output_dir << endl;
	if(exists(output_dir)) remove_all(output_dir);
	create_directory(output_dir);



	// summary
	cout << "Summary" << endl;
	cout << "\tDataset path   : " << dataset_path << endl;
	cout << "\tCheckpoint path: " << checkpoint_path << endl;
	cout << "\tDevice         : " << device << endl;

	// initialize model
	auto mnet = make_shared<struct FactorThetaPredictor>();
	mnet->to(device);
	torch::load(mnet, checkpoint_path);

	// deactivate autograd
	torch::autograd::GradMode::set_enabled(false);

	vector<int> prep_durations;


	for(const auto& entry : directory_iterator(dataset_path)) {
		if(endswith(entry.path().string(), ".graph")) {
			path graph_path = entry.path();
			path feature_path = replace_all(graph_path.string(),".graph", "_features.csv");
			path solution_path = replace_all(graph_path.string(),".graph", ".sol");

			path output_solution_path = output_dir / replace_all(graph_path.filename(), ".graph", ".csv");

			// load METISGraph
			auto graph_load_start = high_resolution_clock::now();
			METISGraph g = METISGraph::load(graph_path);
			auto graph_load_stop = high_resolution_clock::now();


			auto prep_start = high_resolution_clock::now();
			// solve LP relaxation of the problem
			struct MISLPSolution sol = cplex_solve_relaxed_mis(g);
			std::shared_ptr<MISGNNData> mis = MISGNNData::load(graph_path.string(), feature_path.string(), solution_path.string());
			mis->to(device);
			auto prep_stop = high_resolution_clock::now();

			// predict
			auto pred_start = high_resolution_clock::now();
			//Tensor xx = mis->x;//.to(device);
			Tensor xx = mis->x.clone();
			Tensor pred = mnet->forward(xx, mis->edge_index, *mis);
			Tensor strd = torch::zeros({mis->bp_edges->num_ss.sum().item<int>()}).to(kBool);
			Tensor scores = pred.index({mis->var_marginal_mask});
			auto pred_stop = high_resolution_clock::now();

			// predict max-sum
			auto max_sum_start = high_resolution_clock::now();
			xx = mis->x.clone();
			Tensor ms_pred = mnet->forward(xx, mis->edge_index, *mis, "max_sum").argmax(1);
			auto max_sum_stop = high_resolution_clock::now();


			// Greedy solution
			auto greedy_start = high_resolution_clock::now();
			Tensor greedy_sol = mis_solve_greedy_scores(g, scores);
			auto greedy_stop = high_resolution_clock::now();

			// simple greedy sol
			Tensor r_scores = torch::randn({scores.size(0)});
			Tensor simple_greedy = mis_solve_greedy_scores(g, r_scores);

			auto prep_duration = duration_cast<microseconds>(prep_stop-prep_start);
			prep_durations.push_back(prep_duration.count());

			auto pred_duration = duration_cast<microseconds>(pred_stop-pred_start);
			auto graph_load_duration = duration_cast<microseconds>(graph_load_stop-graph_load_start);
			auto greedy_duration = duration_cast<microseconds>(greedy_stop-greedy_start);

			// solve with KaMIS
			string cmd = kamis_exec.string();
			cmd.append(" ");
			cmd.append(graph_path.string());
			cmd.append(" > /dev/null 2>&1");
			auto kamis_start = high_resolution_clock::now();
			std::system(cmd.c_str());
			auto kamis_stop = high_resolution_clock::now();
			auto kamis_duration = duration_cast<microseconds>(kamis_stop-kamis_start);
			auto max_sum_duration = duration_cast<microseconds>(max_sum_stop-max_sum_start);


			//cout << "Graph : " << graph_path << endl;
			//cout << "Optimal cost    = " << mis->y.sum().item<int>() << endl;
			//cout << "Greedy cost     = " << greedy_sol.sum().item<int>() << endl;
			//cout << "MaxSum cost     = " << ms_pred.sum().item<int>() << endl;
			//cout << "Prep time       = " << prep_duration.count() << endl;
			//cout << "Pred time       = " << pred_duration.count() << endl;
			//cout << "Graph load time = " << graph_load_duration.count() << endl;
			//cout << "Greedy time     = " << greedy_duration.count() << endl;
			//cout << "KaMIS time      = " << kamis_duration.count() << endl;
			//cout << "MaxSum time     = " << max_sum_duration.count() << endl << endl;
			
			// output file
			ofstream output;
			output.open(output_path, ios::app);

			output << graph_path << "," << mis->y.sum().item<int>() << "," << kamis_duration.count() << "," << greedy_sol.sum().item<int>() << "," << greedy_duration.count()+prep_duration.count()+pred_duration.count() << "," << simple_greedy.sum().item<int>() << endl;
			output.close();

			// solution file
			output.open(output_solution_path, ios::app);
			for(int i=0;i<scores.size(0);i++) {
				output << greedy_sol[i].item<int>() << "," << scores[i].item<float>() << "," << simple_greedy[i].item<int>() << endl;
			}
			output.close();



			
		}

	}


	return 0;
}
