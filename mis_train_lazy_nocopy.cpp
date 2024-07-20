#include<iostream>
#include<string>
#include<filesystem>
#include<vector>
#include<algorithm>
#include<memory>
#include<random>
#include<chrono>
#include<thread>
#include<utility>

#include<torch/torch.h>

#include"argparse.hpp"

#include"metis_graph.h"
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

// stack into first factor graph
void stack_fg(shared_ptr<struct FactorGraph>& fg1, shared_ptr<struct FactorGraph>& fg2) {
	//struct FactorGraph fg;
	fg1->n += fg2->n;
	fg1->nF += fg2->nF;
	for(int i=0;i<fg2->nF;i++) {
		vector<long int> fneighbors;
		for(int n : fg2->factor_neighbors[i]) fneighbors.push_back(fg1->n + n);
		fg1->factor_neighbors.push_back(fneighbors);
	}
	for(int i=0;i<fg2->n;i++) {
		vector<long int> vneighbors;
		for(int n : fg2->variable_neighbors[i]) vneighbors.push_back(fg1->nF + n);
		fg1->variable_neighbors.push_back(vneighbors);
	}
};

void print_bp_edges(const struct FGEdges& e) {
	cout << "r_edges = "   << e.r_edges.dtype()   << " " << e.r_edges.device() << endl << flush;
	cout << "q_edges = "   << e.q_edges.dtype()   << " " << e.q_edges.device() << endl << flush;
	cout << "m_edges = "   << e.m_edges.dtype()   << " " << e.m_edges.device() << endl << flush;
	cout << "f2v_edges = " << e.f2v_edges.dtype() << " " << e.f2v_edges.device() << endl << flush;
	cout << "vm_edges = "  << e.vm_edges.dtype()  << " " << e.vm_edges.device() << endl << flush;
	cout << "num_vars = "  << e.num_vars.dtype()  << " " << e.num_vars.device() << endl << flush;
	cout << "num_ss = "    << e.num_ss.dtype()    << " " << e.num_ss.device() << endl << flush;
	cout << "num_fact = "  << e.num_fact.dtype()  << " " << e.num_fact.device() << endl << flush;

}

// stack inplace
void stack_bpedges(shared_ptr<struct FGEdges>& edges1, shared_ptr<struct FGEdges>& edges2, shared_ptr<struct FactorGraph>& fg1, shared_ptr<struct FactorGraph>& fg2) {
	torch::Device device = edges1->r_edges.device();
	//struct FGEdges edges(edges1);
	long int num_f2v_subsum = edges1->f2v_edges.index({Slice(),0}).max().item<int>()+1;
	long int num_f2v_messages_1 = fg1->num_f2v_messages();
	//long int num_f2v_messages_2 = fg2.num_f2v_messages();
	long int num_ss_1 = fg1->num_sufficient_statistics();
	long int num_ss_2 = fg2->num_sufficient_statistics();

	Tensor r_offsets = tensor({num_f2v_messages_1,num_f2v_subsum}, {device}).to(torch::kLong);
	Tensor q_offsets = tensor({num_f2v_messages_1,num_f2v_messages_1}, {device}).to(torch::kLong);
	Tensor f2v_offsets_11 = tensor({num_f2v_subsum, num_ss_1}, {device});
	Tensor f2v_offsets_12 = tensor({num_f2v_subsum, num_ss_1+num_f2v_messages_1}, {device});
	Tensor f2v_offsets_21 = tensor({num_f2v_subsum, num_ss_1}, {device});
	Tensor f2v_offsets_22 = tensor({num_f2v_subsum, num_ss_1+num_f2v_messages_1}, {device});
	Tensor m_offsets = tensor({num_ss_1, num_f2v_messages_1}, {device}).to(torch::kLong);
	Tensor vm_offsets = tensor({2*fg1->n, num_f2v_messages_1}, {device}).to(torch::kLong);

	edges1->r_edges = torch::concatenate({edges1->r_edges, edges2->r_edges + r_offsets}, 0);
	edges1->q_edges = torch::concatenate({edges1->q_edges, edges2->q_edges + q_offsets}, 0);
	Tensor tmp_f2v_edges_1 = edges1->f2v_edges;
	Tensor idx0_more = (tmp_f2v_edges_1.index({Slice(),1}) >= num_ss_1);

	tmp_f2v_edges_1.index_put_({ idx0_more, Slice()}, tmp_f2v_edges_1.index({ idx0_more, Slice()}) + tensor({(long int)0, num_ss_2}, {device}).to(torch::kLong));//.add_(tensor({(long int)0, num_ss_1+num_ss_2}));

	Tensor tmp_f2v_edges_2 = edges2->f2v_edges.clone();

	Tensor idx_less = (tmp_f2v_edges_2.index({Slice(),1}) < num_ss_2);

	Tensor idx_more = (tmp_f2v_edges_2.index({Slice(),1}) >= num_ss_2 );

	tmp_f2v_edges_2.index_put_({ idx_less, Slice()}, tmp_f2v_edges_2.index({ idx_less, Slice()}) + tensor({num_f2v_subsum, num_ss_1}, {device}).to(torch::kLong));

	tmp_f2v_edges_2.index_put_({ idx_more, Slice()}, tmp_f2v_edges_2.index({ idx_more, Slice()}) + tensor({num_f2v_subsum, num_ss_1+num_f2v_messages_1}, {device}).to(torch::kLong));

	edges1->f2v_edges = torch::concatenate({tmp_f2v_edges_1, tmp_f2v_edges_2}, 0);
	edges1->m_edges = torch::concatenate({edges1->m_edges, edges2->m_edges+m_offsets});
	edges1->vm_edges = torch::concatenate({edges1->vm_edges, edges2->vm_edges+vm_offsets});

	edges1->num_vars = torch::concatenate({edges1->num_vars,edges2->num_vars});
	edges1->num_ss = torch::concatenate({edges1->num_ss,edges2->num_ss});
	edges1->num_fact = torch::concatenate({edges1->num_fact,edges2->num_fact});
	edges1->num_f2v_msg = torch::concatenate({edges1->num_f2v_msg,edges2->num_f2v_msg});


	edges1->m_stride = torch::concatenate({edges1->m_stride, edges2->m_stride});

	// sufficient statistics index
	Tensor tmp_ss_index_2 = edges2->ss_index.clone();
	tmp_ss_index_2.index_put_({edges2->v_ss_index_mask}, tmp_ss_index_2.index({edges2->v_ss_index_mask})+2*fg1->n);
	tmp_ss_index_2.index_put_({edges2->v_ss_index_mask.logical_not()}, tmp_ss_index_2.index({edges2->v_ss_index_mask.logical_not()})+2*fg1->n+4*(fg1->nF-fg1->n));
	Tensor tmp_ss_index_1 = edges1->ss_index.clone();
	tmp_ss_index_1.index_put_({edges1->v_ss_index_mask.logical_not()}, tmp_ss_index_1.index({edges1->v_ss_index_mask.logical_not()})+2*fg2->n);
	edges1->ss_index = torch::concatenate({tmp_ss_index_1, tmp_ss_index_2});
	edges1->v_ss_index_mask = torch::concatenate({edges1->v_ss_index_mask, edges2->v_ss_index_mask});
};




shared_ptr<struct FactorGraph> copy(shared_ptr<struct FactorGraph>& fg) {
	shared_ptr<struct FactorGraph> nfg = std::make_shared<struct FactorGraph>();
	nfg->n = fg->n;
	nfg->nF = fg->nF;
	nfg->factor_neighbors = vector<vector<long int> >(fg->factor_neighbors.begin(), fg->factor_neighbors.end());
	nfg->variable_neighbors = vector<vector<long int> >(fg->variable_neighbors.begin(), fg->variable_neighbors.end());
	return nfg;

};

shared_ptr<struct FGEdges> copy(shared_ptr<struct FGEdges>& edges) {
	shared_ptr<struct FGEdges> e = make_shared<struct FGEdges>();
	e->r_edges = edges->r_edges.clone();
	e->q_edges = edges->q_edges.clone();
	e->f2v_edges = edges->f2v_edges.clone();
	e->m_edges = edges->m_edges.clone();
	e->vm_edges = edges->vm_edges.clone();
	e->num_vars = edges->num_vars.clone();
	e->num_ss = edges->num_ss.clone();
	e->num_fact = edges->num_fact.clone();
	e->num_f2v_msg = edges->num_f2v_msg.clone();
	e->m_stride = edges->m_stride.clone();
	e->ss_index = edges->ss_index.clone();
	e->v_ss_index_mask = edges->v_ss_index_mask.clone();
	return e;
};



struct MISGNNData {
	torch::Tensor edge_index;
	torch::Tensor x,y,y_ind;
	torch::Tensor var_mask;
	torch::Tensor var_marginal_mask;
	shared_ptr<struct FGEdges> bp_edges;
	shared_ptr<struct FactorGraph> fg;
	int n,m;
	int n_batch;

	MISGNNData() {
		bp_edges = make_shared<struct FGEdges>();
		fg = make_shared<struct FactorGraph>();
		n_batch = 1;
	};
	MISGNNData(const struct MISGNNData& other) : edge_index(other.edge_index), x(other.x), y(other.y), var_mask(other.var_mask), var_marginal_mask(other.var_marginal_mask), bp_edges(other.bp_edges), fg(other.fg), n(other.n), m(other.m), n_batch(other.n_batch)	{
	};


	void to(torch::Device device, bool pin_memory=false) {
		edge_index = edge_index.to(device, pin_memory);
		x = x.to(device, pin_memory);
		y = y.to(device, pin_memory);
		y_ind = y_ind.to(device, pin_memory);
		var_mask = var_mask.to(device, pin_memory);
		var_marginal_mask = var_marginal_mask.to(device, pin_memory);
		bp_edges->to(device, pin_memory);
	}

	void load_graph(path filename) {
		METISGraph g = METISGraph::load(filename);
		fg = make_shared<struct FactorGraph>(from_metis(g));

		bp_edges->make_edges(*fg);

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

		y = torch::zeros({n}).to(torch::kFloat);
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
		// convert solution to factor indicators 
		y_ind = torch::zeros({fg->num_sufficient_statistics()}, {torch::kFloat});
		int pos = 0;
		for(int iF=0;iF<fg->nF;iF++) {
			int factor_size = fg->factor_size(iF);
			vector<long int> factor_neighbors = fg->factor_neighbors[iF];
			for(int v=0;v<pow(2, factor_size);v++) {
				vector<bool> vals = value_to_bool_vector(v,factor_size);
				// check indicator value
				for(int iv=0;iv<factor_size;iv++) {
					if(y[factor_neighbors[iv]].item<bool>()==vals[iv]) y_ind[pos] = 1;
					else y_ind[pos] = 0;
				}
				pos++;
			}
		}

	};

	static shared_ptr<struct MISGNNData> load(path graph_filename, path feature_filename, path solution_filename) {
		shared_ptr<struct MISGNNData> g = make_shared<struct MISGNNData>();

		g->load_graph(graph_filename);

		g->load_features(feature_filename);

		g->load_solution(solution_filename);

		return g;
	};


};

using Data = std::vector<pair<shared_ptr<MISGNNData>, torch::Tensor> >;
using MISGNNExample = torch::data::Example<shared_ptr<MISGNNData>, torch::Tensor>;

class MISGNNDataset : public torch::data::datasets::Dataset<MISGNNDataset, MISGNNExample > {
	using Example = MISGNNExample;
	using Batch = vector<Example>;
	using BatchRequest = c10::ArrayRef<size_t>;
	const vector<pair<shared_ptr<MISGNNData>, torch::Tensor> > data;

	public:
		MISGNNDataset(const Data& data) : data(data) {}
		Example get(size_t index) {
			return Example(data[index].first, data[index].second);
		}
		Batch get_batch(BatchRequest request) {
			return torch::data::datasets::Dataset<MISGNNDataset,Example>::get_batch(request);
		}
		torch::optional<size_t> size() const {
			return data.size();
		}
};

void print_fg(struct FactorGraph& fg) {
	cout << "fg.n=" << fg.n << " fg.nF=" << fg.nF << endl;
	for(int i=0;i<fg.nF;i++) {
		cout << "F=" << i << " [";
		for(size_t j=0;j<fg.factor_neighbors[i].size();j++) cout << fg.factor_neighbors[i][j] << " ";
		cout << "]" << endl;
	}
	for(int i=0;i<fg.n;i++) {
		cout << "V=" << i << " [";
		for(size_t j=0;j<fg.variable_neighbors[i].size();j++) cout << fg.variable_neighbors[i][j] << " ";
		cout << "]" << endl;
	}

}
struct MISGNNStack : public torch::data::transforms::Stack<> {
	using InputBatchType = vector<MISGNNExample>;
	using OutputBatchType = MISGNNExample;
	MISGNNStack() {};
	OutputBatchType apply_batch(InputBatchType examples) {
		auto batch_start = high_resolution_clock::now();
		torch::Device device = torch::kCPU;
		if (torch::cuda::is_available()) {
			device = torch::kCUDA;
		}

		//cout << "thread : " << std::this_thread::get_id() << " in apply_batch bs = " << examples.size() << " " << device << endl;
		if(examples.size()==1) return examples[0]; // TODO : device

		int x_n=0, edge_index_n=0, y_n=0, var_mask_n=0, var_marginal_mask_n=0,
		    x_offset=0, edge_index_offset=0, y_offset=0, var_mask_offset=0, var_marginal_mask_offset=0, y_ind_n=0, y_ind_offset=0;
		int x_dim = 0;
		for(auto & ex : examples) {

			x_n += ex.data->x.size(0);
			if(x_dim == 0) x_dim = ex.data->x.size(1);
			edge_index_n += ex.data->edge_index.size(0);
			y_n += ex.data->y.size(0);
			y_ind_n += ex.data->y_ind.size(0);
			var_mask_n += ex.data->var_mask.size(0);
			var_marginal_mask_n += ex.data->var_marginal_mask.size(0);
		}

		auto l_opts = TensorOptions().dtype(kLong).device(device);
		auto b_opts = TensorOptions().dtype(kBool).device(device);
		auto f_opts = TensorOptions().dtype(kFloat).device(device);
		Tensor x = torch::zeros({x_n, x_dim}, {device}),
		       edge_index = torch::zeros({edge_index_n,2}, l_opts), 
		       target = torch::zeros({y_n}, f_opts),
		       y_ind = torch::zeros({y_ind_n}, f_opts),
		       var_mask = torch::zeros({var_mask_n}, b_opts), 
		       var_marginal_mask = torch::zeros({var_marginal_mask_n}, b_opts);

		int c = 0, n = 0, m = 0;
		uint64_t vertex_offset=0;
		shared_ptr<struct FGEdges> bp_edges;
		shared_ptr<struct FactorGraph> fg;

		for(auto & ex : examples) {

			x.index_put_({Slice(x_offset, x_offset+ex.data->x.size(0)), Slice()},ex.data->x.to(device, true));

			x_offset += ex.data->x.size(0);

			edge_index.index_put_({Slice(edge_index_offset, edge_index_offset+ex.data->edge_index.size(0)), Slice()}, ex.data->edge_index.to(device,true) + vertex_offset);
			edge_index_offset += ex.data->edge_index.size(0);

			target.index_put_({Slice(y_offset, y_offset+ex.target.size(0))}, ex.target.to(device,true));
			y_offset += ex.target.size(0);

			y_ind.index_put_({Slice(y_ind_offset, y_ind_offset+ex.data->y_ind.size(0))}, ex.data->y_ind.to(device,true));
			y_ind_offset += ex.data->y_ind.size(0);



			var_mask.index_put_({Slice(var_mask_offset, var_mask_offset+ex.data->var_mask.size(0))}, ex.data->var_mask.to(device,true));
			var_mask_offset += ex.data->var_mask.size(0);

			var_marginal_mask.index_put_({Slice(var_marginal_mask_offset, var_marginal_mask_offset+ex.data->var_marginal_mask.size(0))}, ex.data->var_marginal_mask.to(device,true));
			var_marginal_mask_offset += ex.data->var_marginal_mask.size(0);




			if(c==0) {
				bp_edges = copy(ex.data->bp_edges);
				fg = copy(ex.data->fg);

			} else {
				stack_bpedges(bp_edges, ex.data->bp_edges, fg, ex.data->fg);
				stack_fg(fg, ex.data->fg);

			}

			n += ex.data->n;
			m += ex.data->m;
			vertex_offset += ex.data->edge_index.max().item<int>()+1;
			c++;

		}


		shared_ptr<MISGNNData> ndata = std::make_shared<MISGNNData>();

		ndata->edge_index = edge_index;
		ndata->x = x;
		ndata->y = target;
		ndata->y_ind = y_ind;
		ndata->var_mask = var_mask;
		ndata->var_marginal_mask = var_marginal_mask;
		ndata->n = n;
		ndata->m = m;
		ndata->fg = fg;
		ndata->bp_edges = bp_edges;
		ndata->n_batch = examples.size();

		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - batch_start);
		//cout << "thread : " << std::this_thread::get_id() << " apply batch " << duration.count() << " bs " << examples.size() << endl;

		ndata->to(device, true);


		return MISGNNExample(ndata, ndata->y);
	};

};

struct FactorThetaPredictor : torch::nn::Module {
	FactorThetaPredictor(int bp_max_iter_=100, double bp_tol_=1e-4)  {
		bp_max_iter = bp_max_iter_;
		bp_tol = bp_tol_;
		l1 = make_shared<struct MPS>(3,64,64);
		fc1 = register_module("fc1", torch::nn::Linear(64, 2));
		fc2 = register_module("fc2", torch::nn::Linear(64, 4));

		register_module("l1", l1);

	};
	Tensor forward(Tensor& x, Tensor& edge_index, MISGNNData& data) {
		//auto forward_start = high_resolution_clock::now();
		x = l1->forward(x, edge_index);

		//Tensor theta_v = fc1->forward(x.index({data.var_mask}));
		//Tensor theta_f = fc2->forward(x.index({data.var_mask.logical_not()}));

		Tensor theta = torch::concatenate({
				torch::flatten(fc1->forward(x.index({data.var_mask}))),
				torch::flatten(fc2->forward(x.index({data.var_mask.logical_not()})))
				}).index({data.bp_edges->ss_index});

		Tensor q = torch::zeros(theta.sizes(),{x.device()});

		Tensor r = sum_product_loopy_belief_propagation(theta, q, *data.bp_edges, *data.fg, bp_max_iter, bp_tol);
		//auto forward_stop = high_resolution_clock::now();
		//cout << torch::all(theta == theta_tmp) << endl;
		//cout << "forward time = " << duration_cast<microseconds>(forward_stop-forward_start).count() << endl;
		return r;
		//return loopy_belief_propagation(theta, q, *data.bp_edges, *data.fg, "sum_product", 100, 1e-4);
	};
	int bp_max_iter;
	double bp_tol;
	std::shared_ptr<struct MPS>   l1{nullptr};
	torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

vector<std::pair<shared_ptr<MISGNNData>,Tensor> > load_dataset(path dataset_path, torch::Device device) {
	vector<std::pair<shared_ptr<MISGNNData>,Tensor> > dataset;
	for(const auto& entry : directory_iterator(dataset_path)) {
		if(endswith(entry.path().string(), ".graph")) {
			path graph_path = entry.path();
			path feature_path = replace_all(graph_path.string(),".graph", "_features.csv");
			path solution_path = replace_all(graph_path.string(),".graph", ".sol");
			shared_ptr<MISGNNData> mis = MISGNNData::load(graph_path.string(), feature_path.string(), solution_path.string());

			mis->to(device, true);
			dataset.push_back(std::pair<shared_ptr<MISGNNData>,Tensor>(mis,mis->y));

		}
	}
	return dataset;

};

template<typename T> float mean(vector<T>& vals) {
	float ans= 0;
	for(size_t i=0;i<vals.size();i++) ans += vals[i];
	ans /= vals.size();
	return ans;

};

int main(int argc, char ** argv) {
	// argument parser
	argparse::ArgumentParser parser("mis_train");
	parser.add_argument("dataset_path")
		.help("dataset path, should contain 'train' and 'test' subdirectories");
	parser.add_argument("-l", "--lr")
		.help("learning rate (default: 1e-5)")
		.default_value(1e-5)
		.scan<'g',double>();
	parser.add_argument("-e","--epochs")
		.help("maximum number of epochs (default: 1000)")
		.default_value(1000)
		.scan<'i',int>();
	parser.add_argument("-b","--batch-size")
		.help("batch size (default: 1)")
		.default_value(1)
		.scan<'i',int>();
	parser.add_argument("-n","--num-workers")
		.help("number of workers (default: 0)")
		.default_value(0)
		.scan<'i',int>();
	parser.add_argument("-o", "--output")
		.help("output director (default: 'output')")
		.default_value("output");
	parser.add_argument("-y", "--overwrite")
		.help("overwrite existing outputs (default: false)")
		.implicit_value(true)
		.default_value(false);
	parser.add_argument("-s", "--split")
		.help("validation split (default: 0.2)")
		.default_value(0.2)
		.scan<'g',double>();
	parser.add_argument("--load-model")
		.help("load model from checkpoint (default: None)")
		.default_value("");


	try {
		parser.parse_args(argc, argv);
	} catch (const std::exception& err) {
		std::cerr << err.what() << endl;
		std::cerr << parser;
		return 1;
	}



	// dataset path
	path dataset_path = parser.get<string>("dataset_path");
	path output_dir = parser.get<string>("--output");
	path model_path = parser.get<string>("--load-model");
	double lr = parser.get<double>("--lr");
	int batch_size = parser.get<int>("--batch-size");
	int max_epochs = parser.get<int>("--epochs");
	int num_workers = parser.get<int>("--num-workers");
	bool overwrite = parser.get<bool>("--overwrite");
	double val_split = parser.get<double>("--split");

	torch::Device device = torch::kCPU;
	if (torch::cuda::is_available()) {
		device = torch::kCUDA;
	}

	path checkpoint_dir = output_dir / "checkpoints";
	path loss_file_path = output_dir / "loss.csv";

	// Training summary
	cout << "Summary" << endl;
	cout << "\tDataset path    : " << dataset_path << endl;
	cout << "\tCheckpoint dir  : " << checkpoint_dir << endl;
	cout << "\tLoss file path  : " << loss_file_path << endl;
	cout << "\tLearning rate   : " << lr << endl;
	cout << "\tBatch size      : " << batch_size << endl;
	cout << "\tMax epochs      : " << max_epochs << endl;
	cout << "\tNum workers     : " << num_workers << endl;
	cout << "\tValidation split: " << val_split << endl;
	cout << "\tDevice          : " << device << endl << endl;
	if(model_path.string().length()>0) {
		if(exists(model_path)) cout << "Loading model from '" << model_path << "'" << endl;
		else cout << "Model path '" << model_path << "' not found. Initializing new model." << endl;
	}

	//initialize model
	auto mnet = make_shared<struct FactorThetaPredictor>();
	mnet->to(device,true);

	torch::optim::Adam optimizer(mnet->parameters(), /*lr=*/lr);
	if(exists(model_path)) {
		torch::load(mnet, model_path);
	}

	// check output directory and file
	if(exists(checkpoint_dir) || exists(loss_file_path)) {
		if(!overwrite) {
			string ans;
			cout << "Outputs already exist. Overwrite ? [y/n] ";
			cin >> ans;
			if(ans.compare("y")!=0) return 2;
		}
		// remove
		if(exists(checkpoint_dir)) remove_all(checkpoint_dir);
		if(exists(loss_file_path)) remove(loss_file_path);
	}
	if(!exists(output_dir)) create_directories(output_dir);
	create_directory(checkpoint_dir);

	path train_dataset_path = dataset_path / "train";

	// load datasets
	vector<std::pair<shared_ptr<MISGNNData>,Tensor> > train_data;
	cout << "Loading train data..." << flush;
	train_data = load_dataset(train_dataset_path, device);
	cout << "done" << endl;
	// split dataset
	size_t val_n = (size_t)(val_split * train_data.size());
	vector<std::pair<shared_ptr<MISGNNData>,Tensor> > val_data(train_data.end() - val_n, train_data.end());
	train_data = vector<std::pair<shared_ptr<MISGNNData>,Tensor> >(train_data.begin(), train_data.begin()+(train_data.size() - val_n));
	
	//test_data = load_dataset(test_dataset_path, device);
	auto train_set = MISGNNDataset(train_data).map(MISGNNStack());
	cout << "Train data size     : " << train_data.size() << endl;
	cout << "Validation data size: " << val_data.size() << endl;

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                            std::move(train_set), 
			    torch::data::DataLoaderOptions()
			    	.batch_size(batch_size)
				.workers(num_workers)
			    );
	//auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        //                    std::move(train_set), batch_size);


	Tensor xx, pred, loss;
	float best_val_loss = 1e14;
	for(int epoch=0;epoch<max_epochs;epoch++) {
		//std::shuffle(std::begin(train_data), std::end(train_data), rng);
		vector<float> epoch_train_losses, epoch_val_losses;

		auto train_start = high_resolution_clock::now();
		for(auto batch: *train_loader) {
			optimizer.zero_grad();

			batch.data->to(device, true);

			xx = batch.data->x.detach().clone();

			pred = mnet->forward(xx, batch.data->edge_index, *batch.data);

			//loss = torch::binary_cross_entropy(
			//		//pred.index({strd.to(device)}).clamp_(0,1),
			//		pred.index({batch.data->var_marginal_mask}).clamp(0,1),
			//		batch.data->y.to(torch::kFloat).clamp(0,1)
			//		);
			loss = torch::binary_cross_entropy(
					pred.clamp(0,1),
					batch.data->y_ind
					);

			epoch_train_losses.push_back(loss.item<float>());
			loss.backward();

                	// Update the parameters based on the calculated gradients.
                	optimizer.step();

		}
		auto train_stop = high_resolution_clock::now();
		auto train_duration = duration_cast<microseconds>(train_stop - train_start);

 		torch::autograd::GradMode::set_enabled(false);
		auto val_start = high_resolution_clock::now();
		for(std::pair<shared_ptr<MISGNNData>,Tensor> & data : val_data) {
			data.first->to(device, true);
			xx = data.first->x.detach().clone();//.to(device);


			pred = mnet->forward(xx, data.first->edge_index, *data.first);

			loss = torch::binary_cross_entropy(
					pred.clamp(0,1),
					data.first->y_ind
					);

			epoch_val_losses.push_back(loss.item<float>());

		}
		auto val_stop = high_resolution_clock::now();
		auto val_duration = duration_cast<microseconds>(val_stop - val_start);

 		torch::autograd::GradMode::set_enabled(true);

		float mean_train_loss = mean<float>(epoch_train_losses), mean_val_loss = mean<float>(epoch_val_losses);
		cout << "Epoch " << epoch << " train_loss = " << mean_train_loss << " val_loss = " << mean_val_loss << " train_t = " << train_duration.count() << " val_t = " << val_duration.count() << endl;

		ofstream myfile;
		myfile.open(loss_file_path, ios::app);
  		myfile << epoch << "," << mean_train_loss << "," << mean_val_loss << "," << train_duration.count() << "," << val_duration.count() << endl;
  		myfile.close();

		//save checkpoint
		if(mean_val_loss < best_val_loss) {
			path checkpoint_path = checkpoint_dir / string("checkpoint_e").append(to_string(epoch)).append(".pt");
			torch::save(mnet, checkpoint_path);
			best_val_loss = mean_val_loss;
		}
	}




	return 0;
}