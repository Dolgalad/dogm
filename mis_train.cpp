#include<iostream>
#include<string>
#include<filesystem>
#include<vector>
#include<algorithm>
#include<memory>
#include<random>
#include<chrono>

#include<torch/torch.h>

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

struct FactorGraph stack_fg(struct FactorGraph& fg1, struct FactorGraph& fg2) {
	struct FactorGraph fg;
	fg.n = fg1.n + fg2.n;
	fg.nF = fg1.nF + fg2.nF;
	fg.factor_neighbors = vector<vector<long int> >(fg1.factor_neighbors.begin(), fg1.factor_neighbors.end());
	fg.variable_neighbors = vector<vector<long int> >(fg1.variable_neighbors.begin(), fg1.variable_neighbors.end());
	for(int i=0;i<fg2.nF;i++) {
		vector<long int> fneighbors;
		for(int n : fg2.factor_neighbors[i]) fneighbors.push_back(fg1.n + n);
		fg.factor_neighbors.push_back(fneighbors);
	}
	for(int i=0;i<fg2.n;i++) {
		vector<long int> vneighbors;
		for(int n : fg2.variable_neighbors[i]) vneighbors.push_back(fg1.nF + n);
		fg.variable_neighbors.push_back(vneighbors);
	}

	return fg;
};

struct FGEdges stack_bpedges(const struct FGEdges& edges1, const struct FGEdges& edges2, struct FactorGraph& fg1, struct FactorGraph& fg2) {
	struct FGEdges edges(edges1);
	long int num_f2v_subsum = edges1.f2v_edges.index({Slice(),0}).max().item<int>()+1;
	long int num_f2v_messages_1 = fg1.num_f2v_messages();
	//long int num_f2v_messages_2 = fg2.num_f2v_messages();
	long int num_ss_1 = fg1.num_sufficient_statistics();
	long int num_ss_2 = fg2.num_sufficient_statistics();

	Tensor r_offsets = tensor({num_f2v_messages_1,num_f2v_subsum});
	Tensor q_offsets = tensor({num_f2v_messages_1,num_f2v_messages_1});
	Tensor f2v_offsets_11 = tensor({num_f2v_subsum, num_ss_1});
	Tensor f2v_offsets_12 = tensor({num_f2v_subsum, num_ss_1+num_f2v_messages_1});
	Tensor f2v_offsets_21 = tensor({num_f2v_subsum, num_ss_1});
	Tensor f2v_offsets_22 = tensor({num_f2v_subsum, num_ss_1+num_f2v_messages_1});
	Tensor m_offsets = tensor({num_ss_1, num_f2v_messages_1});
	Tensor vm_offsets = tensor({2*fg1.n, num_f2v_messages_1});

	edges.r_edges = torch::concatenate({edges.r_edges, edges2.r_edges + r_offsets}, 0);
	edges.q_edges = torch::concatenate({edges.q_edges, edges2.q_edges + q_offsets}, 0);
	Tensor tmp_f2v_edges_1 = edges.f2v_edges;
	Tensor idx0_more = (tmp_f2v_edges_1.index({Slice(),1}) >= num_ss_1);

	tmp_f2v_edges_1.index_put_({ idx0_more, Slice()}, tmp_f2v_edges_1.index({ idx0_more, Slice()}) + tensor({(long int)0, num_ss_2}));//.add_(tensor({(long int)0, num_ss_1+num_ss_2}));
	Tensor tmp_f2v_edges_2 = edges2.f2v_edges.clone();
	Tensor idx_less = (tmp_f2v_edges_2.index({Slice(),1}) < num_ss_2);
	Tensor idx_more = (tmp_f2v_edges_2.index({Slice(),1}) >= num_ss_2 );
	tmp_f2v_edges_2.index_put_({ idx_less, Slice()}, tmp_f2v_edges_2.index({ idx_less, Slice()}) + tensor({num_f2v_subsum, num_ss_1}));
	tmp_f2v_edges_2.index_put_({ idx_more, Slice()}, tmp_f2v_edges_2.index({ idx_more, Slice()}) + tensor({num_f2v_subsum, num_ss_1+num_f2v_messages_1}));

	edges.f2v_edges = torch::concatenate({tmp_f2v_edges_1, tmp_f2v_edges_2}, 0);
	edges.m_edges = torch::concatenate({edges.m_edges, edges2.m_edges+m_offsets});
	edges.vm_edges = torch::concatenate({edges.vm_edges, edges2.vm_edges+vm_offsets});

	edges.num_vars = torch::concatenate({edges1.num_vars,edges2.num_vars});
	edges.num_ss = torch::concatenate({edges1.num_ss,edges2.num_ss});
	edges.num_fact = torch::concatenate({edges1.num_fact,edges2.num_fact});

	return edges;
};


struct FactorGraph copy(const struct FactorGraph& fg) {
	struct FactorGraph nfg;
	nfg.n = fg.n;
	nfg.nF = fg.nF;
	nfg.factor_neighbors = vector<vector<long int> >(fg.factor_neighbors.begin(), fg.factor_neighbors.end());
	nfg.variable_neighbors = vector<vector<long int> >(fg.variable_neighbors.begin(), fg.variable_neighbors.end());
	return nfg;

};

struct FGEdges copy(const struct FGEdges& edges) {
	struct FGEdges e;
	e.r_edges = edges.r_edges.clone();
	e.q_edges = edges.q_edges.clone();
	e.f2v_edges = edges.f2v_edges.clone();
	e.m_edges = edges.m_edges.clone();
	e.vm_edges = edges.vm_edges.clone();
	e.num_vars = edges.num_vars.clone();
	e.num_ss = edges.num_ss.clone();
	e.num_fact = edges.num_fact.clone();
	return e;
};



struct MISGNNData {
	torch::Tensor edge_index;
	torch::Tensor x,y;
	torch::Tensor var_mask;
	torch::Tensor var_marginal_mask;
	struct FGEdges bp_edges;
	struct FactorGraph fg;
	int n,m;

	void to(torch::Device device) {
		edge_index = edge_index.to(device);
		x = x.to(device);
		y = y.to(device);
		var_mask = var_mask.to(device);
		var_marginal_mask = var_marginal_mask.to(device);
		bp_edges.to(device);
	}

	void load_graph(path filename) {
		METISGraph g = METISGraph::load(filename);
		fg = from_metis(g);
		bp_edges.make_edges(fg);
		n=g.nv(), m=g.ne();
		var_mask = torch::zeros({n+m}).to(torch::kBool);
		var_mask.index({Slice(0,n)}) = 1;
		edge_index = torch::zeros({4*m,2}).to(torch::kLong);

		int nSS = fg.num_sufficient_statistics();
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

	static struct MISGNNData load(path graph_filename, path feature_filename, path solution_filename) {
		struct MISGNNData g;
		g.load_graph(graph_filename);
		g.load_features(feature_filename);
		g.load_solution(solution_filename);
		return g;
	};


};

using Data = std::vector<pair<MISGNNData, torch::Tensor> >;
using MISGNNExample = torch::data::Example<MISGNNData, torch::Tensor>;

class MISGNNDataset : public torch::data::datasets::Dataset<MISGNNDataset, MISGNNExample > {
	using Example = MISGNNExample;
	using Batch = vector<Example>;
	using BatchRequest = c10::ArrayRef<size_t>;
	const vector<pair<MISGNNData, torch::Tensor> > data;

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
			//std::cout << "CUDA is available! Training on GPU." << std::endl;
			device = torch::kCUDA;
		}

		//cout << "in apply_batch" << endl;
		if(examples.size()==1) return examples[0]; // TODO : device
		//vector<MISGNNData> test;
		//transform(examples.begin(), examples.end(), back_inserter(test), [](const MISGNNExample &p) {return p.data;});
		//vector<Tensor> x_list,// = {examples[0].data.x},
		//		edge_index_list,// = {examples[0].data.edge_index},
		//		target_list,// = {examples[0].target},
		//		var_mask_list,// = {examples[0].data.var_mask},
		//		var_marginal_mask_list// = {examples[0].data.var_marginal_mask}
		//	;

		int x_n=0, edge_index_n=0, y_n=0, var_mask_n=0, var_marginal_mask_n=0,
		    x_offset=0, edge_index_offset=0, y_offset=0, var_mask_offset=0, var_marginal_mask_offset=0;
		int x_dim = 0;
		for(auto & ex : examples) {
			x_n += ex.data.x.size(0);
			if(x_dim == 0) x_dim = ex.data.x.size(1);
			edge_index_n += ex.data.edge_index.size(0);
			y_n += ex.data.y.size(0);
			var_mask_n += ex.data.var_mask.size(0);
			var_marginal_mask_n += ex.data.var_marginal_mask.size(0);
		}
		auto l_opts = TensorOptions().dtype(kLong).device(device);
		auto b_opts = TensorOptions().dtype(kBool).device(device);
		Tensor x = torch::zeros({x_n, x_dim}, {device}),
		       edge_index = torch::zeros({edge_index_n,2}, l_opts), 
		       target = torch::zeros({y_n}, b_opts),
		       var_mask = torch::zeros({var_mask_n}, b_opts), 
		       var_marginal_mask = torch::zeros({var_marginal_mask_n}, b_opts);
		//vector<struct FGEdges> bp_edges_list = {examples[0].data.bp_edges};
		//vector<struct FactorGraph> fg_list = {examples[0].data.fg};
		int c = 0, n = 0, m = 0;
		uint64_t vertex_offset=0;
		struct FGEdges bp_edges;
		struct FactorGraph fg;
		for(auto & ex : examples) {
			//cout << ex.data.x.sizes() << " " << ex.data.n+ex.data.m << endl;
			//x_list.push_back(ex.data.x);
			x.index_put_({Slice(x_offset, x_offset+ex.data.x.size(0)), Slice()},ex.data.x.to(device));
			//cout << "x" << endl;
			x_offset += ex.data.x.size(0);
			//cout << "edge_index" << endl;

			edge_index.index_put_({Slice(edge_index_offset, edge_index_offset+ex.data.edge_index.size(0)), Slice()}, ex.data.edge_index.to(device) + vertex_offset);
			edge_index_offset += ex.data.edge_index.size(0);
			//cout << "target" << endl;

			target.index_put_({Slice(y_offset, y_offset+ex.target.size(0))}, ex.target.to(device));
			y_offset += ex.target.size(0);
			//cout << "var_mask" << endl;

			var_mask.index_put_({Slice(var_mask_offset, var_mask_offset+ex.data.var_mask.size(0))}, ex.data.var_mask.to(device));
			var_mask_offset += ex.data.var_mask.size(0);
			//cout << "var_marginal_mask" << endl;

			var_marginal_mask.index_put_({Slice(var_marginal_mask_offset, var_marginal_mask_offset+ex.data.var_marginal_mask.size(0))}, ex.data.var_marginal_mask.to(device));
			var_marginal_mask_offset += ex.data.var_marginal_mask.size(0);




			//edge_index_list.push_back(ex.data.edge_index+vertex_offset);
			//target_list.push_back(ex.target);
			//var_mask_list.push_back(ex.data.var_mask);
			//var_marginal_mask_list.push_back(ex.data.var_marginal_mask);
			if(c==0) {
				bp_edges = copy(ex.data.bp_edges);
				fg = copy(ex.data.fg);
			} else {
				bp_edges = stack_bpedges(bp_edges, ex.data.bp_edges, fg, ex.data.fg);
				fg = stack_fg(fg, ex.data.fg);
			}
			n += ex.data.n;
			m += ex.data.m;
			vertex_offset += ex.data.edge_index.max().item<int>()+1;
			c++;
		}

		// Version 0
		//torch::Tensor x = examples[0].data.x, ei = examples[0].data.edge_index;
		//torch::Tensor y = examples[0].target;
		//torch::Tensor var_mask = examples[0].data.var_mask;
		//torch::Tensor var_marginal_mask = examples[0].data.var_marginal_mask;
		//struct FGEdges bp_edges = copy(examples[0].data.bp_edges);
		//struct FactorGraph fg = copy(examples[0].data.fg);
		//int n=examples[0].data.n, m=examples[0].data.m;
		//if(examples.size() > 1) {
		//	uint64_t vertex_offset = ei.max().item<int>()+1;
		//	for(uint64_t i=1;i<examples.size();i++) {
		//		//x = torch::concatenate({x, examples[i].data.x}, 0);
		//		x_list.push_back(examples[i].data.x);
		//		ei = torch::concatenate({ei, vertex_offset + examples[i].data.edge_index}, 0);
		//		y = torch::concatenate({y, examples[i].target}, 0);
		//		var_mask = torch::concatenate({var_mask, examples[i].data.var_mask});
		//		var_marginal_mask = torch::concatenate({var_marginal_mask, examples[i].data.var_marginal_mask});

		//		bp_edges = stack_bpedges(bp_edges, examples[i].data.bp_edges, fg, examples[i].data.fg);
		//		fg = stack_fg(fg, examples[i].data.fg);

		//		n += examples[i].data.n;
		//		m += examples[i].data.m;
		//		vertex_offset += examples[i].data.edge_index.max().item<int>() + 1;
		//	}
		//}


		MISGNNData ndata;
		ndata.edge_index = edge_index;
		//ndata.edge_index = torch::concatenate(edge_index_list, 0);
		ndata.x = x;
		//ndata.x = torch::concatenate(x_list);
		ndata.y = target;
		//ndata.y = torch::concatenate(target_list);
		ndata.var_mask = var_mask;
		//ndata.var_mask = torch::concatenate(var_mask_list);
		ndata.var_marginal_mask = var_marginal_mask;
		//ndata.var_marginal_mask = torch::concatenate(var_marginal_mask_list);
		ndata.n = n;
		ndata.m = m;
		ndata.fg = fg;
		ndata.bp_edges = bp_edges;
		//cout << "out apply batch" << endl;
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - batch_start);
		//cout << "apply batch " << duration.count() << endl;

		ndata.to(device);

		return MISGNNExample(ndata, ndata.y);
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

		for(int s=0;s<data.bp_edges.num_vars.size(0);s++) {
			int vn=data.bp_edges.num_vars[s].item<int>();
			int fn=data.bp_edges.num_fact[s].item<int>()-vn;
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
		Tensor q = torch::zeros({data.fg.num_f2v_messages()},{x.device()});
		return loopy_belief_propagation(theta, q, data.bp_edges, data.fg, "sum_product", 100, 1e-4);

		//return ans;
	};
	std::shared_ptr<struct MPS>   l1{nullptr};
	torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

vector<std::pair<MISGNNData,Tensor> > load_dataset(path dataset_path) {
	vector<std::pair<MISGNNData,Tensor> > dataset;
	for(const auto& entry : directory_iterator(dataset_path)) {
		if(endswith(entry.path().string(), ".graph")) {
			path graph_path = entry.path();
			path feature_path = replace_all(graph_path.string(),".graph", "_features.csv");
			path solution_path = replace_all(graph_path.string(),".graph", ".sol");
			MISGNNData mis = MISGNNData::load(graph_path.string(), feature_path.string(), solution_path.string());
			dataset.push_back(std::pair<MISGNNData,Tensor>(mis,mis.y));
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
	// dataset path
	//path dataset_path = "mis_er_n50-100_p0.15";
	path dataset_path = "mis_er_n5-8_p0.15";
	double lr = 1e-4;
	int batch_size = 1;
	int max_epochs = 10000;
	int num_workers = 5;
	if(argc>=2) dataset_path = path(argv[1]);
	if(argc>=3) lr = atof(argv[2]);
	if(argc>=4) batch_size = atoi(argv[3]);
	if(argc>=5) max_epochs = atoi(argv[4]);

	torch::Device device = torch::kCPU;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available! Training on GPU." << std::endl;
		device = torch::kCUDA;
	}


	path train_dataset_path = dataset_path / "train";
	path test_dataset_path = dataset_path / "test";

	// load datasets
	//vector<MISGNNData> train_data, test_data;
	vector<std::pair<MISGNNData,Tensor> > train_data, test_data;
	cout << "Loading train data..." << flush;
	train_data = load_dataset(train_dataset_path);
	cout << "done" << endl << "Loading test data..." << flush;
	test_data = load_dataset(test_dataset_path);
	cout << "done" << endl << flush;
	//
	auto train_set = MISGNNDataset(train_data).map(MISGNNStack());

	auto train_size = train_set.size().value();
	cout << "Train_size = " << train_size << endl;

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                            std::move(train_set), 
			    torch::data::DataLoaderOptions()
			    	.batch_size(batch_size)
				.workers(num_workers)
			    );
	//auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        //                    std::move(train_set), batch_size);

	//initialize model
	//auto mnet = std::make_shared<MPNet>(3,3,3);
	//auto mnet = std::make_shared<struct MPS>();

	//auto rng = std::default_random_engine {};
	auto mnet = make_shared<struct FactorThetaPredictor>();
	mnet->to(device);

	torch::optim::Adam optimizer(mnet->parameters(), /*lr=*/lr);

	path loss_file = "train_loss.txt";
	if(exists(loss_file)) remove(loss_file);


	path checkpoint_dir = "mis_train_checkpoints";
	//torch::load(mnet, "mis_train_checkpoints/checkpoint_e1.pt");
	//return 0;
	if(exists(checkpoint_dir)) remove_all(checkpoint_dir);
	create_directory(checkpoint_dir);

	for(int epoch=0;epoch<max_epochs;epoch++) {
		//std::shuffle(std::begin(train_data), std::end(train_data), rng);
		vector<float> epoch_train_losses, epoch_val_losses;

		auto train_start = high_resolution_clock::now();
		for(auto batch: *train_loader) {
			optimizer.zero_grad();

			batch.data.to(device);

			Tensor xx = batch.data.x.detach().clone();
			Tensor pred = mnet->forward(xx, batch.data.edge_index, batch.data);
			Tensor strd = torch::zeros({batch.data.bp_edges.num_ss.sum().item<int>()}).to(torch::kBool);
			//int offset = 0;
			//for(int i=0;i<batch.data.bp_edges.num_vars.size(0);i++) {
			//	int n = batch.data.bp_edges.num_vars[i].item<int>();
			//	strd.index_put_({Slice(offset,offset+2*n)}, tensor({0,1}).to(torch::kBool).repeat(n));
			//	offset += batch.data.bp_edges.num_ss[i].item<int>();
			//}
			//cout << strd.size(0) << " " << batch.data.var_marginal_mask.size(0) << " " << batch.data.bp_edges.num_ss.sum().item<int>() << endl;
			//cout << torch::all(strd.to(device) == batch.data.var_marginal_mask) << endl;

			Tensor loss = torch::binary_cross_entropy(
					//pred.index({strd.to(device)}).clamp_(0,1),
					pred.index({batch.data.var_marginal_mask}).clamp(0,1),
					batch.data.y.to(torch::kFloat).clamp(0,1)
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
		for(std::pair<MISGNNData,Tensor> & data : test_data) {
			data.first.to(device);
			Tensor xx = data.first.x.detach().clone();//.to(device);
			Tensor pred = mnet->forward(xx, data.first.edge_index, data.first);
			Tensor strd = torch::zeros({data.first.bp_edges.num_ss.sum().item<int>()}).to(kBool);
			//int offset = 0;
			//for(int i=0;i<data.first.bp_edges.num_vars.size(0);i++) {
			//	int n = data.first.bp_edges.num_vars[i].item<int>();
			//	strd.index_put_({Slice(offset,offset+2*n)}, tensor({0,1}).to(kBool).repeat(n));
			//	offset += data.first.bp_edges.num_ss[i].item<int>();
			//}


			//Tensor marge = pred.index({strd.to(device)}).clamp_(0,1);
                	Tensor loss = torch::binary_cross_entropy(
					//pred.index({strd.to(device)}).clamp_(0,1),
					pred.index({data.first.var_marginal_mask}).clamp(0,1),
					data.first.y.to(torch::kFloat).clamp(0,1)
					);
			epoch_val_losses.push_back(loss.item<float>());

		}
		auto val_stop = high_resolution_clock::now();
		auto val_duration = duration_cast<microseconds>(val_stop - val_start);

 		torch::autograd::GradMode::set_enabled(true);

		float mean_train_loss = mean<float>(epoch_train_losses), mean_val_loss = mean<float>(epoch_val_losses);
		cout << "Epoch " << epoch << " train_loss = " << mean_train_loss << " val_loss = " << mean_val_loss << " train_t = " << train_duration.count() << " val_t = " << val_duration.count() << endl;

		ofstream myfile;
		myfile.open(loss_file, ios::app);
  		myfile << epoch << "," << mean_train_loss << "," << mean_val_loss << "," << train_duration.count() << "," << val_duration.count() << endl;
  		myfile.close();

		//save checkpoint
		path checkpoint_path = checkpoint_dir / string("checkpoint_e").append(to_string(epoch)).append(".pt");
		torch::save(mnet, checkpoint_path);
	}




	return 0;
}
