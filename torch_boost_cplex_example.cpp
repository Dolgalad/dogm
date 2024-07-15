#include<iostream>
#include<fstream>
#include<chrono>
#include<algorithm>
#include<random>

#include<boost/version.hpp>
#include<boost/graph/adjacency_list.hpp>
#include<boost/graph/erdos_renyi_generator.hpp>
#include<boost/random/linear_congruential.hpp>
#include<boost/graph/graph_traits.hpp>
#include<boost/graph/undirected_graph.hpp>

#include<ilcplex/cplex.h>
#include<ilcplex/ilocplex.h>

#include<torch/torch.h>


using namespace std;
using namespace boost;
using namespace torch::indexing;

//typedef boost::adjacency_list<> Graph;
//typedef boost::undirected_graph<> Graph;
//typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS> Graph;
typedef adjacency_list < setS, vecS, undirectedS> Graph;
//typedef boost::undirected_graph< boost::no_property > Graph;
typedef boost::erdos_renyi_iterator<boost::minstd_rand, Graph> ERGen;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef property_map<Graph, vertex_index_t>::type IndexMap;
typedef graph_traits<Graph>::vertex_iterator vertex_iter;
typedef adjacency_list_traits<vecS, vecS, undirectedS>::edge_parallel_category edge_parallel_category;

template<class G>
vector<int> * vertex_list(G& g) {
	vector<int> * list = new vector<int>;
        // get the property map for vertex indices
        IndexMap index = get(vertex_index, g);

        std::pair<vertex_iter, vertex_iter> vp;
        for (vp = vertices(g); vp.first != vp.second; ++vp.first) {
          Vertex v = *vp.first;
	  list->push_back(index[v]);
        }
	return list;
}

template <class G> struct neighbors {
	neighbors(G& g_) : g(g_) {}
        void operator()(Vertex v) const {
                typename graph_traits<G>::adjacency_iterator ai;
                typename graph_traits<G>::adjacency_iterator ai_end;
		typename property_map<Graph, vertex_index_t>::type
                index = get(vertex_index, g);
                std::cout << "adjacent vertices of " << index[v] << ": ";

                for (boost::tie(ai, ai_end) = adjacent_vertices(v, g);
                     ai != ai_end; ++ai)
                        std::cout << index[*ai] <<  " ";
                std::cout << std::endl;
        }
        Graph& g;
};

template <class G> struct neighbors_and_degrees {
	neighbors_and_degrees(G& g_, map<Vertex, vector<Vertex> >& neighbors_) : g(g_), neighbors(neighbors_) {}
        void operator()(Vertex v) const {
                typename graph_traits<G>::adjacency_iterator ai;
                typename graph_traits<G>::adjacency_iterator ai_end;
		//typename property_map<Graph, vertex_index_t>::type
                //index = get(vertex_index, g);
		//uint64_t d=0;
		vector<Vertex> neig;
                for (boost::tie(ai, ai_end) = adjacent_vertices(v, g);
                     ai != ai_end; ++ai)
			neig.push_back(*ai);
		neighbors[v] = neig;
        }
        Graph& g;
	map<Vertex, vector<Vertex> >& neighbors;
};

Vertex minimum_degree_vertex(vector<Vertex>& l, map<Vertex, vector<Vertex> >& neighbors) {
	uint64_t min_d = l.size();
	Vertex current_v = l[0];
	for(Vertex v : l) {
		if(neighbors[v].size() < min_d) {
			min_d = neighbors[v].size();
			current_v = v;
			if(min_d == 0) break;
		}
	}
	return current_v;
}

template<class G>
vector<Vertex> * greedy_maximum_independent_set(G& g) {
	// store the list of vertices
	//
	//typename property_map<Graph, vertex_index_t>::type
        //index = get(vertex_index, g);
	vector<Vertex> W(vertices(g).first, vertices(g).second);
	vector<Vertex> * S = new vector<Vertex>();
	map<Vertex, vector<Vertex> > neighbors;
	std::for_each(vertices(g).first, vertices(g).second,
                  neighbors_and_degrees<G>(g, neighbors));
	vector<Vertex>::iterator p;

	while(W.size() > 0) {
		Vertex v = minimum_degree_vertex(W, neighbors);
		// remove neighbors
		for(Vertex u : neighbors[v]) {
			p = std::find(W.begin(), W.end(), u);
			if(p!=W.end()) W.erase(std::find(W.begin(), W.end(), u));
		}
		p = std::find(W.begin(), W.end(), v);
		if(p!=W.end()) W.erase(std::find(W.begin(), W.end(), v));

		S->push_back(v);
	}
        return S;
}

template<class T> double mean(vector<T>& vals) {
	double c=0;
	for(T v : vals) c+=v;
	return c/vals.size();
}

uint64_t timeSinceEpochMillisec() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

struct MessagePassing : torch::nn::Module {
	MessagePassing(string aggr_="sum") : aggr(aggr_) {
	}
	volatile torch::Tensor message(torch::Tensor&, torch::Tensor&);
	volatile torch::Tensor forward(torch::Tensor&, torch::Tensor&);
	torch::Tensor aggregate(torch::Tensor& msg, torch::Tensor& index, int dim_size) {
		torch::Tensor idx = index.view({-1, 1}).expand_as(msg);
		int dim = msg.size(1);
		if(aggr.compare("sum") == 0) {
			return msg.new_zeros({dim_size,dim}).scatter_add_(0, idx, msg);
		}
		if(aggr.compare("mean") == 0) {
			torch::Tensor count = msg.new_zeros(dim_size);
            		count.scatter_add_(0, index, msg.new_ones(msg.size(0)));
            		count = count.clamp(1);
			torch::Tensor out = msg.new_zeros({dim_size,dim}).scatter_add_(0, idx, msg);
			return out / count.view({-1,1}).expand_as(out);
		}
		if(aggr.compare("min") == 0 || aggr.compare("max") == 0) {
			string aggr_ = "a"+aggr;
			return msg.new_zeros({dim_size,dim}).scatter_reduce_(0, idx, msg, aggr_, false);
		}
		return msg.new_zeros({dim_size,dim}).scatter_add_(0, idx, msg);

	}
	string aggr;
};

//struct MPNet : torch::nn::Module {
struct MPNet : MessagePassing {
	MPNet(int in_dim, int out_dim, int hidden_dim) : MessagePassing("sum") {
	  // Construct and register two Linear submodules.
	  fc1 = register_module("fc1", torch::nn::Linear(in_dim, hidden_dim));
	  fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, out_dim));
	  bias1 = register_parameter("b1", torch::randn(hidden_dim));
	  bias2 = register_parameter("b2", torch::randn(out_dim));

	  dr = torch::nn::Dropout(0.5);
	};

	torch::Tensor message(torch::Tensor& x, torch::Tensor& edge_index) {
		// compute messages along each edge
		x = fc1->forward(x) + bias1;
		x = torch::nn::functional::elu(x);
		x = (x - x.mean(0)) / (x.var(0).sqrt() + 1e-8);
		x = dr(x);
		x = fc2->forward(x) + bias2;

		//x = torch::nn::functional::elu(x);
		torch::Tensor r = x.index({edge_index.index({Slice(), 0}), Slice()});
		r = r + x.index({edge_index.index({Slice(), 1}), Slice()});
		//cout << "r = " << r << endl;
		return r;
	};

	// Implement the Net's algorithm.
	torch::Tensor forward(torch::Tensor& x, torch::Tensor& edge_index) {
 		torch::Tensor dst = edge_index.index({Slice(), 1});
		torch::Tensor msg = message(x, edge_index);
		int dim_size = edge_index.max().item<int>() + 1;
		return aggregate(msg, dst, dim_size);
	};
	
	// Use one of many "standard library" modules.
	torch::nn::Linear fc1{nullptr}, fc2{nullptr};//, fc3{nullptr};
	torch::nn::Dropout dr;
	torch::Tensor bias1, bias2;
};

struct MPS : torch::nn::Module {
	MPS()  {
		//cout << "MPS" << endl;
		l1 = make_shared<MPNet>(2, 64, 64);
		l2 = make_shared<MPNet>(64, 64, 64);
		l3 = make_shared<MPNet>(64, 64, 64);
		l4 = make_shared<MPNet>(64, 1, 64);

		register_module("l1", l1);
		register_module("l2", l2);
		register_module("l3", l3);
		register_module("l4", l4);

	};
	torch::Tensor forward(torch::Tensor& x, torch::Tensor& edge_index) {
		x = l1->forward(x, edge_index);
		x = torch::nn::functional::elu(x);
		x = (x - x.mean(0)) / (x.var(0).sqrt() + 1e-8);
		x = l2->forward(x, edge_index);
		x = torch::nn::functional::elu(x);
		x = (x - x.mean(0)) / (x.var(0).sqrt() + 1e-8);
		x = l3->forward(x, edge_index);
		x = torch::nn::functional::elu(x);
		x = (x - x.mean(0)) / (x.var(0).sqrt() + 1e-8);

		x = l4->forward(x, edge_index);
		return x;
	};
	std::shared_ptr<MPNet>   l1{nullptr}, l2{nullptr},l3{nullptr}, l4{nullptr};
};



struct MISGNNData {
	torch::Tensor edge_index;
	torch::Tensor x,y;
	torch::Tensor var_mask;

	template<class G> void from_graph(G& graph, vector<Vertex>& sol) {
		int n = boost::num_vertices(graph);
		int m = boost::num_edges(graph);
		// initialize edge_index
		this->edge_index = torch::zeros({4*m,2}).to(torch::kLong);
		// initialize features
		this->x = torch::ones({n+m,2}).to(torch::kFloat32);
		this->x.index({Slice(),1}) = torch::randn({n+m});
		this->y = torch::zeros({n,1}).to(torch::kLong);
		this->var_mask = torch::zeros({n+m}).to(torch::kBool);
		this->var_mask.index({Slice(0,n)}) = 1;

		//typename property_map<G, vertex_index_t>::type IndexMap;
		IndexMap index = get(vertex_index, graph);
		typename graph_traits<G>::edge_iterator ei, ei_end;
		int u,v,c=0;
    		for(boost::tie(ei, ei_end) = edges(graph); ei != ei_end; ++ei) {
			u = index[source(*ei,graph)];
			v = index[target(*ei,graph)];
			this->edge_index.index({4*c,0}) = u;
			this->edge_index.index({4*c,1}) = c+n;
			this->edge_index.index({4*c+1,0}) = c+n;
			this->edge_index.index({4*c+1,1}) = u;
			this->edge_index.index({4*c+2,0}) = v;
			this->edge_index.index({4*c+2,1}) = c+n;
			this->edge_index.index({4*c+3,0}) = c+n;
			this->edge_index.index({4*c+3,1}) = v;
			c++;
		}
		for(uint64_t i=0;i<sol.size();i++) this->y.index({(int)index[sol[i]],0}) = 1;
	};
};

int main() {
	std::cout << "Using Boost "
          << BOOST_VERSION / 100000     << "."  // major version
          << BOOST_VERSION / 100 % 1000 << "."  // minor version
          << BOOST_VERSION % 100                // patch level
          << std::endl;
	std::cout << "Test Erdos-Renyi graph generator with Boost" << std::endl;
        boost::minstd_rand gen;

	// Graph size : number of vertices and edge probability
	uint64_t n=50;
	float p = 0.1;
	// Dataset size
	uint64_t N_train = 1000, N_test = 100;

	vector<MISGNNData> train_dataset, test_dataset;

	cout << "Making train dataset...";
	for(uint64_t i=0;i<N_train;i++) {
		// Create graph with 100 nodes and edges with probability 0.05
		Graph g(ERGen(gen, n, p), ERGen(), n);
		// solve with greedy heuristic
		//uint64_t t0 = timeSinceEpochMillisec();
		vector<Vertex> * S = greedy_maximum_independent_set<Graph>(g);
		//uint64_t greedy_solve_time_ms = timeSinceEpochMillisec()-t0;
		//uint64_t greedy_val = S->size();
		// create instance data container
		MISGNNData data;
		data.from_graph<Graph>(g, *S);
		train_dataset.push_back(data);

		delete S;
	}
	cout << "done" << endl;

	cout << "Making test dataset...";
	for(uint64_t i=0;i<N_test;i++) {
		// Create graph with 100 nodes and edges with probability 0.05
		Graph g(ERGen(gen, n, p), ERGen(), n);
		// solve with greedy heuristic
		//uint64_t t0 = timeSinceEpochMillisec();
		vector<Vertex> * S = greedy_maximum_independent_set<Graph>(g);
		//uint64_t greedy_solve_time_ms = timeSinceEpochMillisec()-t0;
		//uint64_t greedy_val = S->size();
		// create instance data container
		MISGNNData data;
		data.from_graph<Graph>(g, *S);
		test_dataset.push_back(data);

		delete S;
	}
	cout << "done" << endl;


	// training
	uint64_t epochs = 1000;
	//auto mnet = std::make_shared<MPNet>();
	auto mnet = std::make_shared<MPS>();

	auto rng = std::default_random_engine {};

	torch::optim::Adam optimizer(mnet->parameters(), /*lr=*/0.0001);

	cout << "Starting training" << endl;
	for(uint64_t e=0;e<epochs;e++) {
		std::shuffle(std::begin(train_dataset), std::end(train_dataset), rng);
		vector<float> epoch_loss, validation_loss;
		for(MISGNNData d : train_dataset) {
			optimizer.zero_grad();
                	// Execute the model on the input data.
			//cout << d.x << endl;
			//cout << d.edge_index << endl;

                	torch::Tensor prediction = mnet->forward(d.x, d.edge_index).index({d.var_mask, Slice()});
			//cout << prediction << endl;
                	// Compute a loss value to judge the prediction of our model.
                	torch::Tensor loss = torch::nll_loss(torch::flatten(prediction.sigmoid().log()), torch::flatten(d.y));
                	// Compute gradients of the loss w.r.t. the parameters of our model.
                	loss.backward();
                	// Update the parameters based on the calculated gradients.
                	optimizer.step();
                	// Output the loss and checkpoint every 100 batches.
			epoch_loss.push_back(loss.item<float>());
		}
 		torch::autograd::GradMode::set_enabled(false);
		for(MISGNNData d : train_dataset) {
			torch::Tensor prediction = mnet->forward(d.x, d.edge_index).index({d.var_mask, Slice()});
                	torch::Tensor loss = torch::nll_loss(torch::flatten(prediction.sigmoid().log()), torch::flatten(d.y));
			validation_loss.push_back(loss.item<float>());

		}

 		torch::autograd::GradMode::set_enabled(true);

                std::cout << "Epoch: " << e << " | Loss: " << mean(epoch_loss) << " | Val loss: " << mean(validation_loss) << std::endl;
		ofstream myfile;
		myfile.open("train.txt", ios::app);
  		myfile << e << "," << mean(epoch_loss) << "," << mean(validation_loss) << "\n";
  		myfile.close();
			//cout << "--------------------------" << endl;
			//cout << mnet->parameters() << endl;
			//cout << "--------------------------" << endl;

	}

        return 0;
}

