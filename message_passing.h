#ifndef MESSAGE_PASSING_H
#define MESSAGE_PASSING_H

#include<torch/torch.h>

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
	MPS(int in_dim, int hidden_dim, int out_dim)  {
		l1 = make_shared<MPNet>(in_dim, hidden_dim, hidden_dim);
		l2 = make_shared<MPNet>(hidden_dim, hidden_dim, hidden_dim);
		l3 = make_shared<MPNet>(hidden_dim, hidden_dim, hidden_dim);
		l4 = make_shared<MPNet>(hidden_dim, out_dim, hidden_dim);

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


#endif
