#ifndef GNNGRAPH_H
#define GNNGRAPH_H

#include<iostream>
#include<algorithm>
#include<vector>

#include<c10/util/ArrayRef.h>
#include<torch/torch.h>

using namespace std;



struct GNNGraph {
	GNNGraph(torch::Tensor& ei, torch::Tensor& x_) : edge_index(ei), x(x_) {};

	GNNGraph to(torch::Device device) {
		x = x.to(device);
		edge_index = edge_index.to(device);
		// TODO: this is not good
		return *this;

	};


	torch::Tensor edge_index;
	torch::Tensor x;
};

struct EdgeOffsetTracker {
	EdgeOffsetTracker() : offset(0) {};
	torch::Tensor operator()(GNNGraph& el) {
		torch::Tensor r = el.edge_index+offset;
		offset += el.edge_index.max().item<int>() + 1;
		return r;
	}

	uint64_t offset;
};



torch::Tensor make_batch(vector<GNNGraph>& examples) {
	vector<torch::Tensor> x_batch_vec, edge_index_batch_vec;
       	std::transform(examples.begin(), examples.end(), std::back_inserter(x_batch_vec), [](const GNNGraph& p) {return p.x;});
       	std::transform(examples.begin(), examples.end(), std::back_inserter(edge_index_batch_vec), EdgeOffsetTracker());

	c10::ArrayRef<torch::Tensor> x_batch_list(x_batch_vec);
	c10::ArrayRef<torch::Tensor> edge_index_batch_list(edge_index_batch_vec);
	torch::Tensor x_batch = torch::concatenate(x_batch_list, 0);
	torch::Tensor edge_index_batch = torch::concatenate(edge_index_batch_list, 0);

	return x_batch;

}


#endif
