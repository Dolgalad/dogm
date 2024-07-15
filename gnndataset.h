#ifndef GNNDATASET_H
#define GNNDATASET_H

#include<vector>
#include<algorithm>

#include<torch/torch.h>

#include"gnngraph.h"

using namespace std;

using Data = std::vector<pair<GNNGraph, torch::Tensor> >;
using GNNExample = torch::data::Example<GNNGraph, torch::Tensor>;

class GNNDataset : public torch::data::datasets::Dataset<GNNDataset, GNNExample > {
	using Example = GNNExample;
	using Batch = vector<Example>;
	using BatchRequest = c10::ArrayRef<size_t>;
	const vector<pair<GNNGraph, torch::Tensor> > data;

	public:
		GNNDataset(const Data& data) : data(data) {}

		Example get(size_t index) {
			return Example(data[index].first, data[index].second);
		}

		Batch get_batch(BatchRequest request) {
			return torch::data::datasets::Dataset<GNNDataset,Example>::get_batch(request);
		}


		torch::optional<size_t> size() const {
			return data.size();
		}



};

struct GNNStack : public torch::data::transforms::Stack<> {
	using InputBatchType = vector<GNNExample>;
	using OutputBatchType = GNNExample;
	GNNStack() {};
	OutputBatchType apply_batch(InputBatchType examples) {
		vector<GNNGraph> test;
		transform(examples.begin(), examples.end(), back_inserter(test), [](const GNNExample &p) {return p.data;});
		cout << "Size test : " << test.size() << endl;
		cout << make_batch(test) << endl;
		torch::Tensor x = examples[0].data.x, ei = examples[0].data.edge_index;
		torch::Tensor y = examples[0].target;
		cout << "ei" << 0 << "\n" << ei << endl;
		if(examples.size() > 1) {
			uint64_t vertex_offset = ei.max().item<int>()+1;
			for(uint64_t i=1;i<examples.size();i++) {
				x = torch::concatenate({x, examples[i].data.x}, 0);
				ei = torch::concatenate({ei, vertex_offset + examples[i].data.edge_index}, 0);
				y = torch::concatenate({y, examples[i].target}, 0);
				vertex_offset += examples[i].data.edge_index.max().item<int>() + 1;
				cout << "ei" << i << "\n" << examples[i].data.edge_index << endl;

			}
		}
		return GNNExample(GNNGraph(ei,x), y);
	};

};


#endif
