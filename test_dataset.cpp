#include<iostream>
#include<vector>

#include<torch/torch.h>
#include<c10/util/ArrayRef.h>

#include"gnngraph.h"
#include"gnndataset.h"


int main() {
	cout << "Test Graph dataset and data loader" << endl;
	vector<pair<GNNGraph, torch::Tensor> > data;
	for(int i=0;i<4;i++) {
		torch::Tensor ei = torch::randint(0,10, {5, 2});
		ei = ei - ei.min();
		torch::Tensor x = torch::randn({ei.max().item<int>()+1,2});
		torch::Tensor y = torch::randint(0,2,{1});
		data.push_back(make_pair(GNNGraph(ei,x), y));
	}
	cout << data[0].first.edge_index.sizes() << endl;
	cout << data[0].first.x.sizes() << endl;
	cout << data[0].second.sizes() << endl;

	//auto train_set = CustomDataset(data).map(torch::data::transforms::Stack<>());
	auto train_set = GNNDataset(data).map(GNNStack());

	auto train_size = train_set.size().value();
	cout << "train_size = " << train_size << endl;
	//auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        //                    std::move(train_set), 3);
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                            std::move(train_set), 4);


	for(auto batch : *train_loader) {
		cout <<"x = \n" << batch.data.x << endl;
		cout << "edge_index = \n" << batch.data.edge_index << endl;
		cout << "y = \n" << batch.target << endl;

	}

	
	return 0;
}
