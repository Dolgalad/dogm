#include"belief_propagation.h"


vector<bool> value_to_bool_vector(unsigned int val, int size) {
	vector<bool> ans;
	for(int i=0;i<size;i++) {
		ans.push_back(val & (1 << i));
	}
	return ans;
};

Tensor scatter_logsumexp(const Tensor& idx, const Tensor& src, float eps=1e-12) {
	torch::Device device = idx.device();
	Tensor max_value_per_index = torch::zeros({idx.max().item<int>()+1}, {device}).scatter_reduce_(0, idx, src, "amax");
	Tensor max_per_src_element = max_value_per_index.gather(0, idx);

	Tensor recentered_scores = src - max_per_src_element;

	Tensor sum_per_index = torch::zeros_like(max_value_per_index).scatter_add_(0, idx, recentered_scores.exp());

	return (sum_per_index + eps).log() + max_value_per_index;
}

Tensor loopy_belief_propagation(Tensor& theta, Tensor& q, FGEdges& edges, FactorGraph& fg, string mode="sum_product", int max_iter=100, double tol=1e-8) {
	torch::Device device = theta.device();

	int nSS = fg.num_sufficient_statistics();
	int n_f2v_subsum = edges.f2v_edges.index({Slice(),0}).max().item<int>()+1;
	int q_n = q.size(0)/2;

	auto q_opts = TensorOptions().device(device).dtype(torch::kLong);
	Tensor q_stride = torch::full({q_n}, 2, q_opts);//.to(torch::kLong).to(theta.device());

	//Tensor q_idx = torch::repeat_interleave(torch::arange(q_stride.size(0)).to(torch::kLong),2).to(device);
	Tensor q_idx = torch::repeat_interleave(torch::arange(q_stride.size(0), q_opts),2);

	vector<long int> factor_value_sizes = fg.factor_value_sizes();

	//auto opts = torch::TensorOptions().dtype(torch::kLong).device(device);
	auto opts1 = torch::TensorOptions().dtype(torch::kLong);


	Tensor factor_sizes = torch::from_blob(fg.factor_sizes().data(), {(long int)factor_value_sizes.size()}, opts1).to(torch::kLong).to(device);

	Tensor m_stride = torch::from_blob(factor_value_sizes.data(), {(long int)factor_value_sizes.size()}, opts1).to(torch::kLong).to(device);

	Tensor m_idx = torch::repeat_interleave(torch::arange(m_stride.size(0), q_opts), m_stride);

	Tensor mu;
	if(mode.compare("sum_product")==0) mu = torch::zeros({nSS}, {device});
	if(mode.compare("max_sum")==0) mu = torch::zeros({2 * fg.n}, {device});

	// Stack
    	Tensor stack = torch::concatenate({theta,q}, 0);
	Tensor v2f_subsum, r, qbar, delta, mup, mufbar, z;
	double u;
	int t = 0;
	while(t<max_iter) {
		if(t!=0) stack.index_put_({Slice(nSS,None)}, q);

		v2f_subsum = torch::zeros({n_f2v_subsum}, {device}).scatter_add_(0, edges.f2v_edges.index({Slice(),0}), stack.index({edges.f2v_edges.index({Slice(),1})}));

		if(mode.compare("sum_product")==0) r = scatter_logsumexp(edges.r_edges.index({Slice(),0}), v2f_subsum.index({edges.r_edges.index({Slice(),1})}));
		if(mode.compare("max_sum")==0) r = torch::zeros_like(q).scatter_reduce_(0, edges.r_edges.index({Slice(),0}), v2f_subsum.index({edges.r_edges.index({Slice(),1})}),"amax");
;

        	// variable to factor messages
        	qbar = torch::zeros_like(q).scatter_add_(0, edges.q_edges.index({Slice(),0}), r.index({edges.q_edges.index({Slice(),1})}));


		//Tensor delta;
		if(mode.compare("sum_product")==0) delta = torch::repeat_interleave(scatter_logsumexp(q_idx, qbar),2);//.to(device);
		if(mode.compare("max_sum")==0) {
			delta = torch::repeat_interleave(
					torch::zeros({q_n},{device}).scatter_add_(0, q_idx, qbar) / 2,
					2);
			//delta = torch::repeat_interleave(delta, 2).to(device);
		}

        	q = qbar - delta;

        	// marginals

        	if(mode.compare("sum_product")==0) {
			mufbar = theta.clone().scatter_add_(0, edges.m_edges.index({Slice(),0}), q.index({edges.m_edges.index({Slice(),1})}));
			z = torch::repeat_interleave(scatter_logsumexp(m_idx, mufbar), m_stride);
	        	mup = (mufbar - z).exp();
		}
		if(mode.compare("max_sum")==0) {
			mup = torch::zeros_like(mu).scatter_add_(0, edges.vm_edges.index({Slice(),0}), r.index({edges.vm_edges.index({Slice(),1})}));
		}

	        u = (mup - mu).abs().max().item<float>();
	    
	        if(torch::any(torch::isinf(mup)).item<int>() || torch::any(torch::isnan(mup)).item<int>()) {
			break;
		}
	        mu = mup;
	        if(u < tol) {
			break;
		}
		t++;
	}
	//if(t == max_iter) {
	//	cout << "max iter reached  " << endl;
	//}

	if(mode.compare("max_sum")==0) {
		return torch::reshape(mu, {fg.n,2});
	}
	return mu;
}


