#include"belief_propagation.h"

using namespace std::chrono;

vector<bool> value_to_bool_vector(unsigned int val, int size) {
	vector<bool> ans;
	for(int i=0;i<size;i++) {
		ans.push_back(val & (1 << i));
	}
	return ans;
};

Tensor scatter_logsumexp(const Tensor& idx, const Tensor& src, float eps=1e-12, int idx_n=-1) {
	torch::Device device = idx.device();

	if (idx_n<0) idx_n = idx.max().item<int>()+1;
	//Tensor max_value_per_index = torch::zeros({idx.max().item<int>()+1}, {device}).scatter_reduce_(0, idx, src, "amax");
	//Tensor max_value_per_index = torch::empty({idx.max().item<int>()+1},{device}).scatter_reduce(0, idx, src, "amax", false); // TODO: watch out
	Tensor max_value_per_index = torch::empty({idx_n},{device}).scatter_reduce(0, idx, src, "amax", false);

	//cout << torch::all(max_value_per_index == max_value_per_index2) << endl;
	//cout << max_value_per_index << endl;
	//cout << torch::stack({max_value_per_index, max_value_per_index2, max_value_per_index - max_value_per_index2}, -1) << endl;

	Tensor max_per_src_element = max_value_per_index.gather(0, idx);

	Tensor recentered_scores = src - max_per_src_element;

	//Tensor sum_per_index = torch::zeros_like(max_value_per_index).scatter_add_(0, idx, recentered_scores.exp());
	Tensor sum_per_index = torch::empty_like(max_value_per_index).scatter_reduce_(0, idx, recentered_scores.exp(), "sum", false);

	return (sum_per_index + eps).log() + max_value_per_index;
}

// FactorGraph
FactorGraph::FactorGraph(const FactorGraph& fg) {
	//cout << "FactorGraph copy constructor" << endl << flush;
	n = fg.n;
	nF = fg.nF;
	factor_neighbors = vector<vector<long int> >(fg.factor_neighbors.begin(), fg.factor_neighbors.end());
	variable_neighbors = vector<vector<long int> >(fg.variable_neighbors.begin(), fg.variable_neighbors.end());
}

long int FactorGraph::num_sufficient_statistics() {
    long int res = 0;
    for(long int i=0;i<nF;i++)  res += pow(2, factor_neighbors[i].size());
    return res;
}
long int FactorGraph::num_f2v_messages() {
    long int res = 0;

    for(long int i=0;i<nF;i++) {
	    res += 2 * factor_neighbors[i].size();
    }
    return res;
}
long int FactorGraph::num_f2v_edges() {
    long int res = 0;
    for(long int i=0;i<nF;i++) res += pow(2, factor_neighbors[i].size()) * pow(factor_neighbors[i].size(), 2);
    return res;
}
long int FactorGraph::num_r_edges() {
    long int res = 0;
    for(long int i=0;i<nF;i++) res += pow(2, factor_neighbors[i].size()) * factor_neighbors[i].size();
    return res;
}
long int FactorGraph::num_q_edges() {
	long int res=0;
	for(long int i=0;i<nF;i++) {
		for(long int n: factor_neighbors[i]) {
			res += 2*(variable_neighbors[n].size()-1);
		}
	}
	return res;
}
long int FactorGraph::num_m_edges() {
	long int res=0;
	for(long int i=0;i<nF;i++) res += factor_size(i) * pow(2,factor_size(i));
	return res;
}
long int FactorGraph::num_vm_edges() {

	long int res = 0;
	for(long int i=0;i<n;i++) {
	       	res += 2*variable_neighbors[i].size();
	}
	return res;
}
vector<long int> FactorGraph::factor_value_sizes() {
	vector<long int> sizes;
	for(long int i=0;i<nF;i++) {
		sizes.push_back(pow(2,factor_neighbors[i].size()));
	}
	return sizes;
}

vector<long int> FactorGraph::factor_value_offsets() {
	vector<long int> offsets;
	for(long int i=0;i<nF;i++) {
		if(i==0) offsets.push_back(0);
		else offsets.push_back(offsets[i-1] + pow(2,factor_neighbors[i-1].size()));
	}
	return offsets;
}
vector<long int> FactorGraph::factor_sizes() {
	vector<long int> sizes;
	for(long int i=0;i<nF;i++) sizes.push_back(factor_neighbors[i].size());
	return sizes;
}

unsigned int FactorGraph::factor_size(int i) { return factor_neighbors[i].size(); };

// FGEdges
FGEdges::FGEdges() {
	//cout << "FGEdges new constructor" << endl;
}
FGEdges::FGEdges(const struct FGEdges& fe) {
	//cout << "FGEdges copy constructor" << endl;
	r_edges = fe.r_edges.clone();
	q_edges = fe.q_edges.clone();
	f2v_edges = fe.f2v_edges.clone();
	m_edges = fe.m_edges.clone();
	vm_edges = fe.vm_edges.clone();
	num_vars = fe.num_vars.clone();
	num_ss = fe.num_ss.clone();
	num_fact = fe.num_fact.clone();
	num_f2v_msg = fe.num_f2v_msg.clone();
	m_stride = fe.m_stride.clone();
	ss_index = fe.ss_index.clone();
	v_ss_index_mask = fe.v_ss_index_mask.clone();
}

void FGEdges::to(torch::Device& device, bool pin_memory=false) {
	r_edges = r_edges.to(device,pin_memory);
	q_edges = q_edges.to(device,pin_memory);
	f2v_edges = f2v_edges.to(device, pin_memory);
	m_edges = m_edges.to(device, pin_memory);
	vm_edges = vm_edges.to(device, pin_memory);
	num_vars = num_vars.to(device, pin_memory);
	num_ss = num_ss.to(device, pin_memory);
	num_fact = num_fact.to(device, pin_memory);
	num_f2v_msg = num_f2v_msg.to(device, pin_memory);
	m_stride = m_stride.to(device, pin_memory);
	ss_index = ss_index.to(device, pin_memory);
	v_ss_index_mask = v_ss_index_mask.to(device, pin_memory);
}



void FGEdges::make_edges(struct FactorGraph& fg) {
	long int nSS = fg.num_sufficient_statistics();
	vector<long int> factor_value_offsets = fg.factor_value_offsets();

	long int f2v_edges_n = fg.num_f2v_edges(), f2v_edges_c=0;
	vector<int> f2v_edges_vec(2*f2v_edges_n);
	//f2v_edges = torch::zeros({f2v_edges_n,2}).to(torch::kLong); // TODO: watch out for types

	long int r_edges_n = fg.num_r_edges(), r_edges_c=0;
	vector<int> r_edges_vec(2*r_edges_n);
	//r_edges = torch::zeros({r_edges_n,2}).to(torch::kLong);

	long int q_edges_n = fg.num_q_edges(), q_edges_c=0;
	vector<int> q_edges_vec(2*q_edges_n);
	//q_edges = torch::zeros({q_edges_n,2}).to(torch::kLong);

	long int m_edges_n = fg.num_m_edges(), m_edges_c=0;
	vector<int> m_edges_vec(2*m_edges_n);
	//m_edges = torch::zeros({m_edges_n,2}).to(torch::kLong);

	long int vm_edges_n = fg.num_vm_edges(), vm_edges_c=0;
	vector<int> vm_edges_vec(2*vm_edges_n);
	//vm_edges = torch::zeros({vm_edges_n,2}).to(torch::kLong);

	num_vars = torch::tensor({fg.n}).to(torch::kLong);
	num_fact = torch::tensor({fg.nF}).to(torch::kLong);
	num_ss = torch::tensor({nSS}).to(torch::kLong);
	num_f2v_msg = torch::tensor({fg.num_f2v_messages()}).to(torch::kLong);

	long int iV;
	long int iF;
	long unsigned int vnii, fni;
	int vni, in, yvi, iu, factor_size, vneighbor, fneighbor, iidx, v;
	vector<long int> * factor_neighbors; 
	vector<long int> * variable_neighbors; 
	vector<long int> * fvn;
	vector<bool> vals;

	for(iV=0;iV<fg.n;iV++) {
		vector<long int> * factors = &(fg.variable_neighbors[iV]);
		for(iF=0;iF<factors->size();iF++) {
			fvn = &(fg.factor_neighbors[factors->at(iF)]);
			iidx = find(fvn->begin(),fvn->end(), iV) - fvn->begin();
			for(v=0;v<2;v++) {
				//vm_edges.index_put_({vm_edges_c,Slice()}, tensor({2*iV+v,factor_value_offsets[iF]+2*iidx+v}));
				vm_edges_vec[2*vm_edges_c] = 2*iV+v;
				vm_edges_vec[2*vm_edges_c+1] = factor_value_offsets[iF]+2*iidx+v;
				vm_edges_c += 1;
			}
		}
	}


	for(iF=0;iF<fg.nF;iF++) {
		factor_size = fg.factor_size(iF);
		factor_neighbors = &(fg.factor_neighbors[iF]);
		for(vnii=0;vnii<factor_neighbors->size();vnii++) {
			vneighbor = factor_neighbors->at(vnii);
			variable_neighbors = &(fg.variable_neighbors[vneighbor]);
			for(vni=0;vni<2;vni++) {
				for(fni=0;fni<variable_neighbors->size();fni++) {
					fneighbor = variable_neighbors->at(fni);
					if(iF!=fneighbor) {
                				//edges.append([f2v_idx[F,i,v], f2v_idx[Fp,i,v]])
						fvn = &(fg.factor_neighbors.at(fneighbor));
						iidx = find(fvn->begin(),fvn->end(), vneighbor) - fvn->begin();

						//q_edges.index_put_({q_edges_c,Slice()},tensor({(long int)(factor_value_offsets[iF]+2*vnii+vni), (long int)(factor_value_offsets[fneighbor]+2*iidx+vni)}));
						q_edges_vec[2*q_edges_c] = factor_value_offsets[iF]+2*vnii+vni;
						q_edges_vec[2*q_edges_c+1] = factor_value_offsets[fneighbor]+2*iidx+vni;
						q_edges_c += 1;
					}
				}
			}
		}
		for(v=0;v<pow(2,factor_size);v++) {
			vals = value_to_bool_vector(v, factor_size);
			for(in=0;in<factor_size;in++) {
		                //edges.append([ss_idx[F,vals], f2v_idx[F,j,yFj]])
				//m_edges.index_put_({m_edges_c,Slice()}, tensor({
				//			factor_value_offsets[iF]+v,
				//			factor_value_offsets[iF]+2*in+vals[in]
				//			}));
				m_edges_vec[2*m_edges_c] = factor_value_offsets[iF]+v;
				m_edges_vec[2*m_edges_c+1] = factor_value_offsets[iF]+2*in+vals[in];
				m_edges_c += 1;

				for(yvi=0;yvi<2;yvi++) {
					if(vals[in]==yvi) {
						//r_edges.index_put_({r_edges_c,Slice()},tensor({factor_value_offsets[iF]+2*in+yvi, r_edges_c}));
						r_edges_vec[2*r_edges_c] = factor_value_offsets[iF]+2*in+yvi;
						r_edges_vec[2*r_edges_c+1] = r_edges_c;
						//f2v_edges.index_put_({f2v_edges_c,Slice()},tensor({r_edges_c, factor_value_offsets[iF]+v}));
						f2v_edges_vec[2*f2v_edges_c] = r_edges_c;
						f2v_edges_vec[2*f2v_edges_c+1] = factor_value_offsets[iF]+v;
						f2v_edges_c += 1;
						for(iu=0;iu<factor_size;iu++) {
							if(iu!=in) {
								//f2v_edges.index_put_({f2v_edges_c,Slice()},tensor({r_edges_c, nSS+factor_value_offsets[iF]+2*iu+vals[iu]}));
								f2v_edges_vec[2*f2v_edges_c] = r_edges_c;
								f2v_edges_vec[2*f2v_edges_c+1] = nSS+factor_value_offsets[iF]+2*iu+vals[iu];
			        	                        f2v_edges_c += 1;

							}
						}
						r_edges_c += 1;
					}
				}
			}
		}
	}
	auto f2v_opts = TensorOptions().dtype(torch::kInt32);
	f2v_edges = torch::from_blob(f2v_edges_vec.data(), {f2v_edges_n,2}, f2v_opts).to(torch::kInt64);
	r_edges = torch::from_blob(r_edges_vec.data(), {r_edges_n,2}, f2v_opts).to(torch::kInt64);
	q_edges = torch::from_blob(q_edges_vec.data(), {q_edges_n,2}, f2v_opts).to(torch::kInt64);
	m_edges = torch::from_blob(m_edges_vec.data(), {m_edges_n,2}, f2v_opts).to(torch::kInt64);
	vm_edges = torch::from_blob(vm_edges_vec.data(), {vm_edges_n,2}, f2v_opts).to(torch::kInt64);

	ss_index = torch::concatenate({
			torch::arange(2*fg.n, {torch::kLong}),
			torch::arange(2*fg.n, 2*fg.n + 4*(fg.nF-fg.n), {torch::kLong})
			});
	v_ss_index_mask = torch::zeros({2*fg.n + 4*(fg.nF-fg.n)}, {torch::kBool});
	v_ss_index_mask.index_put_({Slice(0,2*fg.n)}, v_ss_index_mask.index({Slice(0,2*fg.n)}).logical_not());

	vector<long int> factor_value_sizes = fg.factor_value_sizes();
	m_stride = torch::from_blob(factor_value_sizes.data(), {(long int)factor_value_sizes.size()}, {torch::kLong}).clone();
}


// Loopy Belief Propagation

Tensor sum_product_loopy_belief_propagation(Tensor& theta, Tensor& q, FGEdges& edges, FactorGraph& fg, int max_iter=100, double tol=1e-8) {
	torch::Device device = theta.device();

	int nSS = edges.num_ss.sum().item<int>();
	//int nVars = edges.num_vars.sum().item<int>();
	//int nNodes = (edges.num_vars + edges.num_fact).sum().item<int>();
	int n_f2v_subsum = edges.f2v_edges.index({Slice(),0}).max().item<int>()+1;
	int q_n = q.size(0)/2;

	auto q_opts = TensorOptions().device(device).dtype(torch::kLong);
	Tensor q_stride = torch::full({q_n}, 2, q_opts);//.to(torch::kLong).to(theta.device());

	Tensor q_idx = torch::repeat_interleave(torch::arange(q_stride.size(0), q_opts),2);

	Tensor m_idx = torch::repeat_interleave(torch::arange(edges.m_stride.size(0), q_opts), edges.m_stride);

	int r_edges0_max = edges.r_edges.index({Slice(),0}).max().item<int>()+1;
	int q_idx_max = q_idx.max().item<int>()+1;
	int m_idx_max = m_idx.max().item<int>()+1;

	//cout << "in sum " << nNodes << " " << edges.r_edges.index({Slice(),0}).max().item<int>() << " " << q_idx.max().item<int>() << " " << m_idx.max().item<int>() << endl;


	Tensor mu = torch::zeros({nSS}, {device});

	// Stack
    	Tensor stack = torch::concatenate({theta,q}, 0);
	Tensor v2f_subsum, r, qbar, delta, mup, mufbar, z;
	v2f_subsum = torch::empty({n_f2v_subsum},{device});
	qbar = torch::empty_like(q);
	double u;
	int t = 0;

	while(t<max_iter) {

		if(t!=0) stack.index_put_({Slice(nSS,None)}, q);

		//v2f_subsum = torch::zeros({n_f2v_subsum}, {device}).scatter_add_(0, edges.f2v_edges.index({Slice(),0}), stack.index({edges.f2v_edges.index({Slice(),1})}));
		v2f_subsum = v2f_subsum.scatter_reduce(0, edges.f2v_edges.index({Slice(),0}), stack.index({edges.f2v_edges.index({Slice(),1})}), "sum", false);


		r = scatter_logsumexp(edges.r_edges.index({Slice(),0}), v2f_subsum.index({edges.r_edges.index({Slice(),1})}), 1e-12, r_edges0_max);
        	// variable to factor messages
        	//qbar = torch::zeros_like(q).scatter_add_(0, edges.q_edges.index({Slice(),0}), r.index({edges.q_edges.index({Slice(),1})}));
        	qbar = qbar.scatter_reduce(0, edges.q_edges.index({Slice(),0}), r.index({edges.q_edges.index({Slice(),1})}), "sum", false);

		delta = torch::repeat_interleave(scatter_logsumexp(q_idx, qbar, 1e-12, q_idx_max),2);//.to(device);
        	q = qbar - delta;

        	// marginals
		//mufbar = theta.clone().scatter_add_(0, edges.m_edges.index({Slice(),0}), q.index({edges.m_edges.index({Slice(),1})}));
		mufbar = theta.scatter_add(0, edges.m_edges.index({Slice(),0}), q.index({edges.m_edges.index({Slice(),1})}));

		z = torch::repeat_interleave(scatter_logsumexp(m_idx, mufbar, 1e-12, m_idx_max), edges.m_stride);
		mup = (mufbar - z).exp();

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

	return mu;

}
Tensor loopy_belief_propagation(Tensor& theta, Tensor& q, FGEdges& edges, FactorGraph& fg, string mode="sum_product", int max_iter=100, double tol=1e-8) {
	torch::Device device = theta.device();

	//int nSS = fg.num_sufficient_statistics();
	int nSS = edges.num_ss.sum().item<int>();
	int nVars = edges.num_vars.sum().item<int>();
	//cout << "nSS = " << nSS << " " << edges.num_ss.sum() << endl;
	//cout << "nV = " << fg.n << " " << edges.num_vars.sum() << endl;
	int n_f2v_subsum = edges.f2v_edges.index({Slice(),0}).max().item<int>()+1;
	int q_n = q.size(0)/2;

	auto q_opts = TensorOptions().device(device).dtype(torch::kLong);
	Tensor q_stride = torch::full({q_n}, 2, q_opts);//.to(torch::kLong).to(theta.device());

	//Tensor q_idx = torch::repeat_interleave(torch::arange(q_stride.size(0)).to(torch::kLong),2).to(device);
	Tensor q_idx = torch::repeat_interleave(torch::arange(q_stride.size(0), q_opts),2);

	//vector<long int> factor_value_sizes = fg.factor_value_sizes();

	//auto opts = torch::TensorOptions().dtype(torch::kLong).device(device);
	//auto opts1 = torch::TensorOptions().dtype(torch::kLong);


	//Tensor factor_sizes = torch::from_blob(fg.factor_sizes().data(), {(long int)factor_value_sizes.size()}, opts1).to(torch::kLong).to(device);

	//Tensor m_stride = torch::from_blob(factor_value_sizes.data(), {(long int)factor_value_sizes.size()}, opts1).to(torch::kLong).to(device);

	//cout << "in loopy m_stride = " << m_stride << endl;
	//cout << "\tm_stride = " << edges.m_stride << endl;
	//cout << torch::all(edges.m_stride == m_stride) << endl;

	Tensor m_idx = torch::repeat_interleave(torch::arange(edges.m_stride.size(0), q_opts), edges.m_stride);

	Tensor mu;
	if(mode.compare("sum_product")==0) mu = torch::zeros({nSS}, {device});
	if(mode.compare("max_sum")==0) {
		mu = torch::zeros({2 * nVars}, {device});
	}

	// Stack
    	Tensor stack = torch::concatenate({theta,q}, 0);
	Tensor v2f_subsum, r, qbar, delta, mup, mufbar, z;
	double u;
	int t = 0;

	while(t<max_iter) {

		if(t!=0) stack.index_put_({Slice(nSS,None)}, q);

		v2f_subsum = torch::zeros({n_f2v_subsum}, {device}).scatter_add_(0, edges.f2v_edges.index({Slice(),0}), stack.index({edges.f2v_edges.index({Slice(),1})}));

		if(mode.compare("sum_product")==0) r = scatter_logsumexp(edges.r_edges.index({Slice(),0}), v2f_subsum.index({edges.r_edges.index({Slice(),1})}));
		if(mode.compare("max_sum")==0) {
			r = torch::zeros_like(q).scatter_reduce_(0, edges.r_edges.index({Slice(),0}), v2f_subsum.index({edges.r_edges.index({Slice(),1})}),"amax", false);
		}
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
			z = torch::repeat_interleave(scatter_logsumexp(m_idx, mufbar), edges.m_stride);
	        	mup = (mufbar - z).exp();
		}
		if(mode.compare("max_sum")==0) {
			mup = torch::zeros_like(mu).scatter_add_(0, edges.vm_edges.index({Slice(),0}), r.index({edges.vm_edges.index({Slice(),1})}));

		}

	        u = (mup - mu).abs().max().item<float>();
    
	        if(torch::any(torch::isinf(mup)).item<int>() || torch::any(torch::isnan(mup)).item<int>()) {
			//cout << "bad value" << endl;
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
		return torch::reshape(mu, {nVars,2});
	}
	return mu;
}


