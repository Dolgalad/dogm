#ifndef BELIEF_PROPAGATION_H
#define BELIEF_PROPAGATION_H

#include<iostream>
#include<cmath>
#include<vector>
#include<chrono>

#include<torch/torch.h>

#include"metis_graph.h"

using namespace std;
using namespace torch;
using namespace torch::indexing;

vector<bool> value_to_bool_vector(unsigned int val, int size);
Tensor scatter_logsumexp(const Tensor& idx, const Tensor& src, float eps);


struct FactorGraph {
	long int n, nF;
	vector<vector<long int> > factor_neighbors, variable_neighbors;

	FactorGraph() {};
	FactorGraph(const FactorGraph& fg) {
		n = fg.n;
		nF = fg.nF;
		factor_neighbors = vector<vector<long int> >(fg.factor_neighbors.begin(), fg.factor_neighbors.end());
		variable_neighbors = vector<vector<long int> >(fg.variable_neighbors.begin(), fg.variable_neighbors.end());

	


	};

	long int num_sufficient_statistics() {
	    long int res = 0;
	    for(long int i=0;i<nF;i++)  res += pow(2, factor_neighbors[i].size());
	    return res;
	};
	long int num_f2v_messages() {
	    long int res = 0;

	    for(long int i=0;i<nF;i++) {
		    res += 2 * factor_neighbors[i].size();
	    }
	    return res;
	};
	long int num_f2v_edges() {
            //f2v_edges_n = (2**(factor_sizes) * factor_sizes**2 ).sum()
	    long int res = 0;
	    for(long int i=0;i<nF;i++) res += pow(2, factor_neighbors[i].size()) * pow(factor_neighbors[i].size(), 2);
	    return res;
	}
	long int num_r_edges() {
	    long int res = 0;
	    for(long int i=0;i<nF;i++) res += pow(2, factor_neighbors[i].size()) * factor_neighbors[i].size();
	    return res;
	}
	long int num_q_edges() {
		long int res=0;
		for(long int i=0;i<nF;i++) {
			for(long int n: factor_neighbors[i]) {
				res += 2*(variable_neighbors[n].size()-1);
			}
		}
		return res;
	}
	long int num_m_edges() {
		long int res=0;
		for(long int i=0;i<nF;i++) res += factor_size(i) * pow(2,factor_size(i));
		return res;
	}
	long int num_vm_edges() {

		long int res = 0;
		for(long int i=0;i<n;i++) {
		       	res += 2*variable_neighbors[i].size();
		}
		return res;
	}
	vector<long int> factor_value_sizes() {
		vector<long int> sizes;
		for(long int i=0;i<nF;i++) {
			sizes.push_back(pow(2,factor_neighbors[i].size()));
		}
		return sizes;
	}

	vector<long int> factor_value_offsets() {
		vector<long int> offsets;
		for(long int i=0;i<nF;i++) {
			if(i==0) offsets.push_back(0);
			else offsets.push_back(offsets[i-1] + pow(2,factor_neighbors[i-1].size()));
		}
		return offsets;
	}
	vector<long int> factor_sizes() {
		vector<long int> sizes;
		for(long int i=0;i<nF;i++) sizes.push_back(factor_neighbors[i].size());
		return sizes;
	}

	unsigned int factor_size(int i) { return factor_neighbors[i].size(); };


};

struct FGEdges {
	Tensor r_edges, q_edges, f2v_edges, m_edges, vm_edges;
	Tensor num_vars, num_ss, num_fact;

	FGEdges() {};
	FGEdges(const struct FGEdges& fe) {
		r_edges = fe.r_edges.clone();
		q_edges = fe.q_edges.clone();
		f2v_edges = fe.f2v_edges.clone();
		m_edges = fe.m_edges.clone();
		vm_edges = fe.vm_edges.clone();
		num_vars = fe.num_vars.clone();
		num_ss = fe.num_ss.clone();
		num_fact = fe.num_fact.clone();
	};

	void to(torch::Device device) {
		r_edges = r_edges.to(device);
		q_edges = q_edges.to(device);
		f2v_edges = f2v_edges.to(device);
		m_edges = m_edges.to(device);
		vm_edges = vm_edges.to(device);
		num_vars = num_vars.to(device);
		num_ss = num_ss.to(device);
		num_fact = num_fact.to(device);
	};



	void make_edges(struct FactorGraph& fg) {
		long int nSS = fg.num_sufficient_statistics();
		vector<long int> factor_value_offsets = fg.factor_value_offsets();

		long int f2v_edges_n = fg.num_f2v_edges(), f2v_edges_c=0;
		f2v_edges = torch::zeros({f2v_edges_n,2}).to(torch::kLong);
	
		long int r_edges_n = fg.num_r_edges(), r_edges_c=0;
		r_edges = torch::zeros({r_edges_n,2}).to(torch::kLong);

		long int q_edges_n = fg.num_q_edges(), q_edges_c=0;
		q_edges = torch::zeros({q_edges_n,2}).to(torch::kLong);

		long int m_edges_n = fg.num_m_edges(), m_edges_c=0;
		m_edges = torch::zeros({m_edges_n,2}).to(torch::kLong);

		long int vm_edges_n = fg.num_vm_edges(), vm_edges_c=0;
		vm_edges = torch::zeros({vm_edges_n,2}).to(torch::kLong);

		num_vars = torch::tensor({fg.n}).to(torch::kLong);
		num_fact = torch::tensor({fg.nF}).to(torch::kLong);
		num_ss = torch::tensor({fg.num_sufficient_statistics()}).to(torch::kLong);

		for(long int iV=0;iV<fg.n;iV++) {
			vector<long int> factors = fg.variable_neighbors[iV];
			for(long unsigned int iF=0;iF<factors.size();iF++) {
				vector<long int> fvn = fg.factor_neighbors[factors[iF]];
				int iidx = find(fvn.begin(),fvn.end(), iV) - fvn.begin();
				for(int v=0;v<2;v++) {
					vm_edges.index_put_({vm_edges_c,Slice()}, tensor({2*iV+v,factor_value_offsets[iF]+2*iidx+v}));
					vm_edges_c += 1;
				}
			}
		}

		for(int iF=0;iF<fg.nF;iF++) {
			int factor_size = fg.factor_size(iF);
			vector<long int> factor_neighbors = fg.factor_neighbors[iF];
			for(long unsigned int vnii=0;vnii<factor_neighbors.size();vnii++) {
				int vneighbor = factor_neighbors[vnii];
				vector<long int> variable_neighbors = fg.variable_neighbors[vneighbor];
				for(int vni=0;vni<2;vni++) {
					for(long unsigned int fni=0;fni<variable_neighbors.size();fni++) {
						int fneighbor = variable_neighbors[fni];
						if(iF!=fneighbor) {
                        				//edges.append([f2v_idx[F,i,v], f2v_idx[Fp,i,v]])
							vector<long int> fvn = fg.factor_neighbors[fneighbor];
							int iidx = find(fvn.begin(),fvn.end(), vneighbor) - fvn.begin();
							q_edges.index_put_({q_edges_c,Slice()},tensor({(long int)(factor_value_offsets[iF]+2*vnii+vni), (long int)(factor_value_offsets[fneighbor]+2*iidx+vni)}));
							q_edges_c += 1;
						}
					}
				}
			}
			for(int v=0;v<pow(2,factor_size);v++) {
				std::vector<bool> vals = value_to_bool_vector(v, factor_size);
				for(int in=0;in<factor_size;in++) {
			                //edges.append([ss_idx[F,vals], f2v_idx[F,j,yFj]])
					m_edges.index_put_({m_edges_c,Slice()}, tensor({
								factor_value_offsets[iF]+v,
								factor_value_offsets[iF]+2*in+vals[in]
								}));
					m_edges_c += 1;

					for(int yvi=0;yvi<2;yvi++) {
						if(vals[in]==yvi) {
			                	        //redges[redges_c] = [factor_value_sizes[c_F]+2*c_i+c_v, redges_c]
							r_edges.index_put_({r_edges_c,Slice()},tensor({factor_value_offsets[iF]+2*in+yvi, r_edges_c}));
			                	        //f2v_edges[f2v_edges_c] = [redges_c, factor_value_sizes[c_F]+vals]
			                	        //f2v_edges_c += 1
							f2v_edges.index_put_({f2v_edges_c,Slice()},tensor({r_edges_c, factor_value_offsets[iF]+v}));
							f2v_edges_c += 1;
							for(int iu=0;iu<factor_size;iu++) {
								if(iu!=in) {
				        	                        //f2v_edges[f2v_edges_c] = [redges_c, nSS+factor_value_sizes[c_F]+2*c_j+yFj]
									f2v_edges.index_put_({f2v_edges_c,Slice()},tensor({r_edges_c, nSS+factor_value_offsets[iF]+2*iu+vals[iu]}));
				        	                        f2v_edges_c += 1;

								}
							}
							r_edges_c += 1;
						}
					}
				}
			}
		}
	};
};

Tensor loopy_belief_propagation(Tensor& theta, Tensor& q, FGEdges& edges, FactorGraph& fg, string mode, int max_iter, double tol);

#endif
